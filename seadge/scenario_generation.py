from pathlib import Path
from typing import Optional
import fnmatch
import json
import hashlib
import math
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import os

from seadge.utils.scenario import Scenario, WavSource, load_and_resample_source, prepare_source
from seadge.utils.files import files_in_path_recursive
from seadge.utils.log import log
from seadge.utils.wavfiles import wavfile_samplerate
from seadge.utils.cache import make_pydantic_cache_key
from seadge.utils.dsp import resampling_values, segment_power, gain_for_snr_db
from seadge import config


def _rel_wav_path(p: Path, base: Path) -> str:
    """
    Store path relative to clean_dir when possible; else absolute.
    """
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except Exception:
        return str(p.resolve())


def gen_one_scenario(
    room: config.RoomCfg,
    wav_files: list[Path],
    noise_files: list[Path],
    *,
    fs_target: int,                   # target samplerate (e.g., cfg.dsp.samplerate)
    scenario_duration_s: float,       # total timeline in samples (fs_target domain)
    min_interference_volume: float,
    max_interference_volume: float,
    min_snr_db: float,
    max_snr_db: float,
    num_speakers: int | None = None,  # default: len(room.sources)
    min_wavsource_duration_s: float | None = None,  # None -> equals scenario_duration
    rng: np.random.Generator | None = None,
) -> Optional[Scenario]:
    """
    Build a Scenario using the given room and candidate WAV files.

    Constraints:
      - No reuse of WAV files (requires len(wav_files) >= num_speakers).
      - Per-speaker volume ~ Uniform[min_interference_volume, max_interference_volume].
      - Random target index.
      - Each sourceloc has ≤ 1 wavsource (so no per-sourceloc overlap by construction).
      - Returns None if constraints cannot be satisfied.
    """
    scenario_duration = scenario_duration_s * fs_target
    if min_wavsource_duration_s is not None:
        min_wavsource_duration = min_wavsource_duration_s * fs_target
    # Basic checks / defaults
    if scenario_duration <= 0:
        log.error(f"gen_one_scenario: Scenario duration non-positive: {scenario_duration=}")
        return None
    if not wav_files:
        log.error("gen_one_scenario: No wav files available")
        return None
    if min_interference_volume > max_interference_volume:
        log.error(f"gen_one_scenario: ({min_interference_volume=}) > ({max_interference_volume=})")
        return None

    if num_speakers is None:
        num_speakers = len(room.sources)
    if num_speakers <= 0:
        log.error("gen_one_scenario: num_speakers <= 0")
        return None

    # Must have enough unique wav files (no reuse)
    if len(wav_files) < num_speakers:
        log.error(f"gen_one_scenario: Not enough unique wavfiles (have {len(wav_files)} want {num_speakers})")
        return None

    # Must have enough distinct sourcelocs (one per speaker)
    if len(room.sources) < num_speakers:
        log.error(f"gen_one_scenario: Not enough sources in room (have {len(room.sources)} want {num_speakers})")
        return None

    # Force durations
    if min_wavsource_duration is None:
        min_wavsource_duration = scenario_duration
    if min_wavsource_duration <= 0 or min_wavsource_duration > scenario_duration:
        log.error(f"gen_one_scenario: {min_wavsource_duration=} too short or exceeds {scenario_duration=}")
        return None

    room_hash = make_pydantic_cache_key(room)

    cfg = config.get()
    clean_base = cfg.paths.clean_dir

    rng = rng or np.random.default_rng()

    # Choose K unique files and K unique sourcelocs (indices into room.sources)
    file_idxs = rng.choice(len(wav_files), size=num_speakers, replace=False)
    chosen_files = [wav_files[i] for i in file_idxs]

    sourceloc_idxs = rng.choice(len(room.sources), size=num_speakers, replace=False)
    sourceloc_idxs = list(map(int, sourceloc_idxs))

    # Random target index
    target_idx = int(rng.integers(0, num_speakers))

    # Build wavsource items
    wavsources: list[WavSource] = []
    for path, sloc in zip(chosen_files, sourceloc_idxs):
        try:
            fs_from = int(wavfile_samplerate(path))
        except Exception:
            # If we can’t read header, fail scenario generation cleanly
            return None

        L, M = resampling_values(fs_from, fs_target)

        # Duration/Delay: since we allow at most one per sourceloc, no overlap concerns.
        # If min_wavsource_duration == scenario_duration -> delay=0, duration=scenario_duration
        if min_wavsource_duration == scenario_duration:
            dur = scenario_duration
            delay = 0
        else:
            dur = int(rng.integers(min_wavsource_duration, scenario_duration + 1))
            # ensure it fits
            max_delay = scenario_duration - dur
            if max_delay < 0:
                return None
            delay = int(rng.integers(0, max_delay + 1))

        volume = float(rng.uniform(min_interference_volume, max_interference_volume))

        wavsources.append(
            WavSource(
                wav_path=_rel_wav_path(Path(path), clean_base),
                volume=volume,
                sourceloc=int(sloc),
                delay_samples=int(delay),
                duration_samples=int(dur),
                decimation=int(M),
                interpolation=int(L),
            )
        )

    # Split target / others
    target = wavsources[target_idx]
    target.volume = 1.0 # target always at volume 1.0
    others = [w for j, w in enumerate(wavsources) if j != target_idx]

    # Add noise sources
    noise_path = rng.choice(noise_files)
    try:
        fs_from = int(wavfile_samplerate(noise_path))
    except:
        return None
    L, M = resampling_values(fs_from, fs_target)
    dur = scenario_duration
    delay = 0
    tmp = WavSource(
        wav_path=_rel_wav_path(Path(noise_path), clean_base),
        volume=1.0,
        sourceloc=0,
        delay_samples=int(delay),
        duration_samples=int(dur),
        decimation=int(M),
        interpolation=int(L),
    )
    noise_power = segment_power(load_and_resample_source(tmp))
    target_power = segment_power(load_and_resample_source(target))
    snr_db = round(rng.uniform(min_snr_db, max_snr_db))
    noise_gain = gain_for_snr_db(target_power, noise_power, snr_db)
    noise_wavsource = WavSource(
        wav_path=_rel_wav_path(Path(noise_path), clean_base),
        volume=noise_gain,
        sourceloc=0,
        delay_samples=int(delay),
        duration_samples=int(dur),
        decimation=int(M),
        interpolation=int(L),
    )
    # sanity check snr
    #SNR = 10*log10(Pt / Pn)
    noise_scaled, _ = prepare_source(noise_wavsource, int(scenario_duration))
    scaled_noise_power = segment_power(noise_scaled)
    snr_calc = 10 * np.log10(target_power/scaled_noise_power)
    log.debug(f"{snr_db=}, {snr_calc=}")
    if np.abs(snr_calc-snr_db) > 1: # 1 dB tolerance
        log.warning(f"SNRs do not match after scaling noise source (desired={snr_db}, got={snr_calc})")

    # Assemble Scenario
    scen = Scenario(
        room_id=room_hash,
        duration_samples=int(scenario_duration),
        snr_db=snr_db,
        target_speaker=target,
        interferent_speakers=others,
        noise_sources=[noise_wavsource],
    )

    # Pydantic validation will run here; if it fails, return None
    try:
        scen = Scenario.model_validate(scen.model_dump())
    except Exception:
        return None

    return scen

def _gen_scenarios_for_room(
    room_path: Path,
    wav_files: list[Path],
    noise_files: list[Path],
    scengen_cfg,
    fs: int,
    outpath: Path,
) -> int:
    """Worker: generate all scenarios for a single room."""
    room = config.load_room(room_path)
    room_key = make_pydantic_cache_key(room)

    log.debug(
        f"Generating {scengen_cfg.scenarios_per_room} scenarios "
        f"for room {room_key}"
    )

    if room.noise_source and len(noise_files) < 1:
        log.error(f"Not enough noise wav files")

    n_ok = 0
    for i in range(scengen_cfg.scenarios_per_room):
        log.debug(f"Generating scenario {i} for room {room_key}")
        scen = gen_one_scenario(
            room,
            wav_files,
            noise_files,
            fs_target=fs,
            scenario_duration_s=scengen_cfg.scenario_duration_s,
            min_interference_volume=scengen_cfg.min_interference_volume,
            max_interference_volume=scengen_cfg.max_interference_volume,
            min_snr_db=scengen_cfg.min_snr_db,
            max_snr_db=scengen_cfg.max_snr_db,
            num_speakers=scengen_cfg.num_speakers,
            min_wavsource_duration_s=scengen_cfg.effective_min_wavsource_duration_s,
        )
        if scen:
            scen_hash = make_pydantic_cache_key(scen)
            config.save_json(scen, outpath / f"{scen_hash}.scenario.json")
            log.debug(f"Successfully generated scenario {scen_hash}")
            n_ok += 1
        else:
            log.error("Failed generating scenario")

    return n_ok  # main process doesn’t strictly need this, but nice to have


def gen_scenarios(
    room_dir: Path,
    outpath: Path,
    wav_dir: Path,
    noise_dir: Path,
    scengen_cfg: config.ScenarioGenCfg,
    fs: int,
):
    room_files = list(files_in_path_recursive(room_dir, "*.room.json"))
    wav_files = list(files_in_path_recursive(wav_dir, "*.wav"))
    noise_files = list(files_in_path_recursive(noise_dir, "*.wav")) # len checked in sub function
    outpath.mkdir(parents=True, exist_ok=True)

    if not room_files:
        log.warning("No room files found, nothing to generate")
        return

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    num_processes = int(slurm_cpus) if slurm_cpus else os.cpu_count()
    log.info(f"Generating scenarios for {len(room_files)} rooms using {num_processes} workers")

    worker = partial(
        _gen_scenarios_for_room,
        wav_files=wav_files,
        noise_files=noise_files,
        scengen_cfg=scengen_cfg,
        fs=fs,
        outpath=outpath,
    )

    with Pool(processes=num_processes) as pool:
        # tqdm lives in the main process; each completed room updates it by 1
        for _ in tqdm(
            pool.imap_unordered(worker, room_files),
            total=len(room_files),
            desc="Generating scenarios for rooms",
        ):
            pass

def main():
    cfg = config.get()

    gen_scenarios(
        room_dir=cfg.paths.room_dir,
        outpath=cfg.paths.scenario_dir,
        wav_dir=cfg.paths.clean_dir,
        noise_dir=cfg.paths.noise_dir,
        scengen_cfg=cfg.scenariogen,
        fs=cfg.dsp.datagen_samplerate,
    )
