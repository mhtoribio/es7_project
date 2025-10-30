from pathlib import Path
from typing import Optional
import fnmatch
import json
import hashlib
import math
import numpy as np

from seadge.utils.scenario import Scenario, WavSource
from seadge.utils.files import files_in_path_recursive
from seadge.utils.log import log
from seadge.utils.wavfiles import wavfile_samplerate
from seadge.utils.cache import make_pydantic_cache_key
from seadge import config

def _resampling_values(fs_from: int, fs_to: int) -> tuple[int, int]:
    """
    Compute integer (interpolation, decimation) to map fs_from -> fs_to using gcd.
    Returns (L, M) such that L/M = fs_to/fs_from and both are ints.
    """
    fs_from = int(fs_from)
    fs_to   = int(fs_to)
    if fs_from <= 0 or fs_to <= 0:
        raise ValueError("Sample rates must be positive integers")
    g = math.gcd(fs_from, fs_to)
    return (fs_to // g, fs_from // g)


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
    *,
    fs_target: int,                   # target samplerate (e.g., cfg.dsp.samplerate)
    scenario_duration: int,           # total timeline in samples (fs_target domain)
    min_speaker_volume: float,
    max_speaker_volume: float,
    num_speakers: int | None = None,  # default: len(room.sources)
    min_wavsource_duration: int | None = None,  # None -> equals scenario_duration
    rng: np.random.Generator | None = None,
) -> Optional[Scenario]:
    """
    Build a Scenario using the given room and candidate WAV files.

    Constraints:
      - No reuse of WAV files (requires len(wav_files) >= num_speakers).
      - Per-speaker volume ~ Uniform[min_speaker_volume, max_speaker_volume].
      - Random target index.
      - Each sourceloc has ≤ 1 wavsource (so no per-sourceloc overlap by construction).
      - Returns None if constraints cannot be satisfied.
    """
    # Basic checks / defaults
    if scenario_duration <= 0:
        log.error(f"gen_one_scenario: Scenario duration non-positive: {scenario_duration=}")
        return None
    if not wav_files:
        log.error("gen_one_scenario: No wav files available")
        return None
    if min_speaker_volume > max_speaker_volume:
        log.error(f"gen_one_scenario: ({min_speaker_volume=}) > ({max_speaker_volume=})")
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

        L, M = _resampling_values(fs_from, fs_target)

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

        volume = float(rng.uniform(min_speaker_volume, max_speaker_volume))

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
    others = [w for j, w in enumerate(wavsources) if j != target_idx]

    # Assemble Scenario
    scen = Scenario(
        scenario_type="speaker_mixture",
        room_id=room_hash,
        duration_samples=int(scenario_duration),
        target_speaker=target,
        other_sources=others,
    )

    # Pydantic validation will run here; if it fails, return None
    try:
        scen = Scenario.model_validate(scen.model_dump())
    except Exception:
        return None

    return scen

def gen_scenarios(room_dir: Path, outpath: Path, wav_dir: Path, scengen_cfg: config.ScenarioGenCfg, fs: int):
    room_files = files_in_path_recursive(room_dir, "*.room.json")
    wav_files = files_in_path_recursive(wav_dir, "*.wav")
    for room_path in room_files:
        room = config.load_room(room_path)
        log.info(f"Generating {scengen_cfg.scenarios_per_room} scenarios for room {make_pydantic_cache_key(room)}")
        for i in range(scengen_cfg.scenarios_per_room):
            log.debug(f"Generating scenario {i} for room {make_pydantic_cache_key(room)}")
            scen = gen_one_scenario(
                room, wav_files,
                fs_target=fs,
                scenario_duration=scengen_cfg.scenario_duration,
                min_speaker_volume=scengen_cfg.min_speaker_volume,
                max_speaker_volume=scengen_cfg.max_speaker_volume,
                num_speakers=scengen_cfg.num_speakers,
                min_wavsource_duration=scengen_cfg.effective_min_wavsource_duration,
            )
            if scen:
                scen_hash = make_pydantic_cache_key(scen)
                config.save_json(scen, outpath / f"{scen_hash}.scenario.json")
                log.debug(f"Successfully generated scenario {scen_hash}")
            else:
                log.error(f"Failed generating scenario")

def main():
    cfg = config.get()

    gen_scenarios(
        room_dir=cfg.paths.room_dir,
        outpath=cfg.paths.scenario_dir,
        wav_dir=cfg.paths.clean_dir,
        scengen_cfg=cfg.scenariogen,
        fs=cfg.dsp.datagen_samplerate,
    )
