from scipy.io import wavfile
from pathlib import Path
import numpy as np

from seadge import config
from seadge import room_modelling
from seadge.utils.distant_sim import animate_rir_time, animate_freqresp, sim_distant_src
from seadge.utils.files import files_in_path_recursive
from seadge.utils.log import log
from seadge.utils.scenario import Scenario, load_scenario, prepare_source
from seadge.utils.cache import make_pydantic_cache_key
from seadge.utils.wavfiles import write_wav

def gen_dynamic_rir_animation(room_cfg: config.RoomCfg, fps=30, duration_s=5, src_idx=0, *, mic=0, nfft=None, db=True):
    log.info("Generating moving RIR visualization video")
    cfg = config.get()
    N = int(duration_s * cfg.dsp.datagen_samplerate)

    # Time-domain instantaneous RIR animation
    animate_rir_time(
        room_cfg.sources[src_idx],
        N=N, fs=cfg.dsp.datagen_samplerate,
        room_cfg=room_cfg, cache_root=cfg.paths.rir_cache_dir,
        xfade_ms=cfg.dsp.rirconv_xfade_ms, normalize=cfg.dsp.rirconv_normalize,
        fps=fps, duration_s=duration_s,
        outpath=cfg.paths.debug_dir / "anim" / "rir_time.mp4",
    )

    # Frequency-response |H(f,t)| animation for one mic
    animate_freqresp(
        room_cfg.sources[src_idx],
        N=N, fs=cfg.dsp.datagen_samplerate,
        room_cfg=room_cfg, cache_root=cfg.paths.rir_cache_dir,
        xfade_ms=cfg.dsp.rirconv_xfade_ms, normalize=cfg.dsp.rirconv_normalize,
        fps=fps, duration_s=duration_s,
        mic=mic, db=db, nfft=nfft,
        outpath=cfg.paths.debug_dir / "anim" / f"rir_freq_mic{mic}.mp4",
    )

def simulate_one_scenario(
        scen: Scenario,
        room_dir: Path,
        debug_dir: Path | None,
        rir_cache_dir: Path,
        fs: int,
        xfade_ms: float,
        convmethod: str,
        normalize: str | None,
        early_ms: float,
        early_taper_ms: float,
        ) -> tuple[str, np.ndarray]:
    scen_hash = make_pydantic_cache_key(scen)
    log.debug(f"Simulating scenario {scen_hash}")
    room = config.load_room(room_dir / f"{scen.room_id}.room.json")
    clean_src_ch = [np.zeros(scen.duration_samples, dtype=float) for _ in range(len(room.sources))]

    # Simulate target speaker
    x, loc = prepare_source(scen.target_speaker, scen.duration_samples)
    clean_src_ch[loc] += x

    # Simulate other speakers
    if scen.other_sources:
        for wavsrc in scen.other_sources:
            x, loc = prepare_source(wavsrc, scen.duration_samples)
            clean_src_ch[loc] += x

    # Debug files (clean sources)
    if debug_dir:
        log.debug(f"Writing debug files for clean source mono audio (scenario: {scen_hash})")
        for i, clean in enumerate(clean_src_ch):
            path = debug_dir / scen_hash / f"clean_src{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"Writing debug file {path.relative_to(debug_dir)}")
            wavfile.write(path, fs, clean)

    # Simulate distant sources
    distant_src_ch = []
    for i, clean in enumerate(clean_src_ch):
        x = sim_distant_src(
                clean, room.sources[i],
                fs=fs, room_cfg=room,
                cache_root=rir_cache_dir,
                xfade_ms=xfade_ms,
                method=convmethod,
                normalize=normalize,
                early_ms=None,
                early_taper_ms=0.0,
                )
        distant_src_ch.append(x)

    # Debug files (distant sources)
    if debug_dir:
        log.debug(f"Writing debug files for distant source mono audio (scenario: {scen_hash})")
        for i, distant in enumerate(distant_src_ch):
            path = debug_dir / scen_hash / f"distant_src{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"Writing debug file {path.relative_to(debug_dir)}")
            wavfile.write(path, fs, distant)

    # Debug files (early source images)
    if debug_dir:
        log.debug(f"Writing debug files for early source images (scenario: {scen_hash})")
        for i, clean in enumerate(clean_src_ch):
            s = sim_distant_src(
                    clean, room.sources[i],
                    fs=fs, room_cfg=room,
                    cache_root=rir_cache_dir,
                    xfade_ms=xfade_ms,
                    method=convmethod,
                    normalize=normalize,
                    early_ms=early_ms,
                    early_taper_ms=early_taper_ms,
                    )
            path = debug_dir / scen_hash / f"early_src{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"Writing debug file {path.relative_to(debug_dir)}")
            wavfile.write(path, fs, s)

    # Mix distant sources to single
    shapes = [x.shape for x in distant_src_ch]
    out_len, M = map(max, zip(*shapes))
    y = np.zeros((out_len, M), dtype=float)
    for i, x in enumerate(distant_src_ch):
        y[:x.shape[0],:] += x

    # Normalize output
    y = (0.99 / (np.max(np.abs(y)) + 1e-12)) * y

    return scen_hash, y

def simulate_scenarios(
        scenario_dir: Path,
        outpath: Path,
        rir_cache_dir: Path,
        room_dir: Path,
        debug_dir: Path | None,
        fs: int,
        xfade_ms: float,
        convmethod: str,
        normalize: str | None,
        early_ms: float,
        early_taper_ms: float,
        ):
    scenario_files = files_in_path_recursive(scenario_dir, "*.scenario.json")
    log.info(f"Simulating {len(scenario_files)} scenarios")
    for scenfile in scenario_files:
        scen = load_scenario(scenfile)
        scen_hash, x = simulate_one_scenario(
            scen=scen,
            room_dir=room_dir,
            debug_dir=debug_dir,
            rir_cache_dir=rir_cache_dir,
            fs=fs,
            xfade_ms=xfade_ms,
            convmethod=convmethod,
            normalize=normalize,
            early_ms=early_ms,
            early_taper_ms=early_taper_ms,
        )
        write_wav(outpath / f"{scen_hash}.wav", x, fs=fs)

def main():
    cfg = config.get()

    # Generate moving RIR animation
    if cfg.debug:
        room_files = files_in_path_recursive(cfg.paths.room_dir, "*.room.json")
        if len(room_files) == 0:
            log.error(f"No room files found in directory {cfg.paths.room_dir}")
        room = config.load_room(room_files[0]) # load the first room for animation
        gen_dynamic_rir_animation(room, src_idx=0) # use the first source for animation

    simulate_scenarios(
        scenario_dir=cfg.paths.scenario_dir,
        outpath=cfg.paths.distant_dir,
        rir_cache_dir=cfg.paths.rir_cache_dir,
        room_dir=cfg.paths.room_dir,
        debug_dir=cfg.paths.debug_dir if cfg.debug else None,
        fs=cfg.dsp.datagen_samplerate,
        xfade_ms=cfg.dsp.rirconv_xfade_ms,
        convmethod=cfg.dsp.rirconv_method,
        normalize=cfg.dsp.rirconv_normalize,
        early_ms=cfg.dsp.early_ms,
        early_taper_ms=cfg.dsp.early_taper_ms,
    )
