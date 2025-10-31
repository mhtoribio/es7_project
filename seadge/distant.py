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

def gen_dynamic_rir_animation(room_cfg: config.RoomCfg, fps=30, duration_s=2, src_idx=0, *, mic=0, nfft=None, db=True):
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
        )
        write_wav(outpath / f"{scen_hash}.wav", x, fs=fs)

def main():
    ######################
    # This is just my testing. Will be removed soon. -Markus
    # This is just my testing. Will be removed soon. -Markus
    # This is just my testing. Will be removed soon. -Markus
    # This is just my testing. Will be removed soon. -Markus
    ######################
    cfg = config.get()
    # room = config.load_room("/home/markus/shit/seadge_output/rooms/63e1774e00aae38ac12ed617b2b7e22fe3f60eac.room.json") # mhtdebug
    room_modelling.main()
    # gen_dynamic_rir_animation(room, src_idx=1) # DEBUG

    # # mhtdebug
    # from scipy.io import wavfile
    # _, x = wavfile.read("/home/markus/shit/seadge_clean_data/datasets_fullband/clean_fullband/read_speech/book_00000_chp_0009_reader_06709_3_seg_2.wav")
    # x = (0.99 / 32767) * x
    # from scipy.signal import resample_poly
    # x = resample_poly(x, 1, 3)
    # import numpy as np
    # wavfile.write("/home/markus/shit/isclp-debug/x.wav", cfg.dsp.datagen_samplerate, x)
    # y = sim_distant_src(x, room.sources[0], fs=cfg.dsp.datagen_samplerate, room_cfg=room, cache_root=cfg.paths.rir_cache_dir, xfade_ms=cfg.dsp.rirconv_xfade_ms)
    # y = (32737 / (np.max(np.abs(y)) + 1e-12)) * y
    # wavfile.write("/home/markus/shit/isclp-debug/test0.wav", cfg.dsp.datagen_samplerate, y.astype(np.int16))
    # y = sim_distant_src(x, room.sources[1], fs=cfg.dsp.datagen_samplerate, room_cfg=room, cache_root=cfg.paths.rir_cache_dir, xfade_ms=cfg.dsp.rirconv_xfade_ms)
    # y = (32737 / (np.max(np.abs(y)) + 1e-12)) * y
    # wavfile.write("/home/markus/shit/isclp-debug/test1.wav", cfg.dsp.datagen_samplerate, y.astype(np.int16))

    simulate_scenarios(
        scenario_dir=cfg.paths.scenario_dir,
        outpath=cfg.paths.distant_dir,
        rir_cache_dir=cfg.paths.rir_cache_dir,
        room_dir=cfg.paths.room_dir,
        debug_dir=cfg.paths.debug_dir,
        fs=cfg.dsp.datagen_samplerate,
        xfade_ms=cfg.dsp.rirconv_xfade_ms,
        convmethod=cfg.dsp.rirconv_method,
        normalize=cfg.dsp.rirconv_normalize,
    )
