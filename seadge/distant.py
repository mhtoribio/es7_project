from scipy.io import wavfile
from scipy.signal import resample_poly
from pathlib import Path
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import os

from seadge import config
from seadge.utils import dsp
from seadge.utils.distant_sim import animate_rir_time, animate_freqresp, sim_distant_src
from seadge.utils.files import files_in_path_recursive
from seadge.utils.log import log
from seadge.utils.scenario import Scenario, load_scenario, prepare_source
from seadge.utils.cache import make_pydantic_cache_key
from seadge.utils.wavfiles import write_wav
from seadge.utils.dsp import stft, istft, resampling_values

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

def calc_y_stft_enh(y: np.ndarray, fs_from: int, fs_to: int) -> np.ndarray:
    interp, decim = resampling_values(fs_from, fs_to)
    y_resampled = resample_poly(y, interp, decim)
    y_stft = stft(y_resampled, fs_to, axis=0)
    return np.swapaxes(y_stft, 1, 2) # freqbin x frame x microphone

def calc_s_stft_enh(early_src_ch: list[np.ndarray], fs_from: int, fs_to: int) -> np.ndarray:
    interp, decim = resampling_values(fs_from, fs_to)

    # Assemble numpy array (speaker x time x microphone)
    shapes = [x.shape for x in early_src_ch]
    out_len, M = map(max, zip(*shapes))
    s = np.zeros((len(early_src_ch), out_len, M), dtype=float)
    for i, x in enumerate(early_src_ch):
        s[i, :x.shape[0],:] += x

    s_resampled = resample_poly(s, interp, decim, axis=1)
    s_stft = stft(s_resampled, fs_to, axis=1) # speaker x freqbin x microphone x frame
    return s_stft[:,:, 0, :] # dim: speaker x freqbin x frame - take mic 0 as ref mic

def simulate_one_scenario(
        scen: Scenario,
        room_dir: Path,
        debug_dir: Path | None,
        rir_cache_dir: Path,
        ml_data_dir: Path,
        fs_datagen: int,
        fs_enhancement: int,
        xfade_ms: float,
        convmethod: str,
        normalize: str | None,
        early_tmax_ms: float,
        early_offset_ms: float,
        ) -> tuple[str, np.ndarray, np.ndarray]:
    scen_hash = make_pydantic_cache_key(scen)
    log.debug(f"Simulating scenario {scen_hash}")
    room = config.load_room(room_dir / f"{scen.room_id}.room.json")
    clean_src_ch = [np.zeros(scen.duration_samples, dtype=float) for _ in range(len(room.sources))]

    # Load target speaker
    x, loc = prepare_source(scen.target_speaker, scen.duration_samples)
    clean_src_ch[loc] += x

    # Load other speakers
    if scen.interferent_speakers:
        for wavsrc in scen.interferent_speakers:
            x, loc = prepare_source(wavsrc, scen.duration_samples)
            clean_src_ch[loc] += x

    # Load noises
    clean_noise = np.zeros(scen.duration_samples, dtype=float)
    for noise_src in scen.noise_sources:
        x, _ = prepare_source(noise_src, scen.duration_samples)
        clean_noise += x

    # Debug files (clean sources)
    if debug_dir:
        log.debug(f"Writing debug files for clean source mono audio (scenario: {scen_hash})")
        for i, clean in enumerate(clean_src_ch):
            path = debug_dir / scen_hash / f"clean_src{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"Writing debug file {path.relative_to(debug_dir)}")
            wavfile.write(path, fs_datagen, clean)

        # noise file
        path = debug_dir / scen_hash / f"clean_noise.wav"
        log.debug(f"Writing debug file {path.relative_to(debug_dir)}")
        wavfile.write(path, fs_datagen, clean_noise)

    # Simulate distant sources
    distant_src_ch = []
    for i, clean in enumerate(clean_src_ch):
        x = sim_distant_src(
                clean, room.sources[i],
                fs=fs_datagen, room_cfg=room,
                cache_root=rir_cache_dir,
                xfade_ms=xfade_ms,
                method=convmethod,
                normalize=normalize,
                early_tmax_ms=None,
                early_offset_ms=0.0,
                )
        distant_src_ch.append(x)
    # noise
    if room.noise_source:
        distant_noise = sim_distant_src(
                clean_noise, room.noise_source,
                fs=fs_datagen, room_cfg=room,
                cache_root=rir_cache_dir,
                xfade_ms=xfade_ms,
                method=convmethod,
                normalize=normalize,
                early_tmax_ms=None,
                early_offset_ms=0.0,
                )
    else:
        distant_noise = np.zeros_like(distant_src_ch[0])

    # Debug files (distant sources)
    if debug_dir:
        log.debug(f"Writing debug files for distant source mono audio (scenario: {scen_hash})")
        for i, distant in enumerate(distant_src_ch):
            path = debug_dir / scen_hash / f"distant_src{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"Writing debug file {path.relative_to(debug_dir)}")
            wavfile.write(path, fs_datagen, distant)

        # noise file
        path = debug_dir / scen_hash / f"distant_noise.wav"
        log.debug(f"Writing debug file {path.relative_to(debug_dir)}")
        wavfile.write(path, fs_datagen, distant_noise)

    # Simulate early source images
    early_src_ch = []
    for i, clean in enumerate(clean_src_ch):
        s = sim_distant_src(
                clean, room.sources[i],
                fs=fs_datagen, room_cfg=room,
                cache_root=rir_cache_dir,
                xfade_ms=xfade_ms,
                method=convmethod,
                normalize=normalize,
                early_tmax_ms=early_tmax_ms,
                early_offset_ms=early_offset_ms,
                )
        early_src_ch.append(s)

    # Debug files (early source images)
    if debug_dir:
        log.debug(f"Writing debug files for early source images (scenario: {scen_hash})")
        for i, s in enumerate(early_src_ch):
            path = debug_dir / scen_hash / f"early_src{i}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"Writing debug file {path.relative_to(debug_dir)}")
            wavfile.write(path, fs_datagen, s)

    # Mix distant sources to single
    shapes = [x.shape for x in distant_src_ch] + [distant_noise.shape]
    out_len, M = map(max, zip(*shapes))
    y_gen = np.zeros((out_len, M), dtype=float)
    y_gen[:distant_noise.shape[0], :] += distant_noise # additive noise
    for i, x in enumerate(distant_src_ch):
        y_gen[:x.shape[0],:] += x

    # Normalize output
    y_gen = (0.99 / (np.max(np.abs(y_gen)) + dsp.EPS)) * y_gen

    # Calculate values for ML
    path = ml_data_dir / f"{scen_hash}.npz"
    y_stft_enh = calc_y_stft_enh(y_gen, fs_datagen, fs_enhancement)
    s_stft_enh = calc_s_stft_enh(early_src_ch, fs_datagen, fs_enhancement)
    log.debug(f"Saving PSD training data {path} ({y_stft_enh.shape=}, {s_stft_enh.shape=})")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, Y=y_stft_enh.astype(np.complex64, copy=False),
                        S_early=s_stft_enh.astype(np.complex64, copy=False))

    return scen_hash, y_gen, early_src_ch[0]

def _simulate_one_file(
    scenfile: Path,
    room_dir: Path,
    debug_dir: Path | None,
    rir_cache_dir: Path,
    ml_data_dir: Path,
    fs_datagen: int,
    fs_enhancement: int,
    xfade_ms: float,
    convmethod: str,
    normalize: str | None,
    early_tmax_ms: float,
    early_offset_ms: float,
    outpath: Path,
):
    scen = load_scenario(scenfile)
    scen_hash, y, s = simulate_one_scenario(
        scen=scen,
        room_dir=room_dir,
        debug_dir=debug_dir,
        rir_cache_dir=rir_cache_dir,
        ml_data_dir=ml_data_dir,
        fs_datagen=fs_datagen,
        fs_enhancement=fs_enhancement,
        xfade_ms=xfade_ms,
        convmethod=convmethod,
        normalize=normalize,
        early_tmax_ms=early_tmax_ms,
        early_offset_ms=early_offset_ms,
    )
    interp, decim = resampling_values(fs_datagen, fs_enhancement)
    y_resampled = resample_poly(y, interp, decim)
    write_wav(outpath / f"{scen_hash}.wav", y_resampled, fs=fs_enhancement)
    s_resampled = resample_poly(s[:,0], interp, decim)
    write_wav(outpath / f"{scen_hash}_target.wav", s_resampled, fs=fs_enhancement)

def simulate_scenarios(
    scenario_dir: Path,
    outpath: Path,
    rir_cache_dir: Path,
    room_dir: Path,
    debug_dir: Path | None,
    ml_data_dir: Path,
    fs_datagen: int,
    fs_enhancement: int,
    xfade_ms: float,
    convmethod: str,
    normalize: str | None,
    early_tmax_ms: float,
    early_offset_ms: float,
    max_scenarios: int | None,
):
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    num_processes = int(slurm_cpus) if slurm_cpus else os.cpu_count()
    scenario_files = list(files_in_path_recursive(scenario_dir, "*.scenario.json", maxnum=max_scenarios))
    log.info(f"Simulating {len(scenario_files)} scenarios with {num_processes} workers")
    outpath.mkdir(parents=True, exist_ok=True)

    worker_fn = partial(
        _simulate_one_file,
        room_dir=room_dir,
        debug_dir=debug_dir,
        rir_cache_dir=rir_cache_dir,
        ml_data_dir=ml_data_dir,
        fs_datagen=fs_datagen,
        fs_enhancement=fs_enhancement,
        xfade_ms=xfade_ms,
        convmethod=convmethod,
        normalize=normalize,
        early_tmax_ms=early_tmax_ms,
        early_offset_ms=early_offset_ms,
        outpath=outpath,
    )

    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(scenario_files), desc="Simulating scenarios") as pbar:
            # imap_unordered yields one result as each worker finishes
            for _ in pool.imap_unordered(worker_fn, scenario_files):
                pbar.update()

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
        debug_dir=cfg.paths.debug_dir/"distant" if cfg.debug else None,
        ml_data_dir=cfg.paths.ml_data_dir,
        fs_datagen=cfg.dsp.datagen_samplerate,
        fs_enhancement=cfg.dsp.enhancement_samplerate,
        xfade_ms=cfg.dsp.rirconv_xfade_ms,
        convmethod=cfg.dsp.rirconv_method,
        normalize=cfg.dsp.rirconv_normalize,
        early_tmax_ms=cfg.dsp.early_tmax_ms,
        early_offset_ms=cfg.dsp.early_offset_ms,
        max_scenarios=cfg.scenarios,
    )
