from pathlib import Path
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from scipy.io import loadmat, wavfile
from dataclasses import dataclass
import numpy as np
import os

from seadge.utils.dsp import db2pow, db2mag, stft, istft
from seadge.utils.files import files_in_path_recursive
from seadge.utils.log import log
from seadge.utils.visualization import spectrogram
from seadge.utils.scenario import load_scenario
from seadge.utils.cache import make_pydantic_cache_key
from seadge.utils.isclp_helpers.isclp_kf import build_psi_w_delta
from seadge.utils.isclp_helpers.tau import tau2forget
from seadge.utils.isclp_helpers.spatial import calc_diffcoherence, pos2steervec
from seadge.utils.isclp_helpers.desmooth_GEVD import desmooth_GEVD
from seadge.utils.isclp_helpers.estim_corrmat import estim_corrmat
from seadge.utils.isclp_helpers.estim_psd_retf import estim_psd_retf
from seadge.utils.isclp_helpers.isclp_kf import ISCLP
from seadge import config

@dataclass
class ISCLPConfig:
    fs: int # sample rate
    M: int # Number of microphones
    N: int # Number of interferent speakers
    L: int # Linear Prediction Length
    N_STFT: int # STFT length
    R_STFT: int # STFT hop size
    micpos: list[np.ndarray[tuple[int], np.dtype[np.float32]]]
    sourcepos: list[np.ndarray[tuple[int], np.dtype[np.float32]]]
    alpha_ISCLP_exp_db: float # Forgetting factor alpha exponent (1 - power)
    psi_wLP_db: float                         # LP filter variance (eq. 54)
    psi_wSC_db_range: tuple[float, float]     # SC filter variance (eq. 53)
    beta_ISCLP_db: float # smoothing
    x_tick_prop : tuple[float, float, float]
    y_tick_prop : tuple[float, float, float]
    c_range     : tuple[int, int]
    retf_thres_db : float # RETF update threshold

def enhance_isclp_kf(
        y: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        s: np.ndarray[tuple[int], np.dtype[np.float32]],
        isclp_conf: ISCLPConfig,
        debug_dir: Path | None,
        outpath: Path,
        lap_div_file: Path,
        alpha_file: Path,
        ):
    if debug_dir:
        debug_dir.mkdir(exist_ok=True, parents=True)
    # Initialization / Params
    fs = isclp_conf.fs
    N_STFT_half = isclp_conf.N_STFT // 2 + 1 # num freqbins
    N_STFT = isclp_conf.N_STFT
    R_STFT = isclp_conf.R_STFT
    L = isclp_conf.L
    M = isclp_conf.M

    # ISCLP KF
    alpha_ISCLP_KF = 1 - db2pow(isclp_conf.alpha_ISCLP_exp_db)
    A = np.sqrt(alpha_ISCLP_KF)
    psi_wLP = db2pow(isclp_conf.psi_wLP_db) # LP filter variance (eq. 54)
    psi_wSC = db2pow(np.linspace(*isclp_conf.psi_wSC_db_range, N_STFT_half))
    Psi_w_delta, Psi_w_tilde_init = build_psi_w_delta(N_STFT_half, M, L, psi_wLP, psi_wSC, alpha_ISCLP_KF)
    beta_ISCLP_KF = db2mag(isclp_conf.beta_ISCLP_db)

    # Plots
    x_tick_prop = isclp_conf.x_tick_prop
    y_tick_prop = isclp_conf.y_tick_prop
    c_range     = isclp_conf.c_range

    # SQRT PSD RETF
    zeta = tau2forget(2*M*R_STFT/fs, R_STFT, fs) # forgetting factor zeta
    try:
        tmp = loadmat(lap_div_file) # load laplace coefficients of speech per frequency bins
    except FileNotFoundError:
        log.error(f"Could not find {lap_div_file}. Maybe you didn't download it?")
        exit(-1)
    lap_div = tmp["lap_div"]
    xi_thresh = db2pow(isclp_conf.retf_thres_db)  # RETF updating threshold
    try:
        tmp = loadmat(alpha_file) 
    except FileNotFoundError:
        log.error(f"Could not find {alpha_file}. Maybe you didn't download it?")
        exit(-1)
    alpha_SQRT_PSD_RETF = tmp["alpha_sqrtMP"]
    c = 343 # speed of sound (m/s)
    Gamma = calc_diffcoherence(isclp_conf.micpos, N_STFT, fs, c, 1e-3);
    # initial RETFs
    H_init_FT = pos2steervec(isclp_conf.micpos, isclp_conf.sourcepos, N_STFT_half, fs, c);

    if M != y.shape[1] or M != len(isclp_conf.micpos):
        log.error(f"Mismatch in number of mics {M=} and data channel count {y.shape[1]=}")
        exit(-1)

    # STFT
    y_STFT = stft(y, fs, axis=0)
    s_STFT = stft(s, fs, axis=0)
    y_STFT = np.swapaxes(y_STFT, 1, 2)

    if debug_dir:
        np.save(debug_dir / "Gamma.npy", Gamma)
        spectrogram(y_STFT[:,:,0], scale="mag", x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"y_STFT.png", title=f"Microphone signal")
        spectrogram(s_STFT,        scale="mag", x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"s_STFT.png", title=f"Target speech")

    #### Early PSD estimation, RETF update
    # correlation matrix of microphone signal
    Psi_y_smth, Psi_y_mean = estim_corrmat(y_STFT, zeta);
    # compute GEVD
    P_STFT, lambda_STFT, _ = desmooth_GEVD(Psi_y_smth, Gamma, 0, zeta)

    phi_s_hat, phi_xl_hat, H_hat_prior_STFT, H_hat_post_STFT, H_update_pattern = estim_psd_retf(P_STFT, lambda_STFT, Gamma, H_init_FT,
                   alpha=alpha_SQRT_PSD_RETF,
                   beta=20*lap_div*lap_div,
                   xi_thresh=xi_thresh)

    if debug_dir:
        spectrogram(phi_s_hat[:,:,0], scale='pow', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"phi_s_hat.png", title=f"Target PSD estimate")

    #### ISCLP-KF
    numFrames = y_STFT.shape[1]
    # init outputs
    q_STFT              = np.zeros((N_STFT_half, numFrames), dtype=np.complex128);
    e_prio_STFT         = np.zeros((N_STFT_half, numFrames), dtype=np.complex128);
    e_post_STFT         = np.zeros((N_STFT_half, numFrames), dtype=np.complex128);
    e_post_smooth_STFT  = np.zeros((N_STFT_half, numFrames), dtype=np.complex128);

    # Skip the DC bin by starting from bin 1
    for k in range(1, N_STFT_half):
        # reorganize data
        y_stack = np.swapaxes(np.squeeze(y_STFT[k,:,:]), 0, 1) # from (1 x numFrames x M) to (M x numFrames)
        h_stack = np.swapaxes(np.squeeze(H_hat_post_STFT[k,:,:,0]), 0, 1) # from (1 x numFrames x M x 1) to (M x numFrames)
        # psi_sT_stack = (phi_s_hat[k,:,0])[None, :] # from (1 x numFrames x 1) to (1 x numFrames)
        psi_sT_stack = phi_s_hat[k,:,0] # from (1 x numFrames x 1) to (numFrames)

        q_stack, e_prio_stack, e_post_stack, e_post_smooth_stack = ISCLP(A, Psi_w_delta[k], y_stack, psi_sT_stack, h_stack, Psi_w_tilde_init[k], beta_ISCLP_KF)

        # save output
        q_STFT[k,:]               = q_stack;
        e_prio_STFT[k,:]          = e_prio_stack;
        e_post_STFT[k,:]          = e_post_stack;
        e_post_smooth_STFT[k,:]   = e_post_smooth_stack;

    if debug_dir:
        spectrogram(e_post_STFT, scale='mag', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"e_post.png", title=f"e post, beta = 0")
        spectrogram(e_post_smooth_STFT, scale='mag', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"e_post_smooth.png", title=f"e post, beta = {beta_ISCLP_KF:2f}")

    e_post_TD = istft(e_post_STFT, fs)
    e_post_smooth_TD = istft(e_post_smooth_STFT, fs)

    if debug_dir:
        wavfile.write(debug_dir / "e_post.wav", fs, e_post_TD)
        wavfile.write(debug_dir / "e_post_smooth.wav", fs, e_post_smooth_TD)

def _enhance_one_file(
    scenfile: Path,
    outdir: Path,
    room_dir: Path,
    distant_dir: Path,
    lap_div_file: Path,
    alpha_file: Path,
    debug_dir: Path | None,
    dspconf: config.DspCfg,
    ):
    # Load scenario and room
    scen = load_scenario(scenfile)
    scen_hash = make_pydantic_cache_key(scen)
    log.debug(f"Enhancing scenario {scen_hash}")
    room = config.load_room(room_dir / f"{scen.room_id}.room.json")

    micpos = [np.asarray(x, dtype=np.float32) for x in room.mic_pos]

    # Build source position list
    target_source_idx = scen.target_speaker.sourceloc
    target_loc = room.sources[target_source_idx].location_history[0].location_m
    sourcepos = [np.asarray(target_loc, dtype=np.float32)]
    for i, src in enumerate(room.sources):
        if i == target_source_idx:
            continue # already added the target speaker at index 0 of sourcepos list
        loc = src.location_history[0].location_m # first position
        sourcepos.append(np.asarray(loc, dtype=np.float32))

    # Build ISCLP config class
    isclp_conf = ISCLPConfig(
        fs = dspconf.enhancement_samplerate,
        M = len(micpos),
        N = len(sourcepos),
        L = dspconf.isclpconf.L,
        N_STFT = dspconf.window_len,
        R_STFT = dspconf.hop_size,
        micpos = micpos,
        sourcepos = sourcepos,
        alpha_ISCLP_exp_db = dspconf.isclpconf.alpha_ISCLP_exp_db,
        psi_wLP_db = dspconf.isclpconf.psi_wLP_db,
        psi_wSC_db_range = dspconf.isclpconf.psi_wSC_db_range,
        beta_ISCLP_db = dspconf.isclpconf.beta_ISCLP_db,
        retf_thres_db = dspconf.isclpconf.retf_thres_db,
        x_tick_prop = dspconf.x_tick_prop,
        y_tick_prop = dspconf.y_tick_prop,
        c_range = dspconf.c_range,
    )

    # Load distant and clean speech
    distant_file = distant_dir / f"{scen_hash}.wav"
    fs_wav, y = wavfile.read(distant_file)
    if fs_wav != dspconf.enhancement_samplerate:
        log.error(f"Mismatch in sample rate for distant wav file ({distant_file}) {fs_wav=} != {dspconf.enhancement_samplerate=}")
        exit(-1)
    if y.ndim != 2:
        log.error(f"Error for {distant_file} {y.shape=} (expected ndim=2)")
        exit(-1)

    target_file = distant_dir / f"{scen_hash}_target.wav"
    fs_wav, s = wavfile.read(target_file)
    if fs_wav != dspconf.enhancement_samplerate:
        log.error(f"Mismatch in sample rate for distant wav file ({distant_file}) {fs_wav=} != {dspconf.enhancement_samplerate=}")
        exit(-1)
    if s.ndim != 1:
        log.error(f"Error for {target_file} {s.shape=} (expected ndim=1)")
        exit(-1)

    enhance_isclp_kf(
        y = y,
        s = s,
        isclp_conf = isclp_conf,
        debug_dir = debug_dir / "isclp" / f"{scen_hash}" if debug_dir else None,
        outpath = outdir / "{scen_hash}",
        lap_div_file = lap_div_file,
        alpha_file = alpha_file,
    )

def enhance_scenarios(
    scenario_dir: Path,
    outdir: Path,
    room_dir: Path,
    distant_dir: Path,
    matfile_dir: Path,
    debug_dir: Path | None,
    dspconf: config.DspCfg,
):
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    num_processes = int(slurm_cpus) if slurm_cpus else os.cpu_count()
    scenario_files = list(files_in_path_recursive(scenario_dir, "*.scenario.json"))
    log.info(f"Enhancing {len(scenario_files)} scenarios with {num_processes} workers")

    worker_fn = partial(
        _enhance_one_file,
        outdir=outdir,
        room_dir=room_dir,
        distant_dir=distant_dir,
        lap_div_file=matfile_dir / "lap_div.mat",
        alpha_file=matfile_dir / "alpha_sqrtMP.mat",
        debug_dir=debug_dir,
        dspconf=dspconf,
    )

    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(scenario_files), desc="Enhancing scenarios") as pbar:
            # imap_unordered yields one result as each worker finishes
            for _ in pool.imap_unordered(worker_fn, scenario_files):
                pbar.update()

def main():
    cfg = config.get()
    enhance_scenarios(
        scenario_dir=cfg.paths.scenario_dir,
        outdir=cfg.paths.enhanced_dir,
        room_dir=cfg.paths.room_dir,
        distant_dir=cfg.paths.distant_dir,
        matfile_dir=cfg.paths.matfile_dir,
        debug_dir=cfg.paths.debug_dir if cfg.debug else None,
        dspconf=cfg.dsp,
    )
