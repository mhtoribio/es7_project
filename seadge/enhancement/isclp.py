from pathlib import Path
from scipy.io import loadmat
from dataclasses import dataclass
import numpy as np
import torch

from seadge.utils.log import log
from seadge.utils.dsp import db2pow, db2mag, stft, istft
from seadge.utils.visualization import spectrogram
from seadge.utils.wavfiles import write_wav
from seadge.utils.dsp import complex_to_mag_phase
from seadge.utils.isclp_helpers.isclp_kf import build_psi_w_delta
from seadge.utils.isclp_helpers.tau import tau2forget
from seadge.utils.isclp_helpers.spatial import calc_diffcoherence, pos2steervec
from seadge.utils.isclp_helpers.desmooth_GEVD import desmooth_GEVD
from seadge.utils.isclp_helpers.estim_corrmat import estim_corrmat
from seadge.utils.isclp_helpers.estim_psd_retf import estim_psd_retf
from seadge.utils.isclp_helpers.isclp_kf import ISCLP

from seadge.models.psd_cnn import SimplePSDCNN as psd_model

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

_model = None
def _load_psd_model(modelpath: Path, num_freqbins: int, num_mics: int):
    global _model
    log.debug(f"Loading PSD CNN model from {modelpath}")
    _model = psd_model(num_freqbins=num_freqbins, num_mics=num_mics)
    _saved_model = torch.load(modelpath, map_location='cpu')
    _model.load_state_dict(_saved_model)

def _dnn_psd_estimation(y: np.ndarray):
    #distant_mag, distant_phase = complex_to_mag_phase(y)
    # features: (2K, L, M)
    #features = np.concatenate((distant_mag, distant_phase))
    # features: (1, 2K, L, M)
    mag, _ = complex_to_mag_phase(y)
    features = mag
    features = torch.as_tensor(features[None, :, :, :], dtype=torch.float32, device="cpu")

    if not _model:
        log.error(f"Model not loaded. Error in code.")
        raise SystemError

    with torch.no_grad():
        y_pred_log = _model(features)
        y_pred = torch.expm1(y_pred_log)

    #log.debug(f"{features.shape=}, {y.shape=}, {distant_mag.shape=}, {distant_phase.shape=}, {y_pred.shape=}")
    return y_pred.squeeze(0).numpy()

def _core_isclp(
    N_STFT_half: int,
    numFrames: int,
    A: float,
    y_STFT: np.ndarray,
    H_hat_post_STFT: np.ndarray,
    phi_s_hat: np.ndarray,
    Psi_w_delta: list,
    Psi_w_tilde_init: list,
    beta_ISCLP_KF: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        psi_sT_stack = phi_s_hat[k,:] # from (1 x numFrames) to (numFrames)

        q_stack, e_prio_stack, e_post_stack, e_post_smooth_stack = ISCLP(A, Psi_w_delta[k], y_stack, psi_sT_stack, h_stack, Psi_w_tilde_init[k], beta_ISCLP_KF)

        # save output
        q_STFT[k,:]               = q_stack;
        e_prio_STFT[k,:]          = e_prio_stack;
        e_post_STFT[k,:]          = e_post_stack;
        e_post_smooth_STFT[k,:]   = e_post_smooth_stack;

    return q_STFT, e_prio_STFT, e_post_STFT, e_post_smooth_STFT

def enhance_isclp_kf(
        y: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        s: np.ndarray[tuple[int], np.dtype[np.float32]],
        isclp_conf: ISCLPConfig,
        debug_dir: Path | None,
        lap_div_file: Path,
        alpha_file: Path,
        modelpath: Path,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    log.debug(f"enhance_isclp_kf {isclp_conf=}, {debug_dir=}, {lap_div_file=}, {alpha_file=}, {modelpath=}")

    if debug_dir:
        debug_dir.mkdir(exist_ok=True, parents=True)
    # Initialization / Params
    fs = isclp_conf.fs
    N_STFT_half = isclp_conf.N_STFT // 2 + 1 # num freqbins
    N_STFT = isclp_conf.N_STFT
    R_STFT = isclp_conf.R_STFT
    L = isclp_conf.L
    M = isclp_conf.M

    # Load model
    if not _model:
        _load_psd_model(modelpath, N_STFT_half, M)

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
    psd_true = np.zeros_like(y_STFT[:, :, 0]) # freqbin x frame
    psd_true[:, :s_STFT.shape[1]] = np.abs(s_STFT) ** 2 # s_STFT may be a frame or two shorter than y_STFT

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
        spectrogram(phi_s_hat[:,:,0], scale='pow', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"phi_s_hat_sqrtpsdretf.png", title=f"Target PSD estimate (SQRT-PSD-RETF)")
        spectrogram(psd_true, scale='pow', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"phi_s_hat_truth.png", title=f"Target PSD estimate (Ground Truth)")

    #### ISCLP-KF
    numFrames = y_STFT.shape[1]
    _, _, e_post_vanilla_STFT, e_post_vanilla_smooth_STFT = _core_isclp(
        N_STFT_half=N_STFT_half,
        numFrames=numFrames,
        A=A,
        y_STFT=y_STFT,
        H_hat_post_STFT=H_hat_post_STFT.copy(),
        phi_s_hat=phi_s_hat[:,:,0], # choose speaker 0
        Psi_w_delta=Psi_w_delta,
        Psi_w_tilde_init=Psi_w_tilde_init,
        beta_ISCLP_KF=beta_ISCLP_KF,
    )

    #### Deep-ISCLP-KF
    phi_dnn = _dnn_psd_estimation(y_STFT)
    if debug_dir:
        spectrogram(phi_dnn, scale='pow', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"phi_s_hat_dnn.png", title=f"Target PSD estimate (DNN)")
    _, _, e_post_dnn_STFT, e_post_dnn_smooth_STFT = _core_isclp(
        N_STFT_half=N_STFT_half,
        numFrames=numFrames,
        A=A,
        y_STFT=y_STFT,
        H_hat_post_STFT=H_hat_post_STFT.copy(),
        phi_s_hat=phi_dnn, # (freqbin x frame)
        Psi_w_delta=Psi_w_delta,
        Psi_w_tilde_init=Psi_w_tilde_init,
        beta_ISCLP_KF=beta_ISCLP_KF,
    )

    # ISCLP-KF with oracle PSD
    _, _, e_post_oracle_STFT, e_post_oracle_smooth_STFT = _core_isclp(
        N_STFT_half=N_STFT_half,
        numFrames=numFrames,
        A=A,
        y_STFT=y_STFT,
        H_hat_post_STFT=H_hat_post_STFT.copy(),
        phi_s_hat=psd_true, # (freqbin x frame)
        Psi_w_delta=Psi_w_delta,
        Psi_w_tilde_init=Psi_w_tilde_init,
        beta_ISCLP_KF=beta_ISCLP_KF,
    )

    if debug_dir:
        spectrogram(e_post_vanilla_STFT, scale='mag', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"e_post_vanilla.png", title=f"e post ISCLP-KF, beta = 0")
        spectrogram(e_post_vanilla_smooth_STFT, scale='mag', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"e_post_vanilla_smooth.png", title=f"e post ISCLP-KF, beta = {beta_ISCLP_KF:2f}")
        spectrogram(e_post_dnn_STFT, scale='mag', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"e_post_dnn.png", title=f"e post Deep-ISCLP-KF, beta = 0")
        spectrogram(e_post_dnn_smooth_STFT, scale='mag', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"e_post_dnn_smooth.png", title=f"e post Deep-ISCLP-KF, beta = {beta_ISCLP_KF:2f}")
        spectrogram(e_post_oracle_STFT, scale='mag', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"e_post_oracle.png", title=f"e post ISCLP-KF (Oracle PSD), beta = 0")
        spectrogram(e_post_oracle_smooth_STFT, scale='mag', x_tick_prop=x_tick_prop, y_tick_prop=y_tick_prop, c_range=c_range, filename=debug_dir/"e_post_oracle_smooth.png", title=f"e post ISCLP-KF (Oracle PSD), beta = {beta_ISCLP_KF:2f}")

    e_post_vanilla_TD = istft(e_post_vanilla_STFT, fs)
    e_post_vanilla_smooth_TD = istft(e_post_vanilla_smooth_STFT, fs)
    e_post_dnn_TD = istft(e_post_dnn_STFT, fs)
    e_post_dnn_smooth_TD = istft(e_post_dnn_smooth_STFT, fs)
    e_post_oracle_TD = istft(e_post_oracle_STFT, fs)
    e_post_oracle_smooth_TD = istft(e_post_oracle_smooth_STFT, fs)

    # normalize all
    e_post_vanilla_TD = e_post_vanilla_TD / np.max(np.abs(e_post_vanilla_TD))
    e_post_vanilla_smooth_TD = e_post_vanilla_smooth_TD / np.max(np.abs(e_post_vanilla_smooth_TD))
    e_post_dnn_TD = e_post_dnn_TD / np.max(np.abs(e_post_dnn_TD))
    e_post_dnn_smooth_TD = e_post_dnn_smooth_TD / np.max(np.abs(e_post_dnn_smooth_TD))
    e_post_oracle_TD = e_post_oracle_TD / np.max(np.abs(e_post_oracle_TD))
    e_post_oracle_smooth_TD = e_post_oracle_smooth_TD / np.max(np.abs(e_post_oracle_smooth_TD))

    if debug_dir:
        write_wav(debug_dir / "e_post_vanilla.wav", e_post_vanilla_TD, fs=fs)
        write_wav(debug_dir / "e_post_vanilla_smooth.wav", e_post_vanilla_smooth_TD, fs=fs)
        write_wav(debug_dir / "e_post_dnn.wav", e_post_dnn_TD, fs=fs)
        write_wav(debug_dir / "e_post_dnn_smooth.wav", e_post_dnn_smooth_TD, fs=fs)
        write_wav(debug_dir / "e_post_oracle.wav", e_post_oracle_TD, fs=fs)
        write_wav(debug_dir / "e_post_oracle_smooth.wav", e_post_oracle_smooth_TD, fs=fs)

    return e_post_vanilla_TD, e_post_vanilla_smooth_TD, e_post_dnn_TD, e_post_dnn_smooth_TD, e_post_oracle_TD, e_post_oracle_smooth_TD
