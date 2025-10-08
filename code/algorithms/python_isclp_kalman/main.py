##########################################
# ISCLP - Kalman Filter Matlab to Python
##########################################

import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import get_window, ShortTimeFFT
from scipy.io import loadmat
from estim.sqrt_psd_retf.corrmat.tau import tau2forget, forget2tau
from spatial.doa2steervec import doa2steervec
from spatial.calc_diffcoherence import calc_diffcoherence
from tools.plot_spectogram import plot_spec
from estim.sqrt_psd_retf.corrmat.estim_corrmat import estim_corrmat
from estim.sqrt_psd_retf.desmooth_GEVD import desmooth_GEVD
from estim.sqrt_psd_retf.estim_psd_retf import estim_psd_retf
from pathlib import Path
from estim.isclp_kf.isclp_kf import ISCLP

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("matlab_dir")
parser.add_argument("out_dir")
args = parser.parse_args()

MATLAB_DIR = Path(args.matlab_dir)
OUT_DIR = Path(args.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)
AUDIO_DIR = Path(MATLAB_DIR / "audio")
MATFILE_DIR = Path(MATLAB_DIR / "param/SQRT_PSD_RETF")

# Functions
def db2pow(x):
    return 10**(x/10)

def db2mag(x):
    return 10**(x/20)

### CONFIGURATION

#####################
# ACOUSTIC SETTINGS
#####################

c = 340 # speed of sound (343 m/s)
micPos = np.array([[0,0],[0.08, 0],[0.16, 0],[0.24, 0],[0.32, 0]]) # microphone positions. (0-based index in python) (1-based index in python)
M = np.size(micPos, 0) # number of microphones (rows in micPos) 
sourceAng = np.array([0, 60]) # source angles
SNR = float(input("Which SNR/dB? ")) # returns the SNR in float

# microphone signal components
fs_x1, x1_TD       = wavfile.read(AUDIO_DIR / 'x1.wav')
fs_s1, s1_TD       = wavfile.read(AUDIO_DIR / 's1.wav')
fs_x2, x2_TD       = wavfile.read(AUDIO_DIR / 'x2.wav')
fs_v, v_TD_SNR_0dB = wavfile.read(AUDIO_DIR / 'v_SNR_0dB.wav')
# convert int to float. matlab returns audio file in float. python returns in int16
x1_TD        = (0.99 / 0x7fff) * x1_TD
s1_TD        = (0.99 / 0x7fff) * s1_TD
x2_TD        = (0.99 / 0x7fff) * x2_TD
v_TD_SNR_0dB = (0.99 / 0x7fff) * v_TD_SNR_0dB
# convert to samples x 1 instead of 1d
s1_TD = s1_TD[:,None]
# add noise
v_TD_SNR_scaled = (db2mag(-SNR)) * v_TD_SNR_0dB  # convert magnitude from dB
y_TD = x1_TD + x2_TD + v_TD_SNR_scaled            # mix signal 

########################
# ALGORITHMIC SETTINGS
########################

#### STFT
fs = 16000
N_STFT = 512                                            # frame length (samples/window)
R_STFT = N_STFT // 2                                    # frame shift (samples of overlap = 50 %) 
win = np.sqrt(get_window('hann', N_STFT, fftbins=True)) # sqrt of periodic hann window
N_STFT_half = N_STFT // 2 + 1                           # floor frame shift
f = np.linspace(0, fs/2, N_STFT_half)                   # frequency vector

#### ISCLP KALMAN FILTER
L = 6                                               # prediction length
alpha_ISCLP_KF = 1 - db2pow(-25)                    # forgetting factor alpha (1 - power) 
A = np.sqrt(alpha_ISCLP_KF)
psi_wLP = db2pow(-4)                                # LP filter variance (eq. 54)
psi_wSC = db2pow(np.linspace(0, -15, 257))          # SC filter variance (eq. 53)
psi_LP_exp = psi_wLP ** np.arange(1, L)             # Length L-1
psi_LP_init = np.tile(psi_LP_exp, (M, 1)).flatten() # equivalent to kron()
Psi_w_tilde_init = [None] * N_STFT_half             # initialize list for each frequency bin (cell() in matlab)
Psi_w_delta = [None] * N_STFT_half                  # initialize list for each frequency bin (cell() in matlab)

for k in range(N_STFT_half):
    # diag([psi_wSC(k)*ones(M-1,1); psi_LP_init])
    top = psi_wSC[k] * np.ones(M-1)
    diag_entries = np.concatenate([top, psi_LP_init])
    Psi_w_tilde_init[k] = np.diag(diag_entries) # between (52) and (53) 
    Psi_w_delta[k] = (1-alpha_ISCLP_KF) * Psi_w_tilde_init[k] 

beta_SCLP_KF = db2mag(-2) 

#### PSD ESTIMATION AND RETF UPDATE
# File paths for .mat files
lap_div_file = MATFILE_DIR / "lap_div.mat"
alpha_file = MATFILE_DIR / "alpha_sqrtMP.mat"

zeta = tau2forget(2*M*R_STFT/fs, R_STFT, fs) # forgetting factor zeta
tmp = loadmat(lap_div_file) # load laplace coefficients of speech per frequency bins
lap_div = tmp["lap_div"]
xi_thresh = db2pow(-2)  # RETF updating threshold
tmp = loadmat(alpha_file) 
alpha_SQRT_PSD_RETF = tmp["alpha_sqrtMP"]
# initial RETFs
H_init_FT = doa2steervec(micPos, sourceAng, N_STFT_half, fs, c);
# diffuse coherence matrix
Gamma = calc_diffcoherence(micPos,N_STFT,fs,c,1e-3);

########################
# FIGURE SETTINGS
########################

# spectogram figure settings
xTickProp = (0, R_STFT/fs, fs/R_STFT);
yTickProp = (0, fs/(2000*R_STFT), R_STFT/2);
cRange    = (-55, 5);

########################
#### STFT PROCESSING
########################

STFT = ShortTimeFFT(win, R_STFT, fs, fft_mode="onesided", mfft=N_STFT)
# frequency bin X channel X frame
y_STFT = STFT.stft(y_TD, axis=0)
s1_STFT = STFT.stft(s1_TD, axis=0)
# Swap axes to: frequency bin X frame X channel
y_STFT = np.swapaxes(y_STFT, 1, 2)
s1_STFT = np.swapaxes(s1_STFT, 1, 2)

# Plot
plot_spec(y_STFT[:,:,0], scale="mag", x_tick_prop=xTickProp, y_tick_prop=yTickProp, c_range=cRange, filename=OUT_DIR/"y_STFT.png", title=f"Microphone signal, {SNR = } dB")
plot_spec(s1_STFT[:,:,0], scale="mag", x_tick_prop=xTickProp, y_tick_prop=yTickProp, c_range=cRange, filename=OUT_DIR/"s1_STFT.png", title=f"Target signal")

################################################
#### Early PSD estimation, RETF update
################################################

print(' * estimate early PSDs, update RETFs [2]...');

# correlation matrix of microphone signal
Psi_y_smth, Psi_y_mean = estim_corrmat(y_STFT, zeta);
# compute GEVD
P_STFT, lambda_STFT, _ = desmooth_GEVD(Psi_y_smth, Gamma, 0, zeta)

phi_s_hat, phi_xl_hat, H_hat_prior_STFT, H_hat_post_STFT, H_update_pattern = estim_psd_retf(P_STFT, lambda_STFT, Gamma, H_init_FT,
               alpha=alpha_SQRT_PSD_RETF,
               beta=20*lap_div*lap_div,
               xi_thresh=xi_thresh)

plot_spec(phi_s_hat[:,:,0], scale='pow', x_tick_prop=xTickProp, y_tick_prop=yTickProp, c_range=cRange, filename=OUT_DIR/"phi_s_hat.png", title=f"target PSD estimate, {SNR = } dB")

#########################################
#### ISCLP KALMAN FILTER
#########################################

print(' * run ISCLP Kalman filer [1]...');

# number of frames
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

    q_stack, e_prio_stack, e_post_stack, e_post_smooth_stack = ISCLP(A, Psi_w_delta[k], y_stack, psi_sT_stack, h_stack, Psi_w_tilde_init[k], beta_SCLP_KF)

    # save output
    q_STFT[k,:]               = q_stack;
    e_prio_STFT[k,:]          = e_prio_stack;
    e_post_STFT[k,:]          = e_post_stack;
    e_post_smooth_STFT[k,:]   = e_post_smooth_stack;

# plot
plot_spec(e_post_STFT, scale='mag', x_tick_prop=xTickProp, y_tick_prop=yTickProp, c_range=cRange, filename=OUT_DIR/"e_post.png", title=f"e post, beta = 0, {SNR = } dB")
plot_spec(e_post_smooth_STFT, scale='mag', x_tick_prop=xTickProp, y_tick_prop=yTickProp, c_range=cRange, filename=OUT_DIR/"e_post_smooth.png", title=f"e post, beta = {beta_SCLP_KF:2f}, {SNR = } dB")

#####################
#### ISTFT
#####################

e_post_TD        = STFT.istft(e_post_STFT)
e_post_smooth_TD = STFT.istft(e_post_smooth_STFT)

#####################
#### Write audio
#####################

wavfile.write(OUT_DIR / f"v_SNR_{SNR}dB.wav", fs, v_TD_SNR_scaled)
wavfile.write(OUT_DIR / f"y_SNR_{SNR}dB.wav", fs, y_TD)
wavfile.write(OUT_DIR / f"e_post_SNR_{SNR}dB.wav", fs, e_post_TD)
wavfile.write(OUT_DIR / f"e_post_smooth_SNR_{SNR}dB.wav", fs, e_post_smooth_TD)

# save normalized versions
norm_v_TD_SNR_scaled = (0.99 / np.max(np.abs(v_TD_SNR_scaled))) * v_TD_SNR_scaled
norm_y_TD = (0.99 / np.max(np.abs(y_TD))) * y_TD
norm_e_post_TD = (0.99 / np.max(np.abs(e_post_TD))) * e_post_TD
norm_e_post_smooth_TD = (0.99 / np.max(np.abs(e_post_smooth_TD))) * e_post_smooth_TD

wavfile.write(OUT_DIR / f"norm_v_SNR_{SNR}dB.wav", fs, norm_v_TD_SNR_scaled)
wavfile.write(OUT_DIR / f"norm_y_SNR_{SNR}dB.wav", fs, norm_y_TD)
wavfile.write(OUT_DIR / f"norm_e_post_SNR_{SNR}dB.wav", fs, norm_e_post_TD)
wavfile.write(OUT_DIR / f"norm_e_post_smooth_SNR_{SNR}dB.wav", fs, norm_e_post_smooth_TD)

print(" * Done")
