##########################################
# ISCLP - Kalman Filter Matlab to Python
##########################################

import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import get_window
from scipy.io import loadmat
import tau2forget

# Functions
def db2pow(x):
    return 10**(x/10)

def db2mag(x):
    return 10**(x/20)

### PREAMBLE

### CONFIGURATION

#####################
# ACOUSTIC SETTINGS
#####################

c = 340 # speed of sound (343 m/s)
micPos = np.array([[0,0],[0.08, 0],[0.16, 0],[0.24, 0],[0.32, 0]]) # microphone positions. (0-based index in python) (1-based index in python)
M = np.size(micPos, 0) # number of microphones (rows in micPos) 
sourceAng = np.array([0, 60]) # source angles
SNR = float(input("Which SNR/dB? ")) # returns the SNR in float

'''
# microphone signal components
fs_x1, x1_TD       = wavfile.read(os.path.join('.', 'audio', 'x1.wav'))  
fs_s1, s1_TD       = wavfile.read(os.path.join('.', 'audio', 's1.wav'))
fs_x2, x2_TD       = wavfile.read(os.path.join('.', 'audio', 'x2.wav'))
fs_v, v_TD_SNR_0dB = wavfile.read(os.path.join('.', 'audio', 'v_SNR_0dB.wav'))
# (optional) convert int to float. matlab returns audio file in float. python returns in int16
# x1_TD = x1_TD.astype(np.float32) 
# s1_TD = s1_TD.astype(np.float32)
# x2_TD = x2_TD.astype(np.float32)
# v_TD_SNR_0dB = v_TD_SNR_0dB.astype(np.float32)
v_TD_SNR_scaled = (db2mag(-SNR)) * v_TD_SNR_0dB  # convert magnitude from dB
y_TD = x1_TD + x2_TD + v_TD_SNR_scaled            # mix signal 
'''

########################
# ALGORITHMIC SETTINGS
########################

#### STFT
fs = 16000
N_STFT = 512                                            # frame length (samples/window)
R_STFT = N_STFT / 2                                     # frame shift (samples of overlap = 50 %) 
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
PSI_w_delta = [None] * N_STFT_half                  # initialize list for each frequency bin (cell() in matlab)

for k in range(N_STFT_half):
    # diag([psi_wSC(k)*ones(M-1,1); psi_LP_init])
    top = psi_wSC[k] * np.ones(M-1)
    diag_entries = np.concatenate([top, psi_LP_init])
    Psi_w_tilde_init[k] = np.diag(diag_entries) # between (52) and (53) 
    PSI_w_delta[k] = (1-alpha_ISCLP_KF) * Psi_w_tilde_init[k] 

beta_SCLP_KF = db2mag(-2) 

#### PSD ESTIMATION AND RETF UPDATE
# File paths for .mat files
script_dir = os.path.dirname(__file__)  # folder of main.py
lap_div_file = os.path.join(script_dir, "..", "matfiles", "lap_div.mat")
lap_div_file = os.path.abspath(lap_div_file)
alpha_file = os.path.join(script_dir, "..", "matfiles2", "alpha_sqrtMP.mat")
alpha_file = os.path.abspath(alpha_file)

zeta = tau2forget.tau2forget(2*M*R_STFT/fs, R_STFT, fs) # forgetting factor zeta
tmp = loadmat(lap_div_file) # load laplace coefficients of speech per frequency bins
lap_div = tmp["lap_div"]
xi_tresh = db2pow(-2)  # RETF updating threshold
tmp = loadmat(alpha_file) 
alpha_SQRT_PSD_RETF = tmp["alpha_sqrtMP"]
































