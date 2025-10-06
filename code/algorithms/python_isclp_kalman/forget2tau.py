###############
# forget2tau.m
###############
# converts forgetting factor to time constant

# Input:
# forget:   forgetting factor
# R_STFT:   frame shift (overlap between STFTs)
# fs:       sampling frequency

# Output:
# tau:      time constant    

# --------------------

import numpy as np

def forget2tau(forget, R_STFT, fs):            
    forget = np.asarray(forget)             # ensures forget works if it's a scalar or array
    tau = -R_STFT / (fs * np.log(forget))   # time constant from forgetting factor and STFT overlap
    return tau
