###############
# tau2forget.m
###############
# converts time constant to forgetting factor 

# Input:
# tau:      time constant
# R_STFT:   frame shift (overlap between STFTs)
# fs:       sampling frequency

# Output:
# forget:   forgetting factor    

import numpy as np

def tau2forget(tau, R_STFT, fs):            
    tau = np.asarray(tau)                 # ensures tau works if it's a scalar or array
    forget = np.exp(-R_STFT / (fs * tau)) # forgetting factor from tau and STFT overlap
    return forget
