import numpy as np

def tau2forget(tau, R_STFT, fs):            
    tau = np.asarray(tau)                 # ensures tau works if it's a scalar or array
    forget = np.exp(-R_STFT / (fs * tau)) # forgetting factor from tau and STFT overlap
    return forget

def forget2tau(forget, R_STFT, fs):            
    forget = np.asarray(forget)             # ensures forget works if it's a scalar or array
    tau = -R_STFT / (fs * np.log(forget))   # time constant from forgetting factor and STFT overlap
    return tau
