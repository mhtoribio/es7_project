################
# estim_corrmat
################
# Estimates correlation matrix

# Input:
# X:            STFT data - freqbins x frames x channels
# alpha:        forgetting factor

# Output:
# Psi_x_smth:   smooth correlation matrix estimate - freqbins x frames x channels x channels
# Psi_x_mean:   mean correlation matrix estimate - freqbins x 1 x channels x channels

import numpy as np

def estim_corrmat(X, alpha):

    eps = np.finfo(float).eps # small number

    # dimensions of data (X)
    # N_half: freqbins
    # L:      frames
    # M:      channels
    N_half, L, M = X.shape[:3]

    ### Compute Average PSD Matrix ###
    if alpha != 1:
        R_inst = np.zeros((N_half, L, M, M), dtype=np.complex64)
        for l in range(L):
            for k in range(N_half):
                x = X[k, l, :]
                R_inst[k, l, :, :] = np.outer(x, x.conj())
        Psi_x_mean = np.sum(R_inst, axis=1, keepdims=True) / L # sum all frames (axis=1 of R_inst) and divide by frames
    
    else:
        Psi_x_mean = np.zeros((N_half, 1, M, M), dtype=np.complex64)
        for i in range(L):
            for k in range(N_half):
                x = X[k, l, :]
                Psi_x_mean[k, 0, :, :] += np.outer(x, x.conj()) 
        Psi_x_mean = Psi_x_mean / L
    
    ### Compute Smooth PSD Matrix ###
    R_smth_tmp = eps * np.ones((N_half, 1, M, M), dtype=np.complex64)
    if alpha != 1:
        Psi_x_smth = np.zeros((N_half, L, M, M), dtype=np.complex64)
        for l in range(L):
            R_smth_tmp = alpha*R_smth_tmp + (1-alpha)*R_inst[:, l, :, :]
            Psi_x_smth[:, l, :, :] = R_smth_tmp

    else:
        Psi_x_smth = Psi_x_mean  

    return Psi_x_smth, Psi_x_mean



    


