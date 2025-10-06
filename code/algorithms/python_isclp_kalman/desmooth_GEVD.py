#####################
# desmooth_GEVD.m
#####################
# performs GEVD and subspace-based desmoothing

# INPUT: 
# Psi_x_STFT:                correlation matrix - freqbins x frames x channels x channels
# Gamma_FT:                  diffuse coherence matrix - freqbins x 1 x channels x channels
# 'lambdaMin', lambdaMin:    eigenvalue threshold
# 'forgetPSD', forgetPSD:    forgetting factor for desmoothing

# OUTPUT:
# P_STFT:                    eingevectors - freqbins x frames x channels x channels
# lambda_STFT:               desmoothed eigenvalues - freqbins x frames x channels
# lambda_LP_STFT:            smooth eigenvalues - freqbins x frames x channels

# ------------------

import numpy as np
# from scipy.linalg import eigh

def desmooth_GEVD(Psi_x_STFT, Gamma_FT, varargin):

    # dimensions
    N_FT_half, L, M = Psi_x_STFT.shape[:3]

    forgetPSD = 0 # PSD forgetting factor
    lambdaMin = 0 # eigenvalue minimum threshold

    # read options from input
    for i in range(1, len(varargin), 2):
        if isinstance(varargin(i), 'str'):
            if varargin(i) == 'forgetPSD':
                forgetPSD = varargin(i+1)
            elif varargin(i) == 'lambdaMin':
                lambdaMin = varargin(i+1)

    # init
    P_STFT         = np.zeros(N_FT_half, L, M, M)
    lambda_STFT    = np.zeros(N_FT_half, L, M)
    lambda_LP_STFT = np.zeros(N_FT_half, L, M)

    for k in range (2, N_FT_half):
        R_v = 0

        # Regularization
        Gamma = np.squeeze(Gamma_FT[k, 0, :, :])

        for l in range(L):
            # GEVD
            Psi_x = np.squeeze(Psi_x_STFT[k, l, :, :])
            P, Lambda_LP = np.linalg.eig(Psi_x, Gamma)
            lambda_LP = np.real(np.diag(Lambda_LP)) # ignore complex component and set negative to zero
            d = np.sqrt(np.real(np.diag(P.conj().T @ Gamma @ P)))
            P = P / d[np.newaxis, :] # rescaling W 

            # sorting eigenvalues/eigenvectors
            if l > 1:
                maxIdx = np.argsort(-lambda_LP)        # descending sort of lambda_LP
                Q = np.abs(P_old.conj().T @ Gamma @ P) # Q should approximate I
                
                # temporary variable
                P_tmp = P 
                lambda_LP_tmp = lambda_LP

                # order such that Q approximate I
                for m in maxIdx:
                    maxIdx = np.argmax(Q[:, m])          # returns index of max value
                    Q[maxIdx, :] = 0                     # sets all elements in that row to 0 (ensures that it wont be picked again)
                    P[:, maxIdx] = P_tmp[:, m]           # sort P
                    lambda_LP[maxIdx] = lambda_LP_tmp[m] # sort lambda

            P_old = P

            # save
            P_STFT[k, l, :, :] = np.transpose(P, (2, 3, 0, 1)) # shifts dimension 2 to the right
            lambda_LP_STFT[k, l, :] = np.transpose(lambda_LP, (2, 0, 1)) # shifts dimension 2 to the right

            # Filter eigenvalues to compensate for recursive averaging
            if l > 1:
                # apply HP to lambda
                lambda_LP_old = np.squeeze(lambda_LP_STFT[k, l-1, :])
                lmbda = 1 / (1 - forgetPSD) * lambda_LP - forgetPSD / (1 - forgetPSD) * lambda_LP_old
                lmbda(lmbda < lambdaMin) = lambdaMin

            else:
                lmbda = lambda_LP
                lmbda(lmbda < lambdaMin) = lambdaMin

            # save
            lambda_STFT[k, l, :] = np.transpose(lmbda, (2, 0, 1)) # shifts dimension 2 to the right
            
    return P_STFT, lambda_STFT, lambda_LP_STFT