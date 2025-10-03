###############
# calc_diffcoherence.m
###############
# calculates diffuse coherence matrix.

# INPUT:
# micPos:    microphone positions - channels x coordinates
# N_STFT:    STFT frame length 
# fs:        sampling rate
# c:         speed of sound 
# reg:       regularization (avoids ill-conditioned matrices at very low frequencies)
# type:      {'spherical', 'cylindrical'}, coherence type

# OUTPUT:
# Gamma:     diffuse coherence matrix - freqbins x 1 x channels x channels

import numpy as np
from scipy.special 

def calc_diff_coherence(micPos, N_STFT, fs, c, reg, type='spherical'):
    
    # frequency vector
    N_STFT_half = N_STFT // 2 + 1
    f = np.linspace(0, fs/2, N_STFT_half)
    
    M = micPos.shape[0]
    Gamma = (1 + reg) * np.ones((N_STFT, 1, M, M))

    for m_out in range(M-1):
        for m_in in range(m_out+1, M):
            d = np.linalg.norm(micPos[m_out, :] - micPos[m_in, :])
            if type == 'spherical':
                Gamma[:, 0, m_out, m_in] = np.sinc(2*f*d/c)
            elif type == 'cylindrical':
                Gamma[:, 0, m_out, m_in] = besselj(0, 2*np.pi*f*d/c)
            else:
                raise ValueError("type must be 'spherical' or 'clyndrical'")
            Gamma[:, 0, m_out, m_in] = Gamma[:, 0, m_out, m_in]
    
    return Gamma


















