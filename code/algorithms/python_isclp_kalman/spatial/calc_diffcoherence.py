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

# ---------------------------

import numpy as np
from scipy.special import j0 

def calc_diffcoherence(micPos, N_STFT, fs, c, reg, type='spherical'):
    N_STFT_half = N_STFT // 2 + 1
    f = np.linspace(0.0, fs/2.0, N_STFT_half)          # match MATLAB
    M = micPos.shape[0]

    # Start with (1+reg) on the diagonal and elsewhere (like MATLAB)
    Gamma = (1.0 + reg) * np.ones((N_STFT_half, 1, M, M), dtype=float)

    for m_out in range(M-1):
        for m_in in range(m_out+1, M):
            d = np.linalg.norm(micPos[m_out] - micPos[m_in])
            if type == 'spherical':
                # MATLAB’s sinc is normalized; numpy.sinc matches that: sin(pi x)/(pi x)
                val = np.sinc(2.0 * f * d / c)
            elif type == 'cylindrical':
                val = j0(2.0 * np.pi * f * d / c)
            else:
                raise ValueError("type must be 'spherical' or 'cylindrical'")
            Gamma[:, 0, m_out, m_in] = val
            Gamma[:, 0, m_in, m_out] = val   # <-- mirror to make it symmetric!

    return Gamma
