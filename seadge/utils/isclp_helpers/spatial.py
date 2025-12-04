import numpy as np
import math
from scipy.special import j0 

##############
# Spatial
##############

def pos2steervec(micPos, sourcePos, N_FT_half, fs, c):
    micPos = np.asarray(micPos)
    M = micPos.shape[0] # number of rows (microphones)
    N_src = len(sourcePos)

    H = np.zeros((N_FT_half, 1, M, N_src), dtype=complex)
    f = np.arange(N_FT_half) * fs / (2*(N_FT_half-1))
    d = np.sqrt(np.sum((micPos - np.tile(micPos[0, :], (M, 1)))**2, axis=1)) # distance of each mic relative to the first mic

    # looping sources and frequency bins
    for n in range(N_src):
        # assume far-field (angle to each microphone is the same)
        # angle from y-plane
        sourceAng_rad = math.pi/2 - math.atan2(sourcePos[n][0] - micPos[0][0], sourcePos[n][1] - micPos[0][1])
        for k in range(N_FT_half):
            
            # delay for angle and microphone distances
            delay = np.sin(sourceAng_rad) * d / c
            
            # complex exponential steering vector
            H[k, 0, :, n] = np.exp(-1j * 2 * np.pi * f[k] * delay)
    
    return H

def calc_diffcoherence(micPos, N_STFT, fs, c, reg, type='spherical'):
    N_STFT_half = N_STFT // 2 + 1
    f = np.linspace(0.0, fs/2.0, N_STFT_half)          # match MATLAB
    M = len(micPos)

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
