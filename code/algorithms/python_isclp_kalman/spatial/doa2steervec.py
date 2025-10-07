##################
# doa2steervec.m
##################
# converts DoA in steering vector

# INPUT:
# micPos:         microphone positions - channels x coordinates
# sourceAng:      DoA angles of sources
# fs:             sampling frequency
# c:              speed of sound

# OUTPUT:
# H:              steering vectors - freqbins x 1 x channels x sources

# -----------------

import numpy as np

def doa2steervec(micPos, sourceAng, N_FT_half, fs, c):
    M = micPos.shape[0] # number of rows (microphones)
    N_src = len(sourceAng) 

    H = np.zeros((N_FT_half, 1, M, N_src), dtype=complex)
    f = np.arange(N_FT_half) * fs / (2*(N_FT_half-1))
    d = np.sqrt(np.sum((micPos - np.tile(micPos[0, :], (M, 1)))**2, axis=1)) # distance of each mic relative to the first mic

    # looping sources and frequency bins
    for n in range(N_src):
        for k in range(N_FT_half):
            
            # delay for angle and microphone distances
            delay = np.sin(np.deg2rad(sourceAng[n])) * d / c
            
            # complex exponential steering vector
            H[k, 0, :, n] = np.exp(-1j * 2 * np.pi * f[k] * delay)
    
    return H
