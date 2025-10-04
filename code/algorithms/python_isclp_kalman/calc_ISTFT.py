import numpy as np
from scipy.signal import istft

def calc_ISTFT(X, win, N_STFT, R_STFT, sides='onesided', fs=1.0):
    """
    performs the inverse STFT.
    
    Parameters
    ----------
    X : ndarray (freqbins x frames x channels)
        STFT tensor
    win : ndarray
        Window function
    N_STFT : int
        Frame length (FFT size)
    R_STFT : int
        Frame shift (hop size)
    sides : str
        'onesided' or 'twosided'
    fs : float
        Sampling frequency (optional, default=1.0)
    
    Returns
    -------
    x : ndarray (samples x channels)
        Reconstructed time-domain signal
    """

    if X.ndim == 2:  # single channel
        X = X[:, :, None]

    M = X.shape[2]
    x_list = []

    for m in range(M):
        # Use 'onesided' or 'twosided' mode
        input_sides = 'onesided' if sides == 'onesided' else 'twosided'
        
        _, x_rec = istft(
            X[:, :, m],
            fs=fs,
            window=win,
            nperseg=N_STFT,
            noverlap=N_STFT - R_STFT,
            nfft=N_STFT,
            input_onesided=(sides == 'onesided'),
            boundary=None,
            time_axis=1,
            freq_axis=0
        )
        x_list.append(x_rec)

    # Stack channels
    x = np.stack(x_list, axis=-1)
    return x



""" # --- Test parameters ---
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
fs = 8000               # Sampling frequency
T = 0.2                 # Duration in seconds
t = np.arange(0, T, 1/fs)
f1, f2 = 500, 0       # Frequencies in Hz

# Mono test signal
x = 0.7*np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)

# STFT parameters
N_STFT = 256
R_STFT = 128
win = get_window('hann', N_STFT)

# --- Compute STFT ---
f, time, X = stft(x, fs=fs, window=win, nperseg=N_STFT,
                  noverlap=N_STFT-R_STFT, nfft=N_STFT, return_onesided=True)

# --- Reconstruct signal using calc_ISTFT ---
x_rec = calc_ISTFT(X, win, N_STFT, R_STFT, sides='onesided', fs=fs)

# --- Plots ---
plt.figure(figsize=(12, 8))

# Original vs Reconstructed
plt.subplot(2,1,1)
plt.plot(t, x, label='Original')
plt.plot(t, x_rec[:len(t),0], '--', label='Reconstructed')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Original vs Reconstructed Signal')
plt.legend()

# Magnitude spectrogram
plt.subplot(2,1,2)
plt.pcolormesh(time, f, np.abs(X), shading='gouraud')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Magnitude Spectrogram')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()

# --- Check reconstruction error ---
error = np.max(np.abs(x - x_rec[:len(x),0]))
print(f"Maximum reconstruction error: {error:.2e}")
 """