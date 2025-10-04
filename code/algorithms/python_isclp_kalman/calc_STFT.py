import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# -----------------------------
# Define calc_STFT function
# -----------------------------
def calc_STFT(x, fs, win, N_STFT, R_STFT, sides='onesided'):
    """
    performs the STFT.

    Parameters
    ----------
    x : ndarray (samples x channels)
        Input signal
    fs : int or float
        Sampling frequency
    win : ndarray
        Window function
    N_STFT : int
        Frame length (FFT size)
    R_STFT : int
        Frame shift (hop size)
    sides : str
        'onesided' or 'twosided'

    Returns
    -------
    X : ndarray
        STFT tensor: freqbins x frames x channels
    f : ndarray
        Frequency vector
    """
    if x.ndim == 1:
        x = x[:, None]  # make it (samples x 1)

    M = x.shape[1]
    X_list = []
    
    for m in range(M):
        f, t, Zxx = stft(
            x[:, m],
            fs=fs,
            window=win,
            nperseg=N_STFT,
            noverlap=N_STFT - R_STFT,
            nfft=N_STFT,
            return_onesided=(sides == 'onesided'),
            boundary=None,
            padded=False
        )
        X_list.append(Zxx)
    
    # Stack into freq x frames x channels
    X = np.stack(X_list, axis=-1)
    return X, f

""" # -----------------------------
# Test signal parameters
# -----------------------------
fs = 1000                     # Sampling frequency (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # 1 second
x = np.sin(2*np.pi*50*t)      # 50 Hz sine wave

# STFT parameters
N_STFT = 256
R_STFT = N_STFT/4
win = np.hanning(N_STFT)

# -----------------------------
# Compute STFT
# -----------------------------
X, f = calc_STFT(x, fs, win, N_STFT, R_STFT, sides='onesided')

# -----------------------------
# Find peak frequency in first frame
# -----------------------------
magnitude = np.abs(X[:, 0, 0])
peak_idx = np.argmax(magnitude)
peak_freq = f[peak_idx]

print(f"Peak frequency: {peak_freq:.2f} Hz")

# -----------------------------
# Plot magnitude spectrum
# -----------------------------
plt.figure(figsize=(8,4))
plt.plot(f, magnitude, marker='o')
plt.title("STFT Magnitude of First Frame")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show() """
