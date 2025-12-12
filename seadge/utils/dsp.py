from typing import Optional
import numpy as np
import math
from scipy.signal import ShortTimeFFT, get_window

from seadge import config

EPS = 1e-12

# Lazy loaded cache
# STFT class
_STFT : ShortTimeFFT | None = None
_fs : int | None = None

def _load_stft(fs: int):
    cfg = config.get()

    # Load window
    if cfg.dsp.window_type == "sqrt_hann":
        _win = np.sqrt(get_window('hann', cfg.dsp.window_len, fftbins=True))
    else:
        try:
            _win = get_window(cfg.dsp.window_type, cfg.dsp.window_len, fftbins=True)
        except ValueError:
            raise ValueError(f"Unknown window type {repr(cfg.dsp.window_type)}. Only 'sqrt_hann' and the valid window types from 'scipy.signal.get_window' are supported.")

    # Instantiate ShortTimeFFT class
    global _STFT
    _STFT = ShortTimeFFT(_win, cfg.dsp.hop_size, fs, fft_mode="onesided")

def stft(x: np.ndarray, fs: int, axis: int = -1) -> np.ndarray:
    """Computes the STFT with the settings stored in config.

    Parameters
    ----------
    x : np.ndarray
        The input signal as real or complex valued array. For complex values, the
        property `fft_mode` must be set to 'twosided' or 'centered'.
    axis : int
        The axis of `x` over which to compute the STFT.
        If not given, the last axis is used.
    """
    if _STFT is None or _fs != fs:
        _load_stft(fs)

    if _STFT is not None:
        return _STFT.stft(x, axis=axis)
    else:
        raise SystemError("Something went wrong when instantiating the STFT class")

def istft(S: np.ndarray, fs: int, f_axis: int = -2, t_axis: int = -1) -> np.ndarray:
    """Computes the Inverse STFT with the settings stored in config.

    Parameters
    ----------
    S : np.ndarray
        A complex valued array where `f_axis` denotes the frequency
        values and the `t-axis` dimension the temporal values of the
        STFT values.
    f_axis, t_axis : int
        The axes in `S` denoting the frequency and the time dimension.
    """
    if _STFT is None or _fs != fs:
        _load_stft(fs)

    if _STFT is not None:
        return _STFT.istft(S, f_axis=f_axis, t_axis=t_axis)
    else:
        raise SystemError("Something went wrong when instantiating the STFT class")

def resampling_values(fs_from: int, fs_to: int) -> tuple[int, int]:
    """
    Compute integer (interpolation, decimation) to map fs_from -> fs_to using gcd.
    Returns (L, M) such that L/M = fs_to/fs_from and both are ints.
    """
    fs_from = int(fs_from)
    fs_to   = int(fs_to)
    if fs_from <= 0 or fs_to <= 0:
        raise ValueError("Sample rates must be positive integers")
    g = math.gcd(fs_from, fs_to)
    return (fs_to // g, fs_from // g)

def complex_to_mag_phase(x):
    return np.abs(x), np.angle(x)

def db2pow(x):
    return 10**(x/10)

def db2mag(x):
    return 10**(x/20)
