from typing import Optional
import numpy as np
from scipy.signal import ShortTimeFFT, get_window

from seadge import config

# Lazy loaded cache
# STFT class
_STFT : ShortTimeFFT | None = None

def _load_stft():
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
    _STFT = ShortTimeFFT(_win, cfg.dsp.hop_size, cfg.dsp.samplerate, fft_mode="onesided")

def stft(x: np.ndarray, axis: int = -1) -> np.ndarray:
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
    if _STFT is None:
        _load_stft()

    if _STFT is not None:
        return _STFT.stft(x, axis=axis)
    else:
        raise SystemError("Something went wrong when instantiating the STFT class")

def istft(S: np.ndarray, f_axis: int = -2, t_axis: int = -1) -> np.ndarray:
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
    if _STFT is None:
        _load_stft()

    if _STFT is not None:
        return _STFT.istft(S, f_axis=f_axis, t_axis=t_axis)
    else:
        raise SystemError("Something went wrong when instantiating the STFT class")
