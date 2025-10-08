#!/usr/bin/env python3
# Compares two wav files and returns statistics about them

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np

# Optional imports
try:
    import soundfile as sf  # type: ignore
    HAVE_SF = True
except Exception:
    HAVE_SF = False

try:
    from scipy.io import wavfile
    from scipy.signal import resample_poly, correlate
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


@dataclass
class Audio:
    sr: int
    data: np.ndarray  # shape (N,) mono float64 in [-1, 1] approximately
    orig_channels: int
    path: str
    loader: str


def _normalize_int_array(x: np.ndarray) -> np.ndarray:
    """Convert signed/unsigned integer PCM to float64 in [-1, 1) approximately.
    Detect SciPy's 24-bit left-justified-in-int32 case and scale correctly.
    """
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("Expected integer dtype")

    if x.dtype == np.uint8:
        # Unsigned 8-bit: [0, 255] -> center to zero and scale
        return (x.astype(np.float64) - 128.0) / 128.0

    info = np.iinfo(x.dtype)
    x32 = x.astype(np.int64, copy=False)  # avoid overflow in abs

    if x.dtype == np.int32:
        # Heuristic: check if the least-significant 8 bits are all zero (typical 24-bit packed into int32, left-justified)
        sample = x[:min(len(x), 100000)]
        if np.all((sample & 0xFF) == 0):
            # Scale as left-justified 24-bit: divide by 2**31 (equiv. shift right 8 then divide by 2**23)
            return x.astype(np.float64) / (2**31)
        # Otherwise assume true 32-bit PCM:
        return x.astype(np.float64) / max(abs(info.min), info.max)

    # Generic signed scaling (int16, int24 via int32 handled above, int64 rare):
    return x32.astype(np.float64) / max(abs(info.min), info.max)


def _to_mono(x: np.ndarray, mode: str) -> np.ndarray:
    """x shape (N, C). mode: 'mean', 'left', 'right', 'chN' (0-based N)"""
    if x.ndim == 1:
        return x.astype(np.float64, copy=False)

    C = x.shape[1]
    if mode == "mean":
        return x.mean(axis=1, dtype=np.float64)
    elif mode == "left":
        return x[:, 0].astype(np.float64, copy=False)
    elif mode == "right":
        if C < 2:
            return x[:, 0].astype(np.float64, copy=False)
        return x[:, 1].astype(np.float64, copy=False)
    elif mode.startswith("ch"):
        try:
            idx = int(mode[2:])
        except ValueError:
            raise ValueError(f"Invalid channel specifier: {mode}")
        if not (0 <= idx < C):
            raise ValueError(f"Channel index out of range: 0..{C-1}")
        return x[:, idx].astype(np.float64, copy=False)
    else:
        raise ValueError(f"Unknown mono mode: {mode}")


def _resample_if_needed(x: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return x, sr
    if not HAVE_SCIPY:
        die(f"Sample rates differ ({sr} vs {target_sr}) and SciPy not available for resampling. Install scipy or use --no-resample to forbid it.")
    # Use a rational approximation to resample with integer up/down
    frac = Fraction(target_sr, sr).limit_denominator(100000)
    up, down = frac.numerator, frac.denominator
    y = resample_poly(x, up, down, padtype="line")
    return y, target_sr


def load_audio(path: str, target_sr: Optional[int], mono: str) -> Audio:
    """Load audio as float64 mono. Prefer soundfile; fallback to SciPy wavfile."""
    if HAVE_SF:
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        loader = "soundfile"
        C = data.shape[1]
        y = _to_mono(data, mono)
        if target_sr is not None:
            y, sr = _resample_if_needed(y, sr, target_sr)
        return Audio(sr=sr, data=y, orig_channels=C, path=path, loader=loader)

    if not HAVE_SCIPY:
        die("Neither soundfile nor scipy is available. Please `pip install soundfile` or `pip install scipy`.")

    sr, data = wavfile.read(path, mmap=False)
    loader = "scipy"
    if data.ndim == 1:
        # int/float mono
        if np.issubdtype(data.dtype, np.integer):
            y = _normalize_int_array(data)
        else:
            y = data.astype(np.float64, copy=False)
        C = 1
    else:
        # shape (N, C)
        C = data.shape[1]
        if np.issubdtype(data.dtype, np.integer):
            data = _normalize_int_array(data)
        else:
            data = data.astype(np.float64, copy=False)
        y = _to_mono(data, mono)

    if target_sr is not None:
        y, sr = _resample_if_needed(y, sr, target_sr)
    return Audio(sr=sr, data=y, orig_channels=C, path=path, loader=loader)


def trim_to_overlap(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Given lag where b lags a by 'lag' samples (>0 meaning b starts later),
    return overlapped slices of a and b of equal length.
    """
    Na, Nb = len(a), len(b)
    if lag >= 0:
        start_a = lag
        start_b = 0
    else:
        start_a = 0
        start_b = -lag
    end = min(Na - start_a, Nb - start_b)
    if end <= 0:
        return np.array([], dtype=a.dtype), np.array([], dtype=b.dtype)
    return a[start_a:start_a+end], b[start_b:start_b+end]


def estimate_lag_xcorr(a: np.ndarray, b: np.ndarray, maxlag: Optional[int]) -> Tuple[int, float]:
    """Return (lag, rmax). Uses FFT-based full xcorr unless maxlag is set to limit lags."""
    if not HAVE_SCIPY:
        die("SciPy is required for cross-correlation. Please `pip install scipy`.")
    # Optionally restrict to maxlag by padding windows to reduce memory
    if maxlag is not None:
        # Build windows to search around potential alignment
        # Use valid region by zero-padding b relative to a
        pad = maxlag
        a_pad = np.pad(a, (pad, pad))
        c = correlate(a_pad, b, mode="valid", method="fft")
        # c length = len(a)+2*pad - len(b) + 1 ; we want index relative to pad
        idx = int(np.argmax(c))
        lag = idx - pad
        r = c[idx] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20)
        return lag, float(r)
    else:
        c = correlate(a, b, mode="full", method="fft")
        idx = int(np.argmax(c))
        lag = idx - (len(b) - 1)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20)
        r = c[idx] / denom
        return lag, float(r)


def main():
    p = argparse.ArgumentParser(description="Compare two WAV files with lag/gain alignment and residual SNR.")
    p.add_argument("A")
    p.add_argument("B")
    p.add_argument("--mono", default="mean", help="Downmix mode: mean|left|right|chN (default: mean)")
    p.add_argument("--no-resample", action="store_true", help="Do not resample; error if sample rates differ")
    p.add_argument("--max-seconds", type=float, default=None, help="Limit analysis to first N seconds (faster)")
    p.add_argument("--maxlag-s", type=float, default=None, help="Limit cross-correlation search to ±N seconds")
    p.add_argument("--save-diff", type=str, default=None, help="Path to write aligned residual (float32 WAV)")
    p.add_argument("--save-aligned", type=str, default=None, help="Prefix to write aligned A/B as WAVs (prefix_A.wav, prefix_B.wav)")
    args = p.parse_args()

    # Load A first to determine target sr if we resample B to match A
    A = load_audio(args.A, target_sr=None, mono=args.mono)
    target_sr = None if args.no_resample else A.sr
    B = load_audio(args.B, target_sr=target_sr, mono=args.mono)

    if args.no_resample and A.sr != B.sr:
        die(f"Sample-rate mismatch and --no-resample set: {A.sr} vs {B.sr}")

    # Optional time limit
    if args.max_seconds is not None and args.max_seconds > 0:
        NmaxA = int(args.max_seconds * A.sr)
        NmaxB = int(args.max_seconds * B.sr)
        A.data = A.data[:NmaxA]
        B.data = B.data[:NmaxB]

    # Remove DC
    a = A.data - A.data.mean()
    b = B.data - B.data.mean()

    # Estimate lag (B relative to A)
    maxlag = None if args.maxlag_s is None else int(round(args.maxlag_s * A.sr))
    lag, r = estimate_lag_xcorr(a, b, maxlag=maxlag)

    # Align to overlap
    a_al, b_al = trim_to_overlap(a, b, lag)
    if len(a_al) == 0 or len(b_al) == 0:
        die("No overlap after alignment. Files may be too short or lag too large.")

    # Best scalar gain g to match A ≈ g * B
    bb = float(b_al @ b_al)
    g = 0.0 if bb == 0.0 else float((a_al @ b_al) / bb)

    # Residual & SNR
    res = a_al - g * b_al
    num = float(a_al @ a_al)
    den = float(res @ res) + 1e-20
    snr_db = 10.0 * np.log10(num / den)

    # Peak & RMS of residual (useful when not dB-minded)
    rms_res = float(np.sqrt(np.mean(res**2)))
    peak_res = float(np.max(np.abs(res)))

    # Report
    print("=== WAV Compare ===")
    print(f"A: {A.path} | loader={A.loader} | sr={A.sr} Hz | ch={A.orig_channels} | N={len(A.data)}")
    print(f"B: {B.path} | loader={B.loader} | sr={B.sr} Hz | ch={B.orig_channels} | N={len(B.data)}")
    print(f"Lag (B relative to A): {lag} samples  ({lag / A.sr * 1000:.3f} ms)")
    print(f"Max normalized correlation r: {r:.8f}")
    print(f"Best gain g (A ≈ g·B): {g:.8f}")
    print(f"Residual SNR: {snr_db:.2f} dB")
    print(f"Residual RMS: {rms_res:.6g}, peak: {peak_res:.6g}")
    print("Heuristics:")
    print("  • If |lag| ≈ 0, r ≈ 1.0, and SNR > 60–80 dB ⇒ signals are effectively identical (up to gain).")
    print("  • If g ≈ 1.0 but SNR low ⇒ content differs; listen to residual.")
    print("  • If lag large ⇒ files are time-shifted; SNR should improve after alignment (we already align).")

    # Optional writes
    if args.save_diff or args.save_aligned:
        if not HAVE_SF:
            print("NOTE: soundfile not available, cannot write WAVs. Install with `pip install soundfile` to enable writing.", file=sys.stderr)
        else:
            if args.save_diff:
                import soundfile as sf
                sf.write(args.save_diff, res.astype(np.float32), A.sr, subtype="FLOAT")
                print(f"Wrote residual diff to: {args.save_diff}")
            if args.save_aligned:
                import soundfile as sf
                prefix = args.save_aligned
                sf.write(f"{prefix}_A.wav", a_al.astype(np.float32), A.sr, subtype="FLOAT")
                sf.write(f"{prefix}_B.wav", (g*b_al).astype(np.float32), A.sr, subtype="FLOAT")
                print(f"Wrote aligned A/B to: {prefix}_A.wav , {prefix}_B.wav")


if __name__ == "__main__":
    main()
