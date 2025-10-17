#!/usr/bin/env python3
"""
Compare MATLAB arrays (from .mat) against NumPy (.npy) arrays.

Features
- Loads MATLAB v7/v7.3 .mat (scipy.io.loadmat, fallback to h5py)
- Auto variable name mapping (Psi_y_smth <-> Psi_y_STFT, etc.)
- Finds best scalar gain g* (real) and also best affine alignment (gain+DC offset)
  minimizing ||B - (g*A + c)||_F^2  (computed AFTER skipping DC bin)
- Reports RMS, RMSE, relative RMSE, max abs error, normalized correlation
- Optional plots saved to --out:
    * ORIGINAL spectrogram-style plots of A (NumPy) and B (MATLAB): Frequency (Hz) vs Time (s)
    * Residual spectrograms after gain-only and after gain+DC offset alignment
      - Assumes freq axis has length nfft//2+1 (default 257 for nfft=512)
      - If a 5-sized axis exists => channel axis; selects channel 0
      - If a 2-sized axis exists => speaker axis; plots one figure per speaker
      - **DC bin (k=0) is skipped** in both metrics and plots

Usage
  python compare_matlab_numpy.py --mat data.mat --npy-dir <OUT_DIR_FROM_YOUR_SCRIPT> --out cmp --plot --fs 16000 --nfft 512 --hop 256
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
from numpy.typing import ArrayLike

from scipy.io import loadmat
import warnings

# h5py is only needed if the .mat is v7.3 (HDF5)
try:
    import h5py  # type: ignore
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

# Defaults for plotting axes (overridden by CLI in main)
ARGS_FS = 16000
ARGS_NFFT = 512
ARGS_HOP = 256

# -----------------------------
# Utilities
# -----------------------------

def _to_complex128(x: ArrayLike) -> np.ndarray:
    """Cast to complex128 (safely handles real)."""
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128, copy=False)
    return arr.astype(np.float64, copy=False).astype(np.complex128, copy=False)

def _best_gain_real(a: np.ndarray, b: np.ndarray, nonneg: bool = False) -> float:
    """
    Best real scalar g minimizing ||b - g a||_F^2.
    For complex arrays, uses Re(<a,b>)/<a,a>.
    """
    a = _to_complex128(a)
    b = _to_complex128(b)
    num = np.vdot(a, b).real     # Re(a^H b)
    den = np.vdot(a, a).real     # a^H a  (>= 0)
    if den == 0.0:
        return 0.0
    g = num / den
    if nonneg and g < 0:
        g = 0.0
    return float(g)

def _best_gain_complex(a: np.ndarray, b: np.ndarray) -> complex:
    """Best complex scalar g minimizing ||b - g a||_F^2: g = <a,b> / <a,a>."""
    a = _to_complex128(a)
    b = _to_complex128(b)
    den = np.vdot(a, a)
    if den == 0.0:
        return 0.0 + 0.0j
    return np.vdot(a, b) / den

def _best_affine_real(a: np.ndarray, b: np.ndarray, nonneg: bool = False) -> Tuple[float, complex]:
    """
    Best real gain g and complex DC offset c minimizing ||b - (g a + c)||_F^2.
    Equivalent to linear regression with intercept:
      g = Re(<a_c, b_c>) / <a_c, a_c>,  c = mean(b) - g * mean(a),
    where a_c = a - mean(a), b_c = b - mean(b).
    Returns (g, c). c can be complex if inputs are complex.
    """
    a = _to_complex128(a).ravel()
    b = _to_complex128(b).ravel()

    a_mean = a.mean() if a.size else 0.0
    b_mean = b.mean() if b.size else 0.0
    a_c = a - a_mean
    b_c = b - b_mean

    den = np.vdot(a_c, a_c).real
    if den == 0.0:
        g = 0.0
    else:
        g = (np.vdot(a_c, b_c).real) / den
        if nonneg and g < 0:
            g = 0.0
    c = b_mean - g * a_mean
    return float(g), complex(c)

def _best_affine_complex(a: np.ndarray, b: np.ndarray) -> Tuple[complex, complex]:
    """
    Best complex gain g and complex DC offset c minimizing ||b - (g a + c)||_F^2.
    g = <a_c, b_c> / <a_c, a_c>,  c = mean(b) - g * mean(a)
    """
    a = _to_complex128(a).ravel()
    b = _to_complex128(b).ravel()

    a_mean = a.mean() if a.size else 0.0 + 0.0j
    b_mean = b.mean() if b.size else 0.0 + 0.0j
    a_c = a - a_mean
    b_c = b - b_mean

    den = np.vdot(a_c, a_c)
    if den == 0.0:
        g = 0.0 + 0.0j
    else:
        g = np.vdot(a_c, b_c) / den
    c = b_mean - g * a_mean
    return complex(g), complex(c)

def _masked(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mask NaN/Inf in either array so metrics are comparable."""
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.all(mask):
        a = a[mask]
        b = b[mask]
    return a, b

def _find_axis_by_size(shape: Tuple[int, ...], size: int) -> Optional[int]:
    """Return the first axis index with the given length, else None."""
    for i, s in enumerate(shape):
        if s == size:
            return i
    return None

def _remove_dc_bin(X: np.ndarray, nfft: int) -> np.ndarray:
    """
    Remove the DC bin (k=0) along the frequency axis if present.
    Frequency axis is identified as length nfft//2+1.
    If not found, returns X unchanged.
    """
    X = np.asarray(X)
    nfreq = nfft // 2 + 1
    ax = _find_axis_by_size(X.shape, nfreq)
    if ax is None:
        return X
    slicer = [slice(None)] * X.ndim
    slicer[ax] = slice(1, None)   # drop k=0
    return X[tuple(slicer)]

def _metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Compute metrics (on provided arrays, which are assumed to have DC removed already if desired):
    - best real gain-only residual
    - best real affine (gain + DC offset) residual
    - also report best complex gain / affine parameters (not used for residuals)
    """
    a = _to_complex128(a).ravel()
    b = _to_complex128(b).ravel()
    a, b = _masked(a, b)

    # Gains (real & complex)
    g_real = _best_gain_real(a, b, nonneg=False)
    g_real_nn = _best_gain_real(a, b, nonneg=True)
    g_cplx = _best_gain_complex(a, b)

    # Affine (gain + DC offset)
    g_aff_real, c_aff_real = _best_affine_real(a, b, nonneg=False)
    g_aff_cplx, c_aff_cplx = _best_affine_complex(a, b)

    # Residuals
    r_gain = b - g_real * a
    r_aff  = b - (g_aff_real * a + c_aff_real)

    # RMS helpers (use magnitude for complex)
    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.abs(x) ** 2))) if x.size else float("nan")

    rms_a = rms(a)
    rms_b = rms(b)
    rmse_gain = rms(r_gain)
    rmse_aff  = rms(r_aff)
    rel_rmse_gain = rmse_gain / (rms_b + 1e-20)
    rel_rmse_aff  = rmse_aff  / (rms_b + 1e-20)

    # Max abs error (gain-only residual)
    max_abs_err_gain = float(np.max(np.abs(r_gain))) if r_gain.size else float("nan")

    # Normalized correlation magnitude (scale-invariant similarity)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    corr_mag = float(np.abs(np.vdot(a, b)) / denom) if denom > 0 else float("nan")

    # Residual uniformity heuristic (detect near-constant residuals)
    r_gain_mag = np.abs(r_gain)
    uniform_ratio = float(np.std(r_gain_mag) / (np.mean(r_gain_mag) + 1e-20)) if r_gain_mag.size else float("nan")

    return {
        # Gains
        "gain_real": g_real,
        "gain_real_nonneg": g_real_nn,
        "gain_complex": [g_cplx.real, g_cplx.imag],
        # Affine (real & complex)
        "gain_affine_real": g_aff_real,
        "offset_affine_real": [c_aff_real.real, c_aff_real.imag],
        "gain_affine_complex": [g_aff_cplx.real, g_aff_cplx.imag],
        "offset_affine_complex": [c_aff_cplx.real, c_aff_cplx.imag],
        "offset_affine_abs": float(np.abs(c_aff_real)),
        # Errors
        "rms_ref": rms_b,          # treat 'b' (MATLAB) as reference
        "rms_test": rms_a,         # and 'a' (NumPy) as test
        "rmse_after_gain_real": rmse_gain,
        "rmse_after_affine_real": rmse_aff,
        "rel_rmse_real": rel_rmse_gain,
        "rel_rmse_affine_real": rel_rmse_aff,
        "max_abs_err_after_gain_real": max_abs_err_gain,
        "corr_mag": corr_mag,
        # Uniformity check (close to 0 => residual ~ constant/DC)
        "residual_gain_uniform_ratio": uniform_ratio,
        "n": int(a.size),
    }

def _maybe_permute_to_match(a: np.ndarray, target_shape: Tuple[int, ...]) -> Tuple[np.ndarray, Optional[Tuple[int, ...]]]:
    """
    If a.shape != target_shape, try axis permutations up to rank 4 to match exactly.
    Returns (possibly-permuted array, permutation or None).
    """
    if a.shape == target_shape:
        return a, None
    if a.ndim != len(target_shape):
        return a, None
    if a.ndim > 4:
        return a, None
    # Try all permutations
    import itertools
    for perm in itertools.permutations(range(a.ndim)):
        if tuple(np.array(a.shape)[list(perm)]) == target_shape:
            return np.transpose(a, axes=perm), perm
    return a, None

def _load_mat_safely(mat_path: Path) -> Dict[str, Any]:
    """
    Load a MATLAB file. Try scipy.io.loadmat first, fallback to h5py for v7.3.
    Returns a dict of variables (excluding __header__/__version__/__globals__).
    """
    # Try classic MAT first
    try:
        md = loadmat(mat_path.as_posix(), squeeze_me=False, struct_as_record=False)
        keep = {k: v for k, v in md.items() if not k.startswith("__")}
        return keep
    except NotImplementedError as e:
        # v7.3 likely
        if not HAS_H5PY:
            raise RuntimeError(f"{mat_path} looks like a v7.3 MAT-file. Install h5py to load it.") from e

    # v7.3 via h5py
    out = {}
    with h5py.File(mat_path.as_posix(), "r") as f:
        def read_ds(obj):
            # Convert HDF5 datasets into numpy arrays; handle complex split storage if present
            if isinstance(obj, h5py.Dataset):
                arr = obj[()]
                # MATLAB v7.3 complex may be stored with fields "real" and "imag"
                if arr.dtype.fields and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
                    return arr["real"] + 1j * arr["imag"]
                return arr
            elif isinstance(obj, h5py.Group):
                # Best-effort shallow load; arrays are typical for this use-case
                return None
            else:
                return None

        for k in f.keys():
            val = read_ds(f[k])
            if val is not None:
                out[k] = val
    return out

def _is_cell_array_like(x: Any) -> bool:
    """
    Heuristic: SciPy loadmat represents cell arrays as numpy object arrays.
    We consider it 'cell-like' if dtype is object and elements are arrays or scalars.
    """
    return isinstance(x, np.ndarray) and x.dtype == object

# -----------------------------
# Axes helpers for plotting
# -----------------------------

def _select_and_reorder_freq_frame(
    X: np.ndarray, nfft: int, prefer_ch0: bool = True
) -> Tuple[np.ndarray, int, int, Optional[int]]:
    """
    From X (array), extract a view where the first two axes are:
      (freq, frames), leaving any remaining axes after those (e.g., speakers).
    - Identifies freq axis as size nfft//2+1
    - Identifies channel axis as size 5 and selects channel 0 if present
    - Identifies speaker axis as size 2 (but does NOT slice it; caller will loop)
    - Picks frame axis as the largest remaining axis not in {freq, 5, 2}
    Returns (X_perm, ax_f, ax_t, ax_s) where ax_s is the index of speaker axis
    in the permuted array (or None if not present).
    """
    X = np.asarray(X)
    shape = list(X.shape)
    nfreq = nfft // 2 + 1

    ax_freq = _find_axis_by_size(tuple(shape), nfreq)
    ax_chan = _find_axis_by_size(tuple(shape), 5)
    ax_speak = _find_axis_by_size(tuple(shape), 2)

    # Frame axis: choose a dimension not equal to {nfreq,5,2} and >1; prefer largest
    candidates = [i for i, s in enumerate(shape) if s > 1 and s not in {nfreq, 5, 2}]
    ax_frame = max(candidates, key=lambda i: shape[i]) if candidates else None

    if ax_freq is None or ax_frame is None:
        raise ValueError(f"Could not identify freq/frame axes in shape {tuple(shape)}")

    # Slice channel 0 if a 5-sized axis exists
    if ax_chan is not None and prefer_ch0:
        slicer = [slice(None)] * X.ndim
        slicer[ax_chan] = 0
        X = X[tuple(slicer)]
        # Update axis indices after slicing
        if ax_speak is not None and ax_speak > ax_chan:
            ax_speak -= 1
        if ax_frame > ax_chan:
            ax_frame -= 1
        if ax_freq > ax_chan:
            ax_freq -= 1

    # Build permutation: (freq, frame, ...others...)
    other_axes = [i for i in range(X.ndim) if i not in (ax_freq, ax_frame)]
    perm = [ax_freq, ax_frame] + other_axes
    X_perm = np.transpose(X, axes=perm)

    # Speaker axis new position (if present)
    ax_s_perm = None
    if ax_speak is not None:
        ax_s_perm = perm.index(ax_speak)

    return X_perm, 0, 1, ax_s_perm

# -----------------------------
# Default variable mapping
# -----------------------------

DEFAULT_MAP = {
    # numpy_name : matlab_name
    "Psi_y_smth": "Psi_y_STFT",     # note: same content
    "phi_s_hat": "phi_s_hat",
    "H_hat_prior_STFT": "H_hat_prior_STFT",
    "H_hat_post_STFT": "H_hat_post_STFT",
    "H_update_pattern": "H_update_pattern",
    "Gamma": "Gamma",
}

# -----------------------------
# Plotting (original & residuals)
# -----------------------------

def _plot_original_freq_frame(
    var_name: str,
    X: np.ndarray,
    label: str,   # "A (NumPy)" or "B (MATLAB)"
    fs: int,
    nfft: int,
    hop: int,
    out_dir: Path,
):
    """
    Plot ORIGINAL |X| as spectrogram-style: frequency (y, Hz) vs time (x, s).
    - Picks channel 0 if a 5-channel axis exists.
    - If a 2-sized axis exists, interprets it as speaker index and produces one plot per speaker.
    - **Skips DC bin** (starts from k=1).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.warn(f"matplotlib not available, skipping original plots for {var_name}: {e}")
        return

    X = _to_complex128(np.squeeze(X))
    X_perm, _, _, ax_s = _select_and_reorder_freq_frame(X, nfft, prefer_ch0=True)

    nfreq = nfft // 2 + 1
    nframes = X_perm.shape[1]
    t_axis_end = nframes * (hop / fs)          # seconds
    f_min = fs / nfft                          # lowest plotted freq (drop DC)
    f_max = fs / 2.0

    def get_slice_for_speaker(idx_or_none):
        if idx_or_none is None:
            Xs = X_perm
        else:
            slicer = [slice(None)] * X_perm.ndim
            slicer[ax_s] = idx_or_none
            Xs = X_perm[tuple(slicer)]
        if Xs.ndim > 2:
            axes = tuple(range(2, Xs.ndim))
            Xs = np.sqrt(np.mean(np.abs(Xs) ** 2, axis=axes))
        return Xs

    speakers = [0, 1] if (ax_s is not None and X_perm.shape[ax_s] == 2) else [None]
    out_dir.mkdir(parents=True, exist_ok=True)

    for spk in speakers:
        Xs = get_slice_for_speaker(spk)
        if Xs.ndim != 2:
            raise ValueError(f"{var_name} {label}: expected 2D (freq x frames) after reduction, got {Xs.shape}")
        if Xs.shape[1] == nfreq and Xs.shape[0] != nfreq:
            Xs = Xs.T
        if Xs.shape[0] != nfreq:
            raise ValueError(f"{var_name} {label}: cannot form (freq x frames); got {Xs.shape}")

        Xs = Xs[1:, :]  # drop DC bin

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8.0, 4.2))
        im = plt.imshow(
            np.abs(Xs),
            origin="lower",
            aspect="auto",
            extent=(0.0, t_axis_end, f_min, f_max),
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        title = f"{var_name} {label}"
        if spk is not None:
            title += f" (speaker {spk})"
        title += " [ch0] (no DC)"
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = f"{var_name}_{'A' if 'NumPy' in label else 'B'}"
        if spk is not None:
            fname += f"_s{spk}"
        fname += ".png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

def _plot_residual_freq_frame(
    var_name: str,
    a: np.ndarray,
    b: np.ndarray,
    fs: int,
    nfft: int,
    hop: int,
    out_dir: Path,
):
    """
    Make residual spectrogram plots (|B - gA| and |B - (gA + c)|) with freq (Hz) vs time (s).
    - Picks channel 0 if a 5-channel axis exists.
    - If a 2-sized axis exists, interprets it as speaker index and produces one plot per speaker.
    - **Skips DC bin** in residual display AND computes g, c from DC-removed arrays.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.warn(f"matplotlib not available, skipping residual plots for {var_name}: {e}")
        return

    a = _to_complex128(np.squeeze(a))
    b = _to_complex128(np.squeeze(b))

    # Compute alignment parameters on DC-removed arrays
    a_fit = _remove_dc_bin(a, nfft)
    b_fit = _remove_dc_bin(b, nfft)
    g_gain = _best_gain_real(a_fit, b_fit, nonneg=False)
    g_aff, c_aff = _best_affine_real(a_fit, b_fit, nonneg=False)

    # Residuals (full arrays), then arrange to (freq, frames, ...)
    r_gain = b - g_gain * a
    r_aff  = b - (g_aff * a + c_aff)

    Rg_perm, _, _, ax_s_g = _select_and_reorder_freq_frame(r_gain, nfft, prefer_ch0=True)
    Ra_perm, _, _, ax_s_a = _select_and_reorder_freq_frame(r_aff,  nfft, prefer_ch0=True)

    nfreq = nfft // 2 + 1
    nframes = Rg_perm.shape[1]
    t_axis_end = nframes * (hop / fs)
    f_min = fs / nfft
    f_max = fs / 2.0

    def get_slice(Rp, ax_s, spk):
        if spk is None:
            Rs = Rp
        else:
            slicer = [slice(None)] * Rp.ndim
            slicer[ax_s] = spk
            Rs = Rp[tuple(slicer)]
        if Rs.ndim > 2:
            axes = tuple(range(2, Rs.ndim))
            Rs = np.sqrt(np.mean(np.abs(Rs) ** 2, axis=axes))
        if Rs.ndim != 2:
            raise ValueError(f"{var_name}: expected 2D residual, got {Rs.shape}")
        if Rs.shape[1] == nfreq and Rs.shape[0] != nfreq:
            Rs = Rs.T
        if Rs.shape[0] != nfreq:
            raise ValueError(f"{var_name}: cannot form (freq x frames) residual; got {Rs.shape}")
        return Rs

    speakers = [0, 1] if (ax_s_g is not None and Rg_perm.shape[ax_s_g] == 2) else [None]
    out_dir.mkdir(parents=True, exist_ok=True)

    for spk in speakers:
        Rs_gain = get_slice(Rg_perm, ax_s_g, spk)[1:, :]  # drop DC
        Rs_aff  = get_slice(Ra_perm, ax_s_a, spk)[1:, :]  # drop DC

        # Gain-only residual
        fig = plt.figure(figsize=(8.0, 4.2))
        im = plt.imshow(
            np.abs(Rs_gain),
            origin="lower",
            aspect="auto",
            extent=(0.0, t_axis_end, f_min, f_max),
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        title = f"{var_name} |B - g·A|"
        if spk is not None:
            title += f" (speaker {spk})"
        title += " [ch0] (no DC)"
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = f"{var_name}_residual_gain"
        if spk is not None:
            fname += f"_s{spk}"
        fname += ".png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

        # Affine residual
        fig = plt.figure(figsize=(8.0, 4.2))
        im = plt.imshow(
            np.abs(Rs_aff),
            origin="lower",
            aspect="auto",
            extent=(0.0, t_axis_end, f_min, f_max),
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        title = f"{var_name} |B - (g·A + c)|"
        if spk is not None:
            title += f" (speaker {spk})"
        title += " [ch0] (no DC)"
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = f"{var_name}_residual_affine"
        if spk is not None:
            fname += f"_s{spk}"
        fname += ".png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

# -----------------------------
# Main comparison
# -----------------------------

DEFAULT_MAP = {
    # numpy_name : matlab_name
    "Psi_y_smth": "Psi_y_STFT",
    "phi_s_hat": "phi_s_hat",
    "H_hat_prior_STFT": "H_hat_prior_STFT",
    "H_hat_post_STFT": "H_hat_post_STFT",
    "H_update_pattern": "H_update_pattern",
    "Gamma": "Gamma",
}

def compare_one(
    npy_path: Path,
    mat_vars: Dict[str, Any],
    matlab_name: str,
    real_gain_only: bool = True,
    allow_axis_permutation: bool = True,
    plots_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compare one pair: numpy .npy file vs variable in loaded MATLAB dict.
    Returns metrics and any notes (e.g., used permutation).
    """
    if not npy_path.exists():
        return {"error": f"missing npy {npy_path.name}"}

    if matlab_name not in mat_vars:
        return {"error": f"missing MATLAB var '{matlab_name}' in .mat"}

    a = np.load(npy_path.as_posix(), allow_pickle=True)
    b = mat_vars[matlab_name]

    # Handle MATLAB cell arrays (best effort)
    if _is_cell_array_like(b):
        return {"error": f"MATLAB var '{matlab_name}' is a cell array (object). Export as numeric/complex array to compare."}

    # Ensure numpy array for MATLAB var
    b = np.array(b)

    # Squeeze length-1 dims for more tolerant matching
    a = np.squeeze(a)
    b = np.squeeze(b)

    # Try axis permutation to match shapes
    perm_used = None
    if allow_axis_permutation and a.shape != b.shape:
        a2, perm = _maybe_permute_to_match(a, b.shape)
        if perm is not None:
            a = a2
            perm_used = perm

    # Final shape check
    if a.shape != b.shape:
        return {"error": f"shape mismatch: npy {a.shape} vs mat {b.shape}"}

    # ---- Metrics computed AFTER dropping the DC bin (if freq axis exists) ----
    a_ndc = _remove_dc_bin(a, ARGS_NFFT)
    b_ndc = _remove_dc_bin(b, ARGS_NFFT)
    m = _metrics(a_ndc, b_ndc)

    # Optional spectrogram-style plots (original A/B and residuals), all without DC row
    if plots_dir is not None:
        try:
            # ORIGINALS
            _plot_original_freq_frame(
                var_name=npy_path.stem, X=a, label="A (NumPy)",
                fs=ARGS_FS, nfft=ARGS_NFFT, hop=ARGS_HOP, out_dir=plots_dir
            )
            _plot_original_freq_frame(
                var_name=npy_path.stem, X=b, label="B (MATLAB)",
                fs=ARGS_FS, nfft=ARGS_NFFT, hop=ARGS_HOP, out_dir=plots_dir
            )
            # RESIDUALS (gain-only and affine), params fit on DC-removed arrays
            _plot_residual_freq_frame(
                var_name=npy_path.stem, a=a, b=b,
                fs=ARGS_FS, nfft=ARGS_NFFT, hop=ARGS_HOP, out_dir=plots_dir
            )
        except Exception as e:
            warnings.warn(f"Plot skipped for {npy_path.stem}: {e}")

    out = {
        "npy_file": npy_path.name,
        "mat_var": matlab_name,
        "shape": list(a.shape),
        "perm_used": list(perm_used) if perm_used is not None else None,
        **m,
    }
    return out

def main():
    p = argparse.ArgumentParser(description="Compare MATLAB arrays (.mat) with NumPy arrays (.npy).")
    p.add_argument("--mat", required=True, type=Path, help="Path to MATLAB .mat file (e.g., data.mat)")
    p.add_argument("--npy-dir", required=True, type=Path, help="Directory containing .npy files to compare")
    p.add_argument("--out", type=Path, default=Path("compare_out"), help="Output directory for report/plots")
    p.add_argument("--plot", action="store_true", help="Save ORIGINAL and residual spectrogram plots (freq vs time); DC skipped")
    p.add_argument("--nonneg-gain", action="store_true", help="(report only) also compute best non-negative real gain")
    p.add_argument("--no-permute", action="store_true", help="Do not attempt axis permutations to match shapes")
    p.add_argument("--map", nargs="*", default=[], metavar="NPY=MAT",
                   help="Override/add name mappings, e.g., Psi_y_smth=Psi_y_STFT")
    p.add_argument("--only", nargs="*", default=[],
                   help="Compare only these NPY basenames (without .npy). Defaults to known set + any files present.")
    # Axes parameters for plotting
    p.add_argument("--fs", type=int, default=16000, help="Sampling rate [Hz] for axis labels")
    p.add_argument("--nfft", type=int, default=512, help="FFT length (defines #freq bins nfft//2+1)")
    p.add_argument("--hop", type=int, default=256, help="STFT hop size [samples] for time axis")
    args = p.parse_args()

    # Make plotting params available to compare_one
    global ARGS_FS, ARGS_NFFT, ARGS_HOP
    ARGS_FS = int(args.fs)
    ARGS_NFFT = int(args.nfft)
    ARGS_HOP = int(args.hop)

    out_dir = args.out
    plots_dir = (out_dir / "plots") if args.plot else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Build name map
    name_map = dict(DEFAULT_MAP)
    for spec in args.map:
        if "=" not in spec:
            raise SystemExit(f"--map entries must be NPY=MAT, got: {spec}")
        npy_name, mat_name = spec.split("=", 1)
        name_map[npy_name] = mat_name

    # Load MATLAB variables
    mat_vars = _load_mat_safely(args.mat)

    # Decide which npy files to consider
    present_npy = {p.stem: p for p in args.npy_dir.glob("*.npy")}
    if args.only:
        candidates = {k: present_npy[k] for k in args.only if k in present_npy}
        missing = [k for k in args.only if k not in present_npy]
        if missing:
            print(f"Warning: requested --only {missing} not found in {args.npy_dir}")
    else:
        # default to intersection of known names + whatever exists
        default_keys = set(name_map.keys())
        intersect = default_keys & set(present_npy.keys())
        if not intersect:
            # if nothing intersects, compare all .npy files against same-named MATLAB vars
            for k, path in present_npy.items():
                name_map.setdefault(k, k)
            intersect = set(present_npy.keys())
        candidates = {k: present_npy[k] for k in sorted(intersect)}

    # Compare
    report: Dict[str, Any] = {
        "mat_file": args.mat.as_posix(),
        "npy_dir": args.npy_dir.as_posix(),
        "results": {},
    }

    for npy_name, npy_path in candidates.items():
        mat_name = name_map.get(npy_name, npy_name)
        res = compare_one(
            npy_path=npy_path,
            mat_vars=mat_vars,
            matlab_name=mat_name,
            real_gain_only=True,
            allow_axis_permutation=not args.no_permute,
            plots_dir=plots_dir,
        )
        report["results"][npy_name] = res

        # Pretty print short line
        if "error" in res:
            print(f"[{npy_name}] ERROR: {res['error']}")
        else:
            # Short print including DC-removed affine offset magnitude and affine RMSE
            print(
                f"[{npy_name}] ok shape={tuple(res['shape'])} "
                f"g_real={res['gain_real']:.6g} "
                f"|c|={res['offset_affine_abs']:.6g} "
                f"rmse_gain={res['rmse_after_gain_real']:.6g} "
                f"rmse_aff={res['rmse_after_affine_real']:.6g} "
                f"rel_rmse={res['rel_rmse_real']:.3e} "
                f"corr={res['corr_mag']:.6g}"
                + (f" perm={tuple(res['perm_used'])}" if res['perm_used'] else "")
            )

    # Save JSON report
    json_path = out_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report -> {json_path}")
    if plots_dir:
        print(f"Saved plots  -> {plots_dir}")

if __name__ == "__main__":
    main()
