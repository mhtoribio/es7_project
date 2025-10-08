import numpy as np

def estim_corrmat(X: np.ndarray, alpha: float):
    """
    Estimate correlation matrices from STFT data.

    Parameters
    ----------
    X : np.ndarray
        STFT data, shape (freqbins, frames, channels). If X has extra
        trailing singleton dims (e.g., (F, L, M, 1)), they are ignored.
        Complex dtype is supported.
    alpha : float
        Forgetting factor.

    Returns
    -------
    Psi_x_smth : np.ndarray
        Smoothed correlation matrix estimate.
        - If alpha != 1: shape (freqbins, frames, channels, channels)
        - If alpha == 1: shape (freqbins, 1,      channels, channels)
    Psi_x_mean : np.ndarray
        Mean correlation matrix over frames, shape (freqbins, 1, channels, channels)
    """
    X = np.asarray(X)
    if X.ndim < 3:
        raise ValueError("X must have at least 3 dims: (freqbins, frames, channels).")
    if X.ndim > 3:
        # Allow only trailing singleton dims (like MATLAB's ~ = 1)
        if int(np.prod(X.shape[3:])) != 1:
            raise ValueError("Only trailing singleton dims beyond the 3rd are allowed.")
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    N_half, L, M = X.shape
    real_dtype = np.real(X).dtype
    eps = np.finfo(real_dtype).eps

    # --- Average PSD (correlation) matrix ---
    if alpha != 1:
        # Instantaneous per-(k,l) correlation: R_inst[k,l] = x x^H
        # einsum builds all outer products in one shot: (F,L,M) x (F,L,M) -> (F,L,M,M)
        R_inst = np.einsum('klm,kln->klmn', X, X.conj(), optimize=True)
        Psi_x_mean = R_inst.sum(axis=1, keepdims=True) / L  # (F,1,M,M)
    else:
        # Same mean, computed without storing R_inst
        R_mean = np.einsum('klm,kln->kmn', X, X.conj(), optimize=True) / L  # (F,M,M)
        Psi_x_mean = R_mean[:, None, :, :]  # (F,1,M,M)

    # --- Smoothed PSD (exponential smoothing over frames) ---
    R_smth_tmp = (eps * np.ones((N_half, 1, M, M), dtype=X.dtype))
    if alpha != 1:
        Psi_x_smth = np.empty((N_half, L, M, M), dtype=X.dtype)
        for l in range(L):
            # Match dims: R_inst[:, l, :, :] -> (F,M,M) -> (F,1,M,M)
            R_smth_tmp = alpha * R_smth_tmp + (1 - alpha) * R_inst[:, l:l+1, :, :]
            Psi_x_smth[:, l:l+1, :, :] = R_smth_tmp
    else:
        # With alpha == 1, "smoothing" degenerates to the mean
        Psi_x_smth = Psi_x_mean

    return Psi_x_smth, Psi_x_mean

