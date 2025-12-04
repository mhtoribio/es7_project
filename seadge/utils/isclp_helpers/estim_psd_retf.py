import numpy as np

from .minprob_helpers import solve_RETFup, solve_convMP, solve_convMP_simple, solve_sqrtMP

def estim_psd_retf(
    P_STFT: np.ndarray,           # (F, L, M, M) eigenvectors, columns are eigenvectors
    lambda_STFT: np.ndarray,      # (F, L, M)    eigenvalues
    Gamma_FT: np.ndarray,         # (F, M, M) or (F, 1, M, M)
    H_init_FT: np.ndarray,        # (F, 1, M, N) or (F, L, M, N) – uses first frame as init
    *,
    itmax: int = 20,
    phi_min: float = 10.0 ** (-80.0 / 10.0),
    method: str = "square-root MP",          # {"conventional MP", "square-root MP"}
    alpha: np.ndarray | float | None = None, # scalar or (F,)
    beta:  np.ndarray | float | None = None, # scalar or (F,)
    xi_thresh: float | None = None,
):
    """
    Estimate early speech PSDs and (optionally) update RETFs.

    Returns
    -------
    phi_s_hat_STFT : (F, L, N)               early PSD estimates per source
    phi_d_STFT     : (F, L)                   diffuse PSD estimate
    H_hat_prior_STFT : (F, L, M, N) or None   prior RETF (square-root MP)
    H_hat_post_STFT  : (F, L, M, N) or None   posterior RETF (square-root MP)
    update_RETF      : (F, L, N) or None      RETF update flags (square-root MP)
    """
    # --- shape checks / extraction
    F, L, M = lambda_STFT.shape
    if P_STFT.shape[:3] != (F, L, M) or P_STFT.shape[3] != M:
        raise ValueError("P_STFT must have shape (F, L, M, M) with eigenvectors as columns.")
    if H_init_FT.ndim != 4 or H_init_FT.shape[0] != F or H_init_FT.shape[2] != M:
        raise ValueError("H_init_FT must be (F, 1, M, N) or (F, L, M, N).")
    N = H_init_FT.shape[3]

    # --- per-frequency alpha/beta (like MATLAB alpha(k), beta(k))
    def _to_freq_vec(x, default_val):
        if x is None:
            return np.full(F, default_val, dtype=float)
        if np.isscalar(x):
            return np.full(F, float(x), dtype=float)
        xv = np.asarray(x, dtype=float).reshape(-1)
        if xv.shape != (F,):
            raise ValueError(f"alpha/beta must be scalar or length-{F} vector, got {xv.shape}.")
        return xv

    alpha = _to_freq_vec(alpha, 1e3)
    beta  = _to_freq_vec(beta,  1e3)

    # --- helpers to slice per-frequency items
    def _gamma_k(k: int) -> np.ndarray:
        if Gamma_FT.ndim == 3:      # (F, M, M)
            return Gamma_FT[k]
        elif Gamma_FT.ndim == 4:    # (F, 1, M, M)
            return Gamma_FT[k, 0]
        else:
            raise ValueError("Gamma_FT must be (F, M, M) or (F, 1, M, M).")

    def _H0_k(k: int) -> np.ndarray:
        # MATLAB: squeeze(H_init_FT(k,1,:,:))
        return H_init_FT[k, 0]  # (M, N)

    # --- outputs
    real_dtype = np.result_type(lambda_STFT.dtype, np.float32)
    cplx_dtype = np.result_type(P_STFT.dtype,     np.complex64)

    phi_s_hat_STFT = np.zeros((F, L, N), dtype=real_dtype)
    phi_d_STFT     = np.zeros((F, L),     dtype=real_dtype)

    if method == "square-root MP":
        H_hat_prior_STFT = np.zeros((F, L, M, N), dtype=cplx_dtype)
        H_hat_post_STFT  = np.zeros((F, L, M, N), dtype=cplx_dtype)
        update_RETF      = np.zeros((F, L, N),     dtype=real_dtype)
    elif method == "conventional MP":
        H_hat_prior_STFT = None
        H_hat_post_STFT  = None
        update_RETF      = None
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # --- main loops (skip DC bin: MATLAB k = 2:N_FT_half)
    for k in range(1, F):
        H_hat_post = _H0_k(k).copy()   # (M, N)
        Gamma_k    = _gamma_k(k)       # (M, M)

        for l in range(L):
            P_kl   = P_STFT[k, l]          # (M, M), eigenvectors in columns
            lam_kl = lambda_STFT[k, l]     # (M,)

            # sort eigenpairs by descending eigenvalue
            idx = np.argsort(lam_kl)[::-1]
            Pmax      = P_kl[:, idx[:N]]   # (M, N)
            lambdaMax = lam_kl[idx[:N]]    # (N,)
            lambdaMin = lam_kl[idx[N:]]    # (M-N,)

            # diffuse PSD (mean of residual eigenvalues)
            if lambdaMin.size:
                phi_d = float(lambdaMin.mean())
            else:
                # degenerate N==M → fallback
                phi_d = float(lambdaMax.min()) if lambdaMax.size else 0.0
            phi_d_STFT[k, l] = phi_d

            # early eigenvalues with floor
            lambda_xe = np.maximum(lambdaMax - phi_d, phi_min)

            # Psi_xe = (Gamma_k Pmax diag(sqrtlambda)) (.)^H
            sqrtlambda = np.sqrt(lambda_xe)
            sqrtPsi_xe = (Gamma_k @ Pmax) @ np.diag(sqrtlambda)     # (M, N)
            Psi_xe     = sqrtPsi_xe @ sqrtPsi_xe.conj().T           # (M, M), Hermitian

            if method == "conventional MP":
                phi_s_hat, _ = solve_convMP(H_hat_post, Psi_xe, phi_min, float(alpha[k]), int(itmax))
                phi_s_hat_STFT[k, l] = phi_s_hat
                # no RETF tracking in this mode

            else:  # "square-root MP"
                H_hat_prior = H_hat_post

                # init via simple MP
                phi_init, _   = solve_convMP_simple(H_hat_prior, Psi_xe, phi_min)
                sqrtphi_init  = np.sqrt(np.maximum(phi_init, phi_min))

                # square-root MP (Procrustes)
                sqrtphi_hat, Omega_hat, _, _ = solve_sqrtMP(
                    sqrtPsi_xe, H_hat_prior, sqrtphi_init, float(alpha[k]), int(itmax)
                )
                phi_s_hat = np.abs(sqrtphi_hat) ** 2
                phi_s_hat_STFT[k, l] = phi_s_hat

                # RETF update after warmup (l > 16)
                if l > 16:
                    if xi_thresh is None:
                        raise ValueError("xi_thresh must be provided for RETF updates.")
                    phi_reg = phi_d + 1e-3
                    H_hat_post, up, _, _ = solve_RETFup(
                        sqrtPsi_xe, sqrtphi_hat, Omega_hat, float(xi_thresh),
                        float(phi_reg), float(beta[k]), H_hat_prior
                    )
                else:
                    up = np.zeros(N, dtype=real_dtype)
                    H_hat_post = H_hat_prior

                # save RETFs / updates
                H_hat_prior_STFT[k, l] = H_hat_prior
                H_hat_post_STFT[k,  l] = H_hat_post
                update_RETF[k, l]      = up

    return phi_s_hat_STFT, phi_d_STFT, H_hat_prior_STFT, H_hat_post_STFT, update_RETF
