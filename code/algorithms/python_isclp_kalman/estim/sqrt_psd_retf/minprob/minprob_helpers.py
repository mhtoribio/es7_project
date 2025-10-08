import numpy as np
from typing import Optional, Tuple

def solve_RETFup(
    sqrtPsi_xe: np.ndarray,          # (M, N)  square root corr. matrix
    sqrtphi_s_hat: np.ndarray,       # (N,)    sqrt-PSD estimate
    Omega_hat: np.ndarray,           # (N, N)  unitary estimate
    xiThresh: float,
    reg: float,
    beta: float,
    H_hat_prior: np.ndarray,         # (M, N)
    H: Optional[np.ndarray] = None,  # (M, N)  ground-truth RETF (optional)
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    """
    Posterior RETF update (MP) + optional relative errors.

    Returns
    -------
    H_hat_post : (M, N)
    up         : (N,)  update flags (1 or 0)
    rho_H_prior_rel : Optional[float]
    rho_H_post_rel  : Optional[float]
    """
    M, N = H_hat_prior.shape
    # keep dtype (complex)
    H_hat_post = np.ones((M, N), dtype=H_hat_prior.dtype)

    # phi_s_hat = |sqrtphi_s_hat|^2  (ensure real, non-negative)
    phi_s_hat = np.real(sqrtphi_s_hat.conj() * sqrtphi_s_hat)
    xi = phi_s_hat / (np.sum(phi_s_hat) + reg)

    up = np.zeros(N, dtype=float)

    # update all rows except the first (MATLAB 2:M → Python 1:)
    for n in range(N):
        if xi[n] > xiThresh:
            up[n] = 1.0
            num = (
                sqrtPsi_xe[1:, :] @ (Omega_hat[:, n]) * np.conj(sqrtphi_s_hat[n])
                + beta * H_hat_prior[1:, n]
            )
            den = (phi_s_hat[n] + beta)
            H_hat_post[1:, n] = num / den
        else:
            up[n] = 0.0
            H_hat_post[1:, n] = H_hat_prior[1:, n]

    # relative errors (optional)
    rho_H_prior_rel = None
    rho_H_post_rel = None
    if H is not None:
        H_e_prior = H - H_hat_prior
        H_e_post  = H - H_hat_post
        denom = np.trace((H[1:, :].conj().T @ H[1:, :])).real
        # avoid division by zero
        denom = max(denom, np.finfo(float).eps)
        rho_H_prior_rel = np.trace(H_e_prior.conj().T @ H_e_prior).real / denom
        rho_H_post_rel  = np.trace(H_e_post.conj().T  @ H_e_post ).real / denom

    return H_hat_post, up, rho_H_prior_rel, rho_H_post_rel


def solve_convMP(
    H_hat: np.ndarray,              # (M, N)
    Psi_xe: np.ndarray,             # (M, M)
    phiMin: float,
    alpha: float,
    itMax: int,
    phi_s: Optional[np.ndarray] = None,   # (N,), optional GT
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Conventional MP via projected gradient descent with lower bound phiMin.

    Returns
    -------
    phi_s_hat     : (N,)
    eps_phi_s_rel : Optional[float]
    """
    M, N = H_hat.shape

    # A1 = |H^H H|^2 (elementwise), symmetric real NxN
    G = H_hat.conj().T @ H_hat
    A1 = np.abs(G) ** 2
    A2 = alpha * np.ones((N, N), dtype=A1.dtype)
    A  = A1 + A2

    # b = -Re{ diag(H^H Psi H) + alpha * Psi(0,0) * 1 }
    b_diag = np.diag(H_hat.conj().T @ Psi_xe @ H_hat)
    b = -(np.real(b_diag) + alpha * np.real(Psi_xe[0, 0]) * np.ones(N))

    # small Tikhonov-like stabilizer
    R = (1e-8 * (np.trace(A1).real / max(N, 1))) * np.eye(N, dtype=A.dtype)

    # x0 = -(A+R)\b
    try:
        x0 = -np.linalg.solve(A + R, b)
    except np.linalg.LinAlgError:
        x0 = -np.linalg.lstsq(A + R, b, rcond=None)[0]

    # Gradient of 0.5 x^T A x + b^T x  (A ≈ symmetric real)
    def grad(x: np.ndarray) -> np.ndarray:
        return A @ x + b

    # step size (spectral norm bound)
    spec = np.linalg.norm(A, 2)
    mu = 1.0 / max(spec, np.finfo(float).eps)

    for _ in range(int(itMax)):
        x_old = x0
        g = grad(x0)
        x_step = x0 - mu * g
        # projection: lower bound phiMin
        x0 = np.maximum(x_step, phiMin)
        denom = np.linalg.norm(x_old) + np.finfo(float).eps
        if np.linalg.norm(x0 - x_old) / denom < 1e-6:
            break

    phi_s_hat = x0

    eps_phi_s_rel: Optional[float] = None
    if phi_s is not None:
        e = np.sqrt(np.maximum(phi_s, 0.0)) - np.sqrt(np.maximum(phi_s_hat, 0.0))
        denom = np.sum(np.maximum(phi_s, 0.0)) + np.finfo(float).eps
        eps_phi_s_rel = float((e @ e) / denom)

    return phi_s_hat, eps_phi_s_rel


def solve_convMP_simple(
    H_hat: np.ndarray,              # (M, N)
    Psi_xe: np.ndarray,             # (M, M)
    phiMin: float,
    phi_s: Optional[np.ndarray] = None,   # (N,), optional GT
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Simple conventional MP (single linear solve + clamp).

    Returns
    -------
    phi_s_hat     : (N,)
    eps_phi_s_rel : Optional[float]
    """
    M, N = H_hat.shape

    A = np.abs(H_hat.conj().T @ H_hat) ** 2            # (N, N), real
    b = -np.real(np.diag(H_hat.conj().T @ Psi_xe @ H_hat))  # (N,)

    R = (1e-8 * (np.trace(A).real / max(N, 1))) * np.eye(N, dtype=A.dtype)

    try:
        x = -np.linalg.solve(A + R, b)
    except np.linalg.LinAlgError:
        x = -np.linalg.lstsq(A + R, b, rcond=None)[0]

    phi_s_hat = np.maximum(x, phiMin)

    eps_phi_s_rel: Optional[float] = None
    if phi_s is not None:
        e = np.sqrt(np.maximum(phi_s, 0.0)) - np.sqrt(np.maximum(phi_s_hat, 0.0))
        denom = np.sum(np.maximum(phi_s, 0.0)) + np.finfo(float).eps
        eps_phi_s_rel = float((e @ e) / denom)

    return phi_s_hat, eps_phi_s_rel


def solve_sqrtMP(
    sqrtPsi_xe: np.ndarray,        # (M, N)
    H_hat: np.ndarray,             # (M, N)
    sqrtphi_s_init: np.ndarray,    # (N,)
    alpha: float,
    itMax: int,
    phi_s: Optional[np.ndarray] = None,   # (N,), optional GT
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[np.ndarray]]:
    """
    Square-root MP with orthogonal Procrustes (via SVD).

    Returns
    -------
    sqrtphi_s_hat   : (N,)
    Omega_hat       : (N, N)
    eps_phi_s_rel   : Optional[float]
    eps_phi_s_rel_it: Optional[(T,)]
    """
    # ensure 1-D vectors
    sqrtphi_s_hat = np.asarray(sqrtphi_s_init).astype(complex).copy()
    _, N = H_hat.shape

    compute_eps = phi_s is not None
    eps_phi_s_rel_it = np.zeros(int(itMax) + 1) if compute_eps else None

    for it in range(int(itMax)):
        if compute_eps:
            e = np.sqrt(np.maximum(phi_s, 0.0)) - np.abs(sqrtphi_s_hat)
            denom = np.sum(np.maximum(phi_s, 0.0)) + np.finfo(float).eps
            eps_phi_s_rel_it[it] = float((e @ e) / denom)

        # T = (sqrtPsi_xe)^H * H_hat * diag(sqrtphi_s_hat)
        # T = sqrtPsi_xe.conj().T @ H_hat @ np.diag(sqrtphi_s_hat)
        H_scaled = H_hat * sqrtphi_s_hat[None, :]           # (M, N)
        T = sqrtPsi_xe.conj().T @ H_scaled                  # (N, N)

        # Procrustes: Omega_hat = U * V^H
        U, _, Vh = np.linalg.svd(T, full_matrices=False)
        Omega_hat = U @ Vh  # Vh = V^H

        # update sqrtphi
        sqrtphi_old = sqrtphi_s_hat.copy()

        # numerator: diag(H^H sqrtPsi_xe Omega) + alpha * (Omega^H sqrtPsi_xe[0,:]^H)
        # term1 = np.diag(H_hat.conj().T @ sqrtPsi_xe @ Omega_hat)           # (N,)
        # diag(H^H @ sqrtPsi_xe @ Omega) without forming the full product:
        X = sqrtPsi_xe @ Omega_hat                          # (M, N)
        term1 = np.sum(H_hat.conj() * X, axis=0)            # (N,)
        term2 = Omega_hat.conj().T @ sqrtPsi_xe[0, :].conj()               # (N,)
        num = term1 + alpha * term2

        # denominator: diag(H^H H) + alpha * 1
        # den = np.diag(H_hat.conj().T @ H_hat) + alpha * np.ones(N)
        # diag(H^H H) is just column norms squared:
        den = np.sum(H_hat.conj() * H_hat, axis=0).real + alpha

        sqrtphi_s_hat = num / den

        # break condition on magnitude change
        delta = np.abs(sqrtphi_s_hat) - np.abs(sqrtphi_old)
        denom = (sqrtphi_old.conj() @ sqrtphi_old).real + np.finfo(float).eps
        if (delta.conj() @ delta).real / denom < 1e-6:
            if compute_eps:
                # extend the remaining entries with the last value
                eps_phi_s_rel_it[it + 1 :] = eps_phi_s_rel_it[it]
            break

    eps_phi_s_rel = None
    if compute_eps:
        # take the last filled entry
        last_idx = min(it, int(itMax) - 1)
        eps_phi_s_rel = float(eps_phi_s_rel_it[last_idx])

    return sqrtphi_s_hat, Omega_hat, eps_phi_s_rel, eps_phi_s_rel_it
