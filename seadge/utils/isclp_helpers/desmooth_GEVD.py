import numpy as np
from numpy.typing import NDArray

try:
    # SciPy is needed for the generalized eigenvalue problem A v = w B v
    from scipy.linalg import eig  # not eigh: B may be singular / not PD
except Exception as e:  # pragma: no cover
    raise ImportError("desmooth_GEVD requires scipy.linalg.eig") from e


def desmooth_GEVD(
    Psi_x_STFT: NDArray,   # (F, L, M, M)
    Gamma_FT: NDArray,     # (F, 1, M, M)
    lambdaMin: float,
    forgetPSD: float,
):
    """
    Python/Numpy translation of MATLAB `desmooth_GEVD`.

    Parameters
    ----------
    Psi_x_STFT : array_like, shape (F, L, M, M)
        Correlation matrices per frequency bin and frame.
    Gamma_FT : array_like, shape (F, 1, M, M)
        Diffuse coherence matrices per frequency bin.
    lambdaMin : float
        Eigenvalue floor after de-smoothing.
    forgetPSD : float
        Forgetting factor in [0,1). (1.0 is invalid: division by zero)

    Returns
    -------
    P_STFT : complex ndarray, shape (F, L, M, M)
        Generalized eigenvectors.
    lambda_STFT : float ndarray, shape (F, L, M)
        De-smoothed eigenvalues.
    lambda_LP_STFT : float ndarray, shape (F, L, M)
        “Low-pass” (smoothed / GEVD instantaneous here) eigenvalues.
    """
    F, L, M, M2 = Psi_x_STFT.shape
    assert M == M2, "Last two dims of Psi_x_STFT must be (M, M)"
    assert Gamma_FT.shape[0] == F and Gamma_FT.shape[2:] == (M, M), \
        "Gamma_FT must have shape (F, 1, M, M)"

    if forgetPSD >= 1.0:
        raise ValueError("forgetPSD must be < 1.0")

    # Outputs
    P_STFT         = np.zeros((F, L, M, M), dtype=np.complex128)
    lambda_STFT    = np.zeros((F, L, M), dtype=np.float64)
    lambda_LP_STFT = np.zeros((F, L, M), dtype=np.float64)

    tiny = np.finfo(float).tiny

    # Loop over frequency bins (skip k=0 to mimic MATLAB's 1-based k=2:N)
    for k in range(1, F):
        # Gamma for this frequency
        Gamma = Gamma_FT[k, 0, :, :]

        P_old = None

        for l in range(L):
            # --- GEVD ---
            Psi_x = Psi_x_STFT[k, l, :, :]

            # Solve A v = w B v
            # scipy.linalg.eig returns (w, v), with v columns = eigenvectors
            w, V = eig(Psi_x, Gamma)

            # Keep real part of eigenvalues (as in MATLAB code)
            lambda_LP = np.real(w)

            # Rescale P so that P^H * Gamma * P has unit diagonal
            # (column-wise normalization)
            GPP = V.conj().T @ Gamma @ V
            d = np.real(np.diag(GPP))
            d = np.maximum(d, tiny)
            V = V / np.sqrt(d)[None, :]

            # --- Sort eigenvectors/eigenvalues to track subspace over frames ---
            if P_old is not None:
                # order start: descending by current eigenvalues
                order = np.argsort(lambda_LP)[::-1]

                # Q ≈ I if columns are aligned; use it to map columns
                Q = np.abs(P_old.conj().T @ Gamma @ V)

                V_tmp = V.copy()
                lambda_tmp = lambda_LP.copy()

                # Greedy assignment: for each column (in eigenvalue order),
                # pick the best matching previous column (unique per step).
                assigned_rows = np.zeros(M, dtype=bool)
                V_new = np.empty_like(V)
                lambda_new = np.empty_like(lambda_LP)

                for m in order:
                    # find best row for column m among unassigned rows
                    q_col = Q[:, m].copy()
                    q_col[assigned_rows] = -np.inf
                    r = int(np.argmax(q_col))
                    assigned_rows[r] = True
                    V_new[:, r] = V_tmp[:, m]
                    lambda_new[r] = lambda_tmp[m]

                V = V_new
                lambda_LP = lambda_new

            P_old = V

            # Save P and lambda_LP
            P_STFT[k, l, :, :] = V
            lambda_LP_STFT[k, l, :] = lambda_LP

            # --- De-smooth eigenvalues to compensate for recursive averaging ---
            if l > 0:
                lambda_LP_old = lambda_LP_STFT[k, l - 1, :]
                alpha = 1.0 / (1.0 - forgetPSD)
                beta  = forgetPSD / (1.0 - forgetPSD)
                lam = alpha * lambda_LP - beta * lambda_LP_old
            else:
                lam = lambda_LP

            # Floor at lambdaMin
            lam = np.maximum(lam, lambdaMin)

            # Save
            lambda_STFT[k, l, :] = lam

    return P_STFT, lambda_STFT, lambda_LP_STFT
