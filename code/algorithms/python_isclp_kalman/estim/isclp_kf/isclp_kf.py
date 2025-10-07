import numpy as np

def ISCLP(
    A: np.ndarray,                 # (stateDim, stateDim)
    Psi_w_delta: np.ndarray,       # (stateDim, stateDim)
    y_stack: np.ndarray,           # (M, T)
    psi_s_stack: np.ndarray,       # (T,)
    H_stack: np.ndarray,           # (M, T) or (M, N, T)
    Psi_w_tilde_init: np.ndarray,  # (stateDim, stateDim)
    beta: float,                   # scalar in (0,1], gain-decay limit
):
    """
    Runs the ISCLP Kalman filter in one frequency bin.

    Returns
    -------
    q_stack : (T,)
    e_prio_stack : (T,)
    e_post_stack : (T,)
    e_post_smooth_stack : (T,)
    """
    # --- dimensions
    if H_stack.ndim == 2:
        # single source path
        M, T = H_stack.shape
        N = 1
    else:
        M, N, T = H_stack.shape

    stateDim = Psi_w_tilde_init.shape[0]

    A = A * np.identity(stateDim)

    # --- dtype / eps
    cdtype = np.result_type(
        A.dtype, Psi_w_delta.dtype, y_stack.dtype, H_stack.dtype, np.complex64
    )
    rdtype = np.result_type(psi_s_stack.dtype, np.float32)
    eps = np.finfo(np.float64).eps

    # --- init KF state
    w_hat = np.zeros((stateDim,), dtype=cdtype)
    Psi_w_tilde = Psi_w_tilde_init.astype(cdtype, copy=True)

    y_old = np.zeros((M,), dtype=cdtype)

    # length of LP regressor part (matches MATLAB: stateDim - M + 1)
    Llp = int(stateDim - M + 1)
    if Llp < 0:
        raise ValueError("stateDim - M + 1 must be >= 0 (check your Psi_w_tilde_init size).")
    u_LP = np.zeros((Llp,), dtype=cdtype)

    smooth_gain = 1.0

    # --- outputs
    q_stack = np.zeros((T,), dtype=cdtype)
    e_prio_stack = np.zeros((T,), dtype=cdtype)
    e_post_stack = np.zeros((T,), dtype=cdtype)
    e_post_smooth_stack = np.zeros((T,), dtype=cdtype)

    I_M = np.eye(M, dtype=cdtype)

    for t in range(T):
        # -- load data
        y = y_stack[:, t].astype(cdtype, copy=False)
        psi_s = float(np.asarray(psi_s_stack[t], dtype=rdtype))

        if H_stack.ndim == 2:
            H = H_stack[:, t].reshape(M, 1).astype(cdtype, copy=False)  # (M,1)
        else:
            H = H_stack[:, :, t].astype(cdtype, copy=False)             # (M,N)

        # -- Spatio-Temporal Pre-Processing
        # g = sum(H * inv(H^H H), axis=1)
        HH = H.conj().T @ H                         # (N,N)
        # robust inverse
        try:
            inv_HH = np.linalg.inv(HH)
        except np.linalg.LinAlgError:
            inv_HH = np.linalg.pinv(HH)
        G = H @ inv_HH                               # (M,N)
        g = np.sum(G, axis=1)                        # (M,)

        # B = [I - H * inv(H^H H) * H^H](:, 1:M-N)
        P = H @ inv_HH @ H.conj().T                 # (M,M)
        Btmp = I_M - P
        MN = max(M - N, 0)
        B = Btmp[:, :MN]                            # (M, M-N)

        # q and u
        # q = g' y   (i.e., conj(g)^T y)
        q = np.vdot(g, y)
        u_SC = B.conj().T @ y                       # (M-N,)
        # LP regressor shift: [y_old; u_LP(1:end-M)]
        if Llp >= M:
            u_LP = np.concatenate((y_old, u_LP[: Llp - M])) if Llp > 0 else u_LP
        elif Llp > 0:
            # If Llp < M, we only take the first Llp entries of y_old
            u_LP = y_old[:Llp]
        # total regressor
        u = np.concatenate((u_SC, u_LP)).astype(cdtype, copy=False)  # (stateDim,)

        # -- Kalman Filter
        # time update
        w_hat = A @ w_hat
        Psi_w_tilde = A.conj().T @ Psi_w_tilde @ A + Psi_w_delta
        # enforce Hermitian symmetry
        Psi_w_tilde = 0.5 * (Psi_w_tilde + Psi_w_tilde.conj().T)

        # prior error (conjugate form as in MATLAB)
        e_prio_conj = np.conj(q) - np.vdot(u, w_hat)  # scalar

        # error PSD
        psi_e = float(np.real(u.conj().T @ (Psi_w_tilde @ u)) + psi_s + eps)

        # gain
        k = (Psi_w_tilde @ u) / psi_e                 # (stateDim,)

        # measurement update
        w_hat = w_hat + k * e_prio_conj
        Psi_w_tilde = Psi_w_tilde - np.outer(k, (u.conj().T @ Psi_w_tilde))

        # -- Spectral Post-Processing
        gain = psi_s / psi_e
        smooth_gain = max(gain, beta * smooth_gain)

        e_post_conj        = gain        * e_prio_conj
        e_post_smooth_conj = smooth_gain * e_prio_conj

        # -- Save (note MATLAB stores the conjugate of *_conj)
        q_stack[t]              = q
        e_prio_stack[t]         = np.conj(e_prio_conj)
        e_post_stack[t]         = np.conj(e_post_conj)
        e_post_smooth_stack[t]  = np.conj(e_post_smooth_conj)

        # update y_old
        y_old = y

    return q_stack, e_prio_stack, e_post_stack, e_post_smooth_stack
