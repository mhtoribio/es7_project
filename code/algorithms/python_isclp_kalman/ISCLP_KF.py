import numpy as np

def ISCLP(A, Psi_w_delta, y_stack, psi_s_stack, H_stack, Psi_w_tilde_init, beta):
    """
    Runs the ISCLP Kalman filter in one frequency bin.

    Parameters
    ----------
    A : np.ndarray
        State transition matrix - stateDim x stateDim
    Psi_w_delta : np.ndarray
        Process noise correlation matrix - stateDim x stateDim
    y_stack : np.ndarray
        Microphone signal - channels x frames
    psi_s_stack : np.ndarray
        Target signal PSD - frames
    H_stack : np.ndarray
        RETF - channels (x sources) x frames (if more than one source)
    Psi_w_tilde_init : np.ndarray
        Initial state estimation error correlation matrix - stateDim x stateDim
    beta : float
        Gain decay limitation

    Returns
    -------
    q_stack, e_prio_stack, e_post_stack, e_post_smooth_stack : np.ndarray
        Outputs - all arrays with shape (frames,)
    """

    # get dimensions
    if H_stack.ndim == 2:
        N = 1
        M, numFrames = H_stack.shape
    else:
        M, N, numFrames = H_stack.shape
    stateDim = Psi_w_tilde_init.shape[0]

    # initKF
    #
    w_hat = np.zeros((stateDim, 1), dtype=complex)           # state estimate
    Psi_w_tile = Psi_w_tilde_init.copy()                     # state estimation error correlation
    y_old = np.zeros((M, 1), dtype=complex)                 # previous microphone signal
    u_LP = np.zeros((stateDim - M + 1, 1), dtype=complex)   # long-term prediction part
    smooth_gain = 1.0                                        # initialize smoothing gain

    # init output
    q_stack = np.zeros(numFrames, dtype=complex)
    e_prio_stack = np.zeros(numFrames, dtype=complex)
    e_post_stack = np.zeros(numFrames, dtype=complex)
    e_post_smooth_stack = np.zeros(numFrames, dtype=complex)

    for i_frame in range(numFrames):

        #%% Load Data
        y = y_stack[:, i_frame:i_frame+1]  # keep as column vector
        psi_s = psi_s_stack[i_frame]
        if N == 1:
            H = H_stack[:, i_frame:i_frame+1]
        else:
            H = H_stack[:, :, i_frame]

        #%% Spatio-Temporal Pre-Processing
        # compute g, B
        g = np.sum(H @ np.linalg.pinv(H.T @ H), axis=1, keepdims=True)
        Btmp = np.eye(M) - H @ np.linalg.pinv(H.T @ H) @ H.T
        B = Btmp[:, :M-N]

        # update q, u
        q = g.T @ y
        u_SC = B.T @ y
        u_LP = np.vstack([y_old, u_LP[:-M, :]])
        u = np.vstack([u_SC, u_LP])

        #%% Kalman Filter
        # time update: state estimate
        w_hat = A @ w_hat
        # time update: state estimation error correlation matrix
        Psi_w_tile = A.T @ Psi_w_tile @ A + Psi_w_delta
        # symmetrize (just in case, avoiding accuracy issues)
        Psi_w_tile = 0.5 * (Psi_w_tile + Psi_w_tile.T)
        # error
        e_prio_conj = np.conj(q) - (u.T @ w_hat)
        # error PSD
        psi_e = np.real(u.T @ Psi_w_tile @ u) + psi_s + np.finfo(float).eps
        # Kalman gain
        k = Psi_w_tile @ u / psi_e
        # measurement update: state estimate
        w_hat = w_hat + k * e_prio_conj
        # measurement update: state estimation error correlation matrix
        Psi_w_tile = Psi_w_tile - k @ (u.T @ Psi_w_tile)

        #%% Spectral Post Processing
        # compute gains
        gain = psi_s / psi_e
        smooth_gain = max(gain, beta * smooth_gain)
        # apply gains
        e_post_conj = gain * e_prio_conj
        e_post_smooth_conj = smooth_gain * e_prio_conj

        #%% Save Data
        q_stack[i_frame] = q
        e_prio_stack[i_frame] = np.conj(e_prio_conj)
        e_post_stack[i_frame] = np.conj(e_post_conj)
        e_post_smooth_stack[i_frame] = np.conj(e_post_smooth_conj)
        # save previous mic. signal
        y_old = y

    return q_stack, e_prio_stack, e_post_stack, e_post_smooth_stack
