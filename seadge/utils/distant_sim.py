# utils/distant_sim.py

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.signal import oaconvolve, fftconvolve

from seadge import config
from seadge.utils.log import log
from seadge.utils.cache import load_rir_mem_or_die, make_rir_cache_key

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _xfade_windows(L: int) -> Tuple[np.ndarray, np.ndarray]:
    """Half-cosine crossfade windows that sum to 1."""
    if L <= 0:
        return np.array([]), np.array([])
    t = np.linspace(0, np.pi, L, endpoint=False)
    w_up   = 0.5 - 0.5*np.cos(t)
    w_down = 1.0 - w_up
    return w_down, w_up

def _pre_window(pos_len: int, xfade_len: int, first: bool = False, last: bool = False) -> np.ndarray:
    """
    Window for clean speech at position i.
    Crossfaded in [tstart_i, tstart_i+xfade_len] and [tstart_next, tstart_next+xfade_len]
    """
    if xfade_len <= 0:
        return np.ones(pos_len)

    win_len = pos_len+xfade_len if not last else pos_len
    win = np.ones(win_len)
    w_down, w_up = _xfade_windows(xfade_len)
    if not first:
        win[0: xfade_len] = w_up
    if not last:
        win[pos_len: pos_len+xfade_len] = w_down
    return win

def _convolve_seg(x_seg: np.ndarray, H_RM: np.ndarray, method: str = "oaconv") -> np.ndarray:
    """
    x_seg: (Ns,)
    H_RM:  (R, M)
    return: (Ns+R-1, M)
    """
    assert x_seg.ndim == 1
    x_seg = np.asarray(x_seg, float)
    H_RM  = np.asarray(H_RM,  float)
    if method == "oaconv":
        return oaconvolve(x_seg[:, None], H_RM, mode="full", axes=(0,))
    else:
        return fftconvolve(x_seg[:, None], H_RM, mode="full", axes=(0,))

def _normalize_rir(H: np.ndarray, mode: Optional[str]) -> np.ndarray:
    """
    Normalize RIR stack (R, M).
    mode: None | "direct" | "energy"
    """
    if mode is None:
        return H
    if H.size == 0:
        return H
    if mode == "direct":
        nz = int(np.argmax(np.any(H != 0.0, axis=1)))
        g = float(np.mean(np.abs(H[nz, :])) + 1e-12)
        return H / g
    if mode == "energy":
        g = float(np.sqrt(np.sum(H**2)) / np.sqrt(max(1, H.shape[1])) + 1e-12)
        return H / g
    return H

def _load_schedule(
    src: config.SourceSpec,
    *,
    fs: int,
    room_cfg,
    cache_root: Path,
    normalize: Optional[str],
) -> List[Tuple[int, np.ndarray]]:
    """
    Load [(start_sample, H(R,M)), ...] from cache. No recompute; raises on miss.
    Assumes src.location_history exists (your config).
    """
    schedule: List[Tuple[int, np.ndarray]] = []
    for loc in src.location_history:
        key = make_rir_cache_key(room_cfg, fs, loc)
        H = load_rir_mem_or_die(key, str(cache_root))  # raises if missing
        H = _normalize_rir(H, normalize)
        schedule.append((int(loc.start_sample), H))
    schedule.sort(key=lambda t: t[0])
    return schedule

# ----------------------------------------------------------------------
# Instantaneous RIR at output time n (ground truth for visualization)
# ----------------------------------------------------------------------

def _eff_xfade_len_for_boundary(
    s0_prev: int, y_prev_len: int,
    s0_next: int, y_next_len: int,
    b: int, Lxf: int,
) -> int:
    """Effective crossfade length, exactly like sim_distant_src's min(...)."""
    off_prev   = b - s0_prev
    avail_prev = max(0, y_prev_len - off_prev)  # prev left after boundary
    avail_next = y_next_len                     # next from start
    return max(0, min(Lxf, avail_prev, avail_next))


def _instantaneous_rir_at_time(
    schedule: list[tuple[int, np.ndarray]],  # [(start, H(R,M))], sorted
    n: int,
    *,
    starts: list[int],
    ends: list[int],          # input end (exclusive) per segment
    Lx: list[int],            # input slice length per segment
    Rks: list[int],           # RIR length per segment
    y_lens: list[int],        # per-segment output length = Lx + Rk - 1
    Lxf: int,                 # xfade length in samples
    Rmax: int,
    N: int,                   # clean length
    dtype=np.float64,
) -> np.ndarray:
    """
    Exact h_n (Rmax, M) matching the fast renderer:
      - outside fades: sum all active segments' windowed taps,
      - inside [b, b+L_eff): convex mix of the two adjacent segments' windowed taps only.
    """
    if not schedule:
        return np.zeros((0, 0), dtype=dtype)

    K = len(schedule)
    M = schedule[0][1].shape[1]
    h = np.zeros((Rmax, M), dtype=dtype)

    # --------- check if inside an *effective* forward fade ----------
    if K >= 2 and Lxf > 0:
        boundaries = starts[1:]
        j = int(np.searchsorted(boundaries, n, side="right") - 1)  # last boundary <= n
        if 0 <= j < (K - 1):
            b = boundaries[j]
            u = n - b
            if u >= 0:
                L_eff = _eff_xfade_len_for_boundary(
                    s0_prev=starts[j],  y_prev_len=y_lens[j],
                    s0_next=starts[j+1], y_next_len=y_lens[j+1],
                    b=b, Lxf=Lxf,
                )
                if 0 <= u < L_eff:
                    # two-segment convex mix (overwrite semantics)
                    w_down, w_up = _xfade_windows(L_eff)

                    # prev segment j
                    H_prev, R_prev = schedule[j][1], Rks[j]
                    idx_prev = n - starts[j]  # output index for seg j at time n
                    if 0 <= idx_prev <= (Lx[j] + R_prev - 2):
                        r_low  = max(0, idx_prev - (Lx[j] - 1))
                        r_high = min(R_prev - 1, idx_prev)
                        if r_high >= r_low:
                            end = min(Rmax, r_high + 1)
                            h[r_low:end, :] += w_down[u] * H_prev[r_low:end, :]

                    # next segment j+1
                    H_next, R_next = schedule[j+1][1], Rks[j+1]
                    idx_next = n - starts[j+1]  # equals u
                    if 0 <= idx_next <= (Lx[j+1] + R_next - 2):
                        r_low  = max(0, idx_next - (Lx[j+1] - 1))
                        r_high = min(R_next - 1, idx_next)
                        if r_high >= r_low:
                            end = min(Rmax, r_high + 1)
                            h[r_low:end, :] += w_up[u] * H_next[r_low:end, :]

                    return h
                # else: nominally within Lxf but beyond L_eff → fall through to sum

    # --------- not in effective fade: sum all active segments ----------
    for k in range(K):
        s0k, Hk = starts[k], schedule[k][1]
        Rk = Rks[k]
        # active if n within this segment's output span
        if not (s0k <= n <= (s0k + Lx[k] + Rk - 2)):
            continue
        idx = n - s0k
        r_low  = max(0, idx - (Lx[k] - 1))
        r_high = min(Rk - 1, idx)
        if r_high >= r_low:
            end = min(Rmax, r_high + 1)
            h[r_low:end, :] += Hk[r_low:end, :]

    return h


def convolve_time_varying_exact_via_hn(
    clean: np.ndarray,
    src,
    *,
    fs: int,
    room_cfg,
    cache_root: Path,
    xfade_ms: float,
    normalize: str | None = "direct",
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Ground-truth renderer using the *instantaneous* RIR h_n you animate:
      y[n,:] = sum_r h_n[r,:] * x[n-r]
    h_n is built to be mathematically identical to sim_distant_src.
    """
    x = np.asarray(clean, float)
    N = x.shape[0]

    schedule = _load_schedule(src, fs=fs, room_cfg=room_cfg, cache_root=cache_root, normalize=normalize)
    if not schedule:
        return np.zeros((N, len(room_cfg.mic_pos)), dtype=dtype)

    # Precompute geometry
    K       = len(schedule)
    starts  = [int(s) for s, _ in schedule]
    Rks     = [H.shape[0] for _, H in schedule]
    Rmax    = max(Rks)
    M       = schedule[0][1].shape[1]
    ends    = [starts[k+1] if k+1 < K else N for k in range(K)]     # input end (exclusive)
    Lx      = [max(0, ends[k] - starts[k]) for k in range(K)]       # input slice length
    y_lens  = [Lx[k] + Rks[k] - 1 for k in range(K)]
    Lxf     = int(round(xfade_ms * 1e-3 * fs))

    y = np.zeros((N + Rmax - 1, M), dtype=dtype)

    for n in range(y.shape[0]):
                # --- build instantaneous kernel (Rmax, M) ---
        h_n = _instantaneous_rir_at_time(
            schedule, n,
            starts=starts, ends=ends, Lx=Lx, Rks=Rks, y_lens=y_lens,
            Lxf=Lxf, Rmax=Rmax, N=N, dtype=dtype,
        )

        # number of taps that can contribute to y[n]
        r_need = min(Rmax, n + 1, N)

        # --- gather input slice, pad both sides if needed, and reverse ---
        x_start = n - r_need + 1
        x_end   = n + 1
        # clip to [0, N]
        left_clip  = max(0, -x_start)
        right_clip = max(0, x_end - N)

        xi_core = x[max(0, x_start) : min(x_end, N)]
        if left_clip or right_clip:
            xi = np.pad(xi_core, (left_clip, right_clip), mode="constant")
        else:
            xi = xi_core
        # ensure exact length r_need
        if xi.shape[0] != r_need:
            if xi.shape[0] < r_need:
                xi = np.pad(xi, (0, r_need - xi.shape[0]), mode="constant")
            else:
                xi = xi[:r_need]
        xi = xi[::-1]  # (r_need,)

        # --- make kernel rows match xi length exactly ---
        k = xi.shape[0]  # should equal r_need
        if h_n.shape[0] < k:
            Huse = np.pad(h_n, ((0, k - h_n.shape[0]), (0, 0)), mode="constant")
        else:
            Huse = h_n[:k, :]

        # (k,) @ (k, M) -> (M,)
        y[n, :] = xi @ Huse

    return y

# ----------------------------------------------------------------------
# FAST renderer: convolve-each-segment, then crossfade outputs (production)
# ----------------------------------------------------------------------

def sim_distant_src(
    clean: np.ndarray,
    src: config.SourceSpec,
    *,
    fs: int,
    room_cfg,             # RoomCfg (mic_pos already expanded)
    cache_root: Path,
    xfade_ms: float = 64.0,
    method: str = "oaconv",           # "oaconv" or "fft"
    normalize: Optional[str] = "direct",  # None|"direct"|"energy"
) -> np.ndarray:
    """
    Convolve a mono clean source with a time-varying RIR sequence (one per pose),
    crossfading at pose boundaries. Returns (N_out, M).

    NOTE: WILL NOT recompute RIRs. If any RIR is missing from cache,
    load_rir_mem_or_die(...) is expected to raise.
    """
    clean = np.asarray(clean, float)
    N = clean.shape[0]
    xfade_len = max(0, int(round(xfade_ms * 1e-3 * fs)))

    # 1) schedule: (start_sample, H(R,M))
    schedule: List[Tuple[int, np.ndarray]] = _load_schedule(
        src, fs=fs, room_cfg=room_cfg, cache_root=cache_root, normalize=normalize
    )

    if not schedule:
        return np.zeros((N, len(room_cfg.mic_pos)), float)

    # 2) Precompute segment convolutions
    M = schedule[0][1].shape[1]
    Rmax = max(H.shape[0] for _, H in schedule)
    y = np.zeros((N + Rmax - 1, M), float)

    seg_out: List[Tuple[int, np.ndarray]] = []  # (s0, yk)
    for idx, (s0, H) in enumerate(schedule):
        last = idx+1 < len(schedule)
        s1 = schedule[idx+1][0] if last else N
        s0 = max(0, min(s0, N))
        s1 = max(s0, min(s1, N))
        win = _pre_window(s1-s0, xfade_len, first = idx==0, last=last)
        x_clean_seg = clean[s0:s1+xfade_len]
        x_seg = x_clean_seg * win if (idx+1 < len(schedule)) else x_clean_seg * win[0: s1-s0]
        yk = _convolve_seg(x_seg, H, method=method)  # (len+R-1, M)

        # mhtdebug
        # import matplotlib.pyplot as plt
        # plt.plot(win)
        # plt.title("win")
        # plt.savefig("/home/markus/shit/win.png")
        # plt.close()
        # plt.plot(x_clean_seg)
        # plt.title("clean_seg")
        # plt.savefig("/home/markus/shit/clean_seg.png")
        # plt.close()
        # plt.plot(x_seg)
        # plt.title("x_seg")
        # plt.savefig("/home/markus/shit/seg.png")
        # plt.close()
        # plt.plot(yk)
        # plt.title("yk")
        # plt.savefig("/home/markus/shit/yk.png")
        # plt.close()
        # exit()

        seg_out.append((s0, yk))

    # 3) Overlap-add all segments
    for s0, yk in seg_out:
        y[s0:s0+yk.shape[0], :] += yk

    return y


# ----------------------------------------------------------------------
# Animations using instantaneous RIR per framj
# ----------------------------------------------------------------------

from pathlib import Path
from tqdm import tqdm

def _save_animation_with_progress(ani, writer, outpath, total_frames: int):
    """Save a Matplotlib animation with a tqdm progress bar."""
    from pathlib import Path as _Path
    with tqdm(total=total_frames, desc=f"Writing { _Path(outpath).name }") as pbar:
        last = -1
        def _cb(i, n):
            nonlocal last
            if i > last:
                pbar.update(i - last)
                last = i
        ani.save(outpath, writer=writer, progress_callback=_cb)

def _make_schedule_geometry_for_anim(src, *, fs, room_cfg, cache_root, xfade_ms, normalize):
    # Load cached RIRs
    schedule = _load_schedule(src, fs=fs, room_cfg=room_cfg,
                              cache_root=cache_root, normalize=normalize)
    if not schedule:
        raise ValueError("No RIRs found in cache for this source.")

    K       = len(schedule)
    starts  = [int(s) for s, _ in schedule]
    Rks     = [H.shape[0] for _, H in schedule]
    Rmax    = max(Rks)
    M       = schedule[0][1].shape[1]
    Lxf     = int(round(xfade_ms * 1e-3 * fs))

    return schedule, starts, Rks, Rmax, M, Lxf

def animate_rir_time(
    src,                               # config.SourceSpec
    *,
    fs: int,
    room_cfg,
    cache_root: Path,
    xfade_ms: float,
    outpath: Path,
    N: int,                            # timeline length to animate (samples)
    fps: int = 30,
    duration_s: float | None = None,
    early_ms: float | None = None,     # show only first X ms of the RIR
    normalize: str | None = "direct",  # must match renderer
    mics_overlay: bool = True,
    share_ylim: bool = True,
):
    """
    Animate time-domain instantaneous RIR h_n produced by _instantaneous_rir_at_time.
    Saves MP4 if ffmpeg is available and outpath ends with .mp4, else GIF.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from tqdm.auto import tqdm

    schedule, starts, Rks, Rmax, M, Lxf = _make_schedule_geometry_for_anim(
        src, fs=fs, room_cfg=room_cfg, cache_root=cache_root,
        xfade_ms=xfade_ms, normalize=normalize
    )
    ends   = [starts[k+1] if k+1 < len(starts) else N for k in range(len(starts))]
    Lx     = [max(0, ends[k] - starts[k]) for k in range(len(starts))]
    y_lens = [Lx[k] + Rks[k] - 1 for k in range(len(starts))]

    # Frame times
    step = max(1, int(round(fs / fps)))
    t_idx = np.arange(0, N, step, dtype=int)
    if duration_s is not None:
        max_frames = int(np.floor(duration_s * fps))
        t_idx = t_idx[:max_frames]
    Nt = t_idx.size
    if Nt == 0:
        raise ValueError("No frames to animate (Nt=0). Increase duration or fps.")

    # Plot axis (lag)
    if early_ms is not None:
        Rplot = min(Rmax, int(round(early_ms * 1e-3 * fs)))
    else:
        Rplot = Rmax
    tau_ms = (np.arange(Rplot) / fs) * 1000.0

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # First pass: determine y-limits (optional but nice)
    ymax = 0.0
    for n in tqdm(t_idx, desc="Scanning RIR range for y-limits"):
        h_n = _instantaneous_rir_at_time(
            schedule, int(n),
            starts=starts, ends=ends, Lx=Lx, Rks=Rks, y_lens=y_lens,
            Lxf=Lxf, Rmax=Rmax, N=N, dtype=np.float64
        )
        ymax = max(ymax, float(np.max(np.abs(h_n[:Rplot]))))
    if ymax == 0.0:
        ymax = 1.0

    # Figure
    fig, ax = plt.subplots(figsize=(9, 4))
    lines = []
    if mics_overlay:
        for m in range(M):
            (ln,) = ax.plot(tau_ms, np.zeros_like(tau_ms), lw=0.9, label=f"mic {m}")
            lines.append(ln)
        if M <= 12:
            ax.legend(loc="upper right", fontsize=8)
    else:
        for m in range(M):
            (ln,) = ax.plot(tau_ms, np.zeros_like(tau_ms), lw=0.9)
            lines.append(ln)

    ax.set_xlabel("Lag τ [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, lw=0.3, alpha=0.6)
    if share_ylim:
        ax.set_ylim(-1.05 * ymax, 1.05 * ymax)

    def _title(fi: int) -> str:
        t_sec = t_idx[fi] / fs
        return f"Instantaneous RIR (hₙ) — t = {t_sec:6.3f} s"

    ax.set_title(_title(0))

    def update(fi: int):
        n = int(t_idx[fi])
        h_n = _instantaneous_rir_at_time(
            schedule, n,
            starts=starts, ends=ends, Lx=Lx, Rks=Rks, y_lens=y_lens,
            Lxf=Lxf, Rmax=Rmax, N=N, dtype=np.float64
        )
        Hs = h_n[:Rplot, :]  # (Rplot, M)
        for m in range(M):
            lines[m].set_ydata(Hs[:, m])
        ax.set_title(_title(fi))
        return lines

    # Save animation
    ani = FuncAnimation(fig, update, frames=Nt, blit=False)
    try:
        if outpath.suffix.lower() == ".mp4":
            writer = FFMpegWriter(fps=fps, bitrate=2400)
            _save_animation_with_progress(ani, writer, outpath, Nt)
        else:
            raise RuntimeError("force GIF")
    except Exception:
        from matplotlib.animation import PillowWriter
        if outpath.suffix.lower() != ".gif":
            outpath = outpath.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        _save_animation_with_progress(ani, writer, outpath, Nt)

    plt.close(fig)
    return outpath


def animate_freqresp(
    src,                               # config.SourceSpec
    *,
    fs: int,
    room_cfg,
    cache_root: Path,
    xfade_ms: float,
    outpath: Path,
    N: int,                            # timeline length to animate (samples)
    fps: int = 30,
    duration_s: float | None = None,
    mic: int = 0,
    db: bool = True,
    nfft: int | None = None,
    normalize: str | None = "direct",
):
    """
    Animate |FFT{h_n}|\n for the chosen mic, using instantaneous kernels per frame.
    Saves MP4 if ffmpeg is available and outpath ends with .mp4, else GIF.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from tqdm.auto import tqdm

    schedule, starts, Rks, Rmax, M, Lxf = _make_schedule_geometry_for_anim(
        src, fs=fs, room_cfg=room_cfg, cache_root=cache_root,
        xfade_ms=xfade_ms, normalize=normalize
    )
    ends   = [starts[k+1] if k+1 < len(starts) else N for k in range(len(starts))]
    Lx     = [max(0, ends[k] - starts[k]) for k in range(len(starts))]
    y_lens = [Lx[k] + Rks[k] - 1 for k in range(len(starts))]

    if not (0 <= mic < M):
        raise IndexError(f"mic {mic} out of range [0,{M-1}]")

    # Frame times
    step = max(1, int(round(fs / fps)))
    t_idx = np.arange(0, N, step, dtype=int)
    if duration_s is not None:
        max_frames = int(np.floor(duration_s * fps))
        t_idx = t_idx[:max_frames]
    Nt = t_idx.size
    if Nt == 0:
        raise ValueError("No frames to animate (Nt=0). Increase duration or fps.")

    # FFT grid
    if nfft is None:
        # next power of two >= Rmax
        nfft = 1 << max(0, int(np.ceil(np.log2(max(1, Rmax)))))
    F = nfft // 2 + 1
    f = np.fft.rfftfreq(nfft, 1.0 / fs)

    # First pass: y-limits
    y_min, y_max = +np.inf, -np.inf
    for n in tqdm(t_idx, desc="Scanning FR range for y-limits"):
        h_n = _instantaneous_rir_at_time(
            schedule, int(n),
            starts=starts, ends=ends, Lx=Lx, Rks=Rks, y_lens=y_lens,
            Lxf=Lxf, Rmax=Rmax, N=N, dtype=np.float64
        )
        H = np.fft.rfft(h_n[:, mic], n=nfft, axis=0)  # (F,)
        mag = np.abs(H)
        if db:
            mag = 20 * np.log10(mag + 1e-12)
        y_min = min(y_min, float(np.min(mag)))
        y_max = max(y_max, float(np.max(mag)))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -60.0, 0.0
    pad = 0.05 * (y_max - y_min + 1e-9)

    # Figure
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    (line,) = ax.plot(f, np.zeros_like(f), lw=0.9)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]" if db else "Magnitude")
    ax.grid(True, lw=0.3, alpha=0.6)
    ax.set_ylim(y_min - pad, y_max + pad)

    def _title(fi: int) -> str:
        t_sec = t_idx[fi] / fs
        return f"|H(f, t)| (mic {mic}) — t = {t_sec:6.3f} s"

    ax.set_title(_title(0))

    def update(fi: int):
        n = int(t_idx[fi])
        h_n = _instantaneous_rir_at_time(
            schedule, n,
            starts=starts, ends=ends, Lx=Lx, Rks=Rks, y_lens=y_lens,
            Lxf=Lxf, Rmax=Rmax, N=N, dtype=np.float64
        )
        H = np.fft.rfft(h_n[:, mic], n=nfft, axis=0)
        mag = np.abs(H)
        if db:
            mag = 20 * np.log10(mag + 1e-12)
        line.set_ydata(mag)
        ax.set_title(_title(fi))
        return (line,)

    # Save animation
    ani = FuncAnimation(fig, update, frames=Nt, blit=False)
    try:
        if outpath.suffix.lower() == ".mp4":
            writer = FFMpegWriter(fps=fps, bitrate=2400)
            _save_animation_with_progress(ani, writer, outpath, Nt)
        else:
            raise RuntimeError("force GIF")
    except Exception:
        from matplotlib.animation import PillowWriter
        if outpath.suffix.lower() != ".gif":
            outpath = outpath.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        _save_animation_with_progress(ani, writer, outpath, Nt)

    plt.close(fig)
    return outpath
