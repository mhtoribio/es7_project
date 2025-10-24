################
# Visualization
################
import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt
from seadge import config
from typing import Optional
from pathlib import Path
 
# Spectrogram
def spectrogram(
    spec,
    filename,                           # output filepath
    scale="mag",                        # 'mag' | 'pow' | 'lin'
    x_tick_prop=None,                   # (start, res, dist) or None
    y_tick_prop=None,                   # (start, res, dist) or None
    c_range=None,                       # (vmin, vmax) or None
    plot_colorbar=True,
    dpi=150,
    title=None
):
    """
    Save a spectrogram-like image of `spec` to `filename`.

    Parameters
    ----------
    spec : 2D array (freqbins x frames)
    scale : {'mag','pow','lin'}
        'mag' -> 20*log10(|spec|)
        'pow' -> 10*log10(|spec|)
        'lin' -> spec (no conversion)
    x_tick_prop : tuple or None
        (x_start, x_res, x_dist). Labels = x_start + ticks*x_res.
    y_tick_prop : tuple or None
        (y_start, y_res, y_dist). Labels = y_start + ticks*y_res.
    c_range : tuple or None
        (vmin, vmax) for color scaling.
    plot_colorbar : bool
    filename : str
        Path to save the image.
    dpi : int
        Save resolution.
    """

    cfg = config.get() # Get configs from config.py
    spec = np.asarray(spec)

    # MATLAB-ish helpers
    tiny = np.finfo(float).tiny
    def mag2db(x): return 20.0 * np.log10(np.maximum(x, tiny))
    def pow2db(x): return 10.0 * np.log10(np.maximum(x, tiny))

    # Scaling
    if scale == "mag":
        img = mag2db(np.abs(spec))
    elif scale == "pow":
        img = pow2db(np.abs(spec))
    elif scale == "lin":
        img = spec
    else:
        raise ValueError("undefined scaling option. Use 'mag', 'pow', or 'lin'.")

    # Plot
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    im_kwargs = dict(origin="lower", aspect="auto")
    if c_range is not None:
        im_kwargs.update(vmin=c_range[0], vmax=c_range[1])
    else:
        im_kwargs.update(vmin=cfg.dsp.c_range[0], vmax=cfg.dsp.c_range[1])
    im = ax.imshow(img, **im_kwargs)

    # Match MATLAB: no tick marks, y increasing upwards
    ax.tick_params(length=0)

    # Colorbar
    if plot_colorbar:
        fig.colorbar(im, ax=ax)

    # X ticks
    if x_tick_prop is not None:
        x_start, x_res, x_dist = x_tick_prop
        x_ticks = np.arange(0, spec.shape[1], int(x_dist))
        x_labels = x_start + x_ticks * x_res  # Python is 0-based (MATLAB had (ticks-1)*res)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{v}" for v in np.atleast_1d(x_labels)])
    else:
        x_start, x_res, x_dist = cfg.dsp.x_tick_prop
        x_ticks = np.arange(0, spec.shape[1], int(x_dist))
        x_labels = x_start + x_ticks * x_res  # Python is 0-based (MATLAB had (ticks-1)*res)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{v}" for v in np.atleast_1d(x_labels)])
        
    # Y ticks
    if y_tick_prop is not None:
        y_start, y_res, y_dist = y_tick_prop
        y_ticks = np.arange(0, spec.shape[0], int(y_dist))
        y_labels = y_start + y_ticks * y_res
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{v}" for v in np.atleast_1d(y_labels)])
    else:
        y_start, y_res, y_dist = cfg.dsp.y_tick_prop
        y_ticks = np.arange(0, spec.shape[0], int(y_dist))
        y_labels = y_start + y_ticks * y_res
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{v}" for v in np.atleast_1d(y_labels)])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [kHz]') 

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_rir(
    rir: np.ndarray,
    fs: int,
    outputfile: Path,
    channel: int | None = None,
    early_ms: float | None = None,
    dpi=150,
    title: str | None = None,
) -> None:
    """
    Plot an impulse response array and write the figure to disk.

    Parameters
    ----------
    rir : np.ndarray
        Array of shape (R, M) where R is RIR length and M is #mics.
    fs : int
        Sample rate [Hz].
    outputfile : pathlib.Path
        Destination path (png/pdf/svg etc. based on suffix).
    channel : int | None
        If set, plot only this channel (0-based). Otherwise plot all channels.
    early_ms : float | None
        If set (e.g., 80.0), only plot the first early_ms milliseconds.
    dpi : int
        Save resolution.
    title : str | None
        Optional plot title.
    """
    rir = np.asarray(rir)
    if rir.ndim != 2:
        raise ValueError(f"rir must be 2D (R, M), got shape {rir.shape}")

    R, M = rir.shape

    # Slice early window if requested
    if early_ms is not None:
        L_e = max(1, int(round(early_ms * 1e-3 * fs)))
        L_e = min(L_e, R)
    else:
        L_e = R

    # Choose channels
    if channel is not None:
        if not (0 <= channel < M):
            raise IndexError(f"channel {channel} out of range [0, {M-1}]")
        data = rir[:L_e, channel][:, None]   # (L_e, 1)
        labels = [f"ch {channel}"]
    else:
        data = rir[:L_e, :]                  # (L_e, M)
        labels = [f"ch {m}" for m in range(M)]

    # Time axis in ms
    t_ms = (np.arange(L_e) / fs) * 1000.0

    # Make sure output dir exists
    outputfile = Path(outputfile)
    outputfile.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    if data.shape[1] == 1:
        ax.plot(t_ms, data[:, 0], linewidth=1.0)
    else:
        # Overlay all channels
        for m in range(data.shape[1]):
            ax.plot(t_ms, data[:, m], linewidth=0.9, alpha=0.9, label=labels[m])
        if data.shape[1] <= 12:
            ax.legend(loc="upper right", ncols=2, fontsize=8)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    if title is None:
        ttl = "Impulse response"
        if early_ms is not None:
            ttl += f" (first {early_ms:g} ms)"
        ax.set_title(ttl)
    else:
        ax.set_title(title)

    # If early_ms was given, limit x-axis accordingly (cosmetic)
    if early_ms is not None:
        ax.set_xlim(0, early_ms)

    fig.tight_layout()
    fig.savefig(outputfile, dpi=dpi)
    plt.close(fig)

def _arrow_from_pose(x, y, azimuth_deg, scale=0.3):
    from math import radians, sin, cos
    th = radians(azimuth_deg)
    return x, y, scale*cos(th), scale*sin(th)

def plot_room_topdown(room_cfg, outputfile: Path, *,
                      show_dirs: bool = True,
                      annotate: bool = True,
                      dpi: int = 150) -> None:
    """
    Visualize room (top-down x–y):
      - walls
      - microphones
      - sources and their motion (location_history)
    Saves to 'outputfile'.
    """
    Lx, Ly, Lz = room_cfg.dimensions_m
    mics = np.asarray(room_cfg.mic_pos, float)   # shape (M, 3)
    assert mics.ndim == 2 and mics.shape[1] == 3

    fig, ax = plt.subplots(figsize=(7, 5))

    # Room rectangle
    ax.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], linewidth=1.5)

    # Mics
    ax.scatter(mics[:, 0], mics[:, 1], marker="^", s=60, label="Mics")
    if annotate:
        for i, (mx, my, mz) in enumerate(mics):
            ax.text(mx, my, f"m{i}", fontsize=8, ha="left", va="bottom")

    # Sources
    for s_idx, src in enumerate(room_cfg.sources):
        # path of this source
        xs, ys = [], []
        for pose in src.location_history:
            x, y, z = pose.location_m
            xs.append(x); ys.append(y)
        xs = np.array(xs); ys = np.array(ys)

        # connect trajectory
        ax.plot(xs, ys, linestyle="--", linewidth=1.0, label=f"src{s_idx} path")
        ax.scatter(xs, ys, s=25)  # waypoints

        # draw facing direction at each waypoint (optional)
        if show_dirs:
            for pose in src.location_history:
                x, y, z = pose.location_m
                _, _, u, v = _arrow_from_pose(x, y, pose.azimuth_deg, scale=0.35)
                ax.arrow(x, y, u, v, head_width=0.10, head_length=0.12, length_includes_head=True)

        # label first waypoint
        if annotate and len(src.location_history) > 0:
            x0, y0, _ = src.location_history[0].location_m
            ax.text(x0, y0, f"s{s_idx}", fontsize=8, ha="right", va="top")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, Lx + 0.2)
    ax.set_ylim(-0.2, Ly + 0.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Room (top-down)")
    if len(room_cfg.sources) + 1 <= 12:
        ax.legend(loc="upper right", fontsize=8)

    outputfile = Path(outputfile)
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outputfile, dpi=dpi)
    plt.close(fig)

def plot_room_3d(room_cfg, outputfile: Path, *, dpi: int = 150) -> None:
    """
    Quick 3D scatter of mics + source waypoints. (No walls/box surfaces for simplicity.)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)
    Lx, Ly, Lz = room_cfg.dimensions_m
    mics = np.asarray(room_cfg.mic_pos, float)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    # mics
    ax.scatter(mics[:,0], mics[:,1], mics[:,2], marker="^", s=50, label="Mics")

    # sources
    for s_idx, src in enumerate(room_cfg.sources):
        pts = np.array([pose.location_m for pose in src.location_history], float)
        ax.plot(pts[:,0], pts[:,1], pts[:,2], linestyle="--", linewidth=1.0, label=f"src{s_idx} path")
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=20)

    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Room (3D)")
    if len(room_cfg.sources) + 1 <= 12:
        ax.legend(loc="upper right", fontsize=8)

    outputfile = Path(outputfile)
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outputfile, dpi=dpi)
    plt.close(fig)
