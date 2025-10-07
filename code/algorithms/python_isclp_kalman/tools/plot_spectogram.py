import numpy as np
import matplotlib.pyplot as plt

def plot_spec(
    spec,
    scale="mag",                    # 'mag' | 'pow' | 'lin'
    x_tick_prop=None,               # (start, res, dist) or None
    y_tick_prop=None,               # (start, res, dist) or None
    c_range=None,                   # (vmin, vmax) or None
    plot_colorbar=True,
    filename="spectrogram.png",     # output filepath
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
        ax.set_xticks([])

    # Y ticks
    if y_tick_prop is not None:
        y_start, y_res, y_dist = y_tick_prop
        y_ticks = np.arange(0, spec.shape[0], int(y_dist))
        y_labels = y_start + y_ticks * y_res
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{v}" for v in np.atleast_1d(y_labels)])
    else:
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
