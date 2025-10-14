#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import logging
from dataclasses import dataclass
import tqdm
import pyroomacoustics as pra
from pathlib import Path
import matplotlib.pyplot as plt
import itertools

#--------------------------- CLI -----------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    def dir_path_allow_new(s: str) -> Path:
        p = Path(s).expanduser()
        # Don't use strict=True so non-existing paths are fine
        p = p.resolve(strict=False)

        # If it exists, it must be a directory
        if p.exists() and not p.is_dir():
            raise argparse.ArgumentTypeError(f"Not a directory: {p}")

        # Optional: sanity check parent if it doesn't exist yet
        # (you can drop this if you truly don't care)
        # if not p.exists() and not p.parent.exists():
        #     raise argparse.ArgumentTypeError(f"Parent does not exist: {p.parent}")

        return p

    parser = argparse.ArgumentParser()

    # data directories
    parser.add_argument("--output-data-dir", type=dir_path_allow_new, required=True,
                        help="Base directory for output data (M-channel wav files)")
    # debug directories
    parser.add_argument("--debug-data-dir",  type=dir_path_allow_new, help="Output debug data to this directory")
    
    # pyroom parameters
    parser.add_argument("--fs", type=int, default=16000, help="Sample rate in Hz (default 16000)")
    parser.add_argument("--room-dimensions", type=float, nargs=3, metavar=('Lx', 'Ly', 'Lz'), 
                        default= (5.0, 6.0, 3.0), help="Room dimensions in meters(Lx, Ly, Lz) (default = (5.0, 6.0, 3.0))")
    parser.add_argument("--rt60", type=float, default=0.45, help="RT60 value in seconds (default 0.45)")
    parser.add_argument("--max-order", type=int, default=10, help="image-source order (default=10 is fine for small rooms)")
    parser.add_argument("--eps", type=float, default=0.05, 
                        help="small gap from the wall (meters), must be > 0 (default = 0.05)")

    parser.add_argument("-v","--verbose", action="count", default=0, 
                        help="Increase verbosity (-v, -vv)")
    
    args = parser.parse_args()
    return args

# ---------- Config ----------

@dataclass(frozen=True)
class Config:
    debug_data_dir: Path | None
    output_data_dir: Path
    room_dimensions: float
    rt60: float
    eps: float
    max_order: int
    log_level: int
    fs: int

def make_config(args: argparse.Namespace) -> Config:
    level = logging.WARNING - 10*min(args.verbose, 2)
    return Config(
        debug_data_dir=args.debug_data_dir,
        output_data_dir=args.output_data_dir.expanduser().resolve(),
        room_dimensions=args.room_dimensions,
        rt60=args.rt60,
        eps=args.eps,
        max_order=args.max_order, 
        log_level=level,
        fs=args.fs
    )

# --------------------------- Parameters --------------------------

WALL = "front"   # choose: "front" (y≈0) or "back" (y≈Ly)

# Microphone array (set your exact capsule coordinates here, meters)
# Example: an 8-mic linear bar on the wall at y=2.5 m, 1.2 m high, 3.5 cm spacing
# You can replace this whole block with your measured coordinates.
def linear_bar(origin=(0.1, 2.5, 0.0), height=1.2, M=8, spacing=0.035, yaw_deg=0.0):
    ox, oy, oz = origin
    half = (M - 1) * spacing / 2.0
    xs = np.linspace(-half, half, M)
    ys = np.zeros(M)
    zs = np.zeros(M)
    P = np.stack([xs, ys, zs], axis=1)
    yaw = np.deg2rad(yaw_deg)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw),  np.cos(yaw), 0.0],
                   [0.0,          0.0,         1.0]])
    P = (P @ Rz.T)
    P[:,0] += ox
    P[:,1] += oy
    P[:,2] += height
    return P.T  # (3, M)

def wall_center_origin(cfg: Config, M, spacing, wall="front"):
    Lx, Ly, _ = cfg.room_dimensions
    ox = Lx / 2.0           # centered along the wall
    oy = cfg.eps if wall == "front" else (Ly - cfg.eps)
    return (ox, oy, 0.0)



# Speaker "seats" to render (x,y,z) meters
def seats_along_table(cfg: Config, speaker_height=1.30, speakers_per_side=4, diff_x=2, start_dist_y=2, diff_dist_y=1):
    center_x = cfg.room_dimensions[0] / 2.0
    seats = []
    for i in range(speakers_per_side):
        seats.append([center_x - diff_x / 2.0, start_dist_y + i * diff_dist_y, speaker_height])
        seats.append([center_x + diff_x / 2.0, start_dist_y + i * diff_dist_y, speaker_height])
    return np.asarray(seats).T


def plot_ir_debug(cfg: Config, room, seat_idx=0, mic_idx=0, debug_dir: Path | str = None, early_ms=80, headless=True):
    """
    Save two quick plots for a single RIR:
      - Full impulse response
      - Early window (e.g., first 80 ms)
    """
    if debug_dir is None:
        # user didn't request debug output: skip plotting
        return

    import numpy as np
    from pathlib import Path
    import matplotlib

    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    debug_out = Path(debug_dir)
    debug_out.mkdir(parents=True, exist_ok=True)

    # Grab RIR
    rir = np.asarray(room.rir[mic_idx][seat_idx], dtype=float)
    t = np.arange(len(rir)) / cfg.fs
    # Full IR
    plt.figure(figsize=(8,3))
    plt.plot(t, rir)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"RIR seat{seat_idx:02d} \u2192 mic{mic_idx:02d} (full)")
    plt.tight_layout()
    plt.savefig(debug_out / f"rir_full_s{seat_idx:02d}_m{mic_idx:02d}.png", dpi=150)
    plt.close()

    # Early window
    N_early = int(early_ms * 1e-3 * cfg.fs)
    plt.figure(figsize=(8,3))
    plt.plot(t[:N_early], rir[:N_early])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"RIR seat{seat_idx:02d} \u2192 mic{mic_idx:02d} (first {early_ms} ms)")
    plt.tight_layout()
    plt.savefig(debug_out / f"rir_early{early_ms}ms_s{seat_idx:02d}_m{mic_idx:02d}.png", dpi=150)
    plt.close()



# ------------------------------ RIR render ------------------------------

def run(cfg: Config):
    out_dir = cfg.output_data_dir / "rir_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_out_dir = cfg.debug_data_dir
    if debug_out_dir is not None:
        debug_out_dir = debug_out_dir / "plots"
        debug_out_dir.mkdir(parents=True, exist_ok=True)

    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive uniform absorption from desired RT60 (Sabine)
    absorption = pra.inverse_sabine(cfg.rt60, cfg.room_dimensions)[0]

    room = pra.ShoeBox(p=np.array(cfg.room_dimensions, dtype=float),
                    fs=cfg.fs,
                    absorption=absorption,
                    max_order=cfg.max_order)

    # Add microphone array
    MIC_POS = linear_bar(origin=wall_center_origin(cfg, M=8, spacing=0.035, wall=WALL), height=1.20,
    M=8, spacing=0.035, yaw_deg=0.0)            # 0° = array axis along x ⇒ parallel to that wall


    SEATS = seats_along_table(cfg)

    room.add_microphone_array(pra.MicrophoneArray(MIC_POS, fs=cfg.fs))
    M = room.mic_array.M
    logging.info(f"Room {cfg.room_dimensions} m, RT60≈{cfg.room_dimensions}s (absorption={absorption:.3f}), fs={cfg.fs} Hz")
    logging.info(f"Mic array: {M} channels")

    for i in range(SEATS.shape[1]):
        s = SEATS[:, i]
        room.add_source(s)

    # 3d plot
    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    room.plot(ax=ax3)  # pyroomacoustics expects a 3D axis
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.set_zlabel('z [m]')
    ax3.set_title('Room (3D)')
    ax3.view_init(elev=35, azim=20)
    ax3.set_box_aspect((cfg.room_dimensions[0], cfg.room_dimensions[1], cfg.room_dimensions[2]))  # equal-ish scaling
    plt.tight_layout()
    plt.savefig(out_dir / "room_3d.png")

    # 2d plot
    fig, ax = plt.subplots()
    ax.add_patch(plt.Rectangle((0, 0), cfg.room_dimensions[0], cfg.room_dimensions[1], fill=False, linewidth=1))
    ax.scatter(MIC_POS[0], MIC_POS[1], marker='^', s=50, label='mics')
    ax.scatter(SEATS[0],   SEATS[1],   marker='o', s=40, label='seats')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, cfg.room_dimensions[0]); ax.set_ylim(0, cfg.room_dimensions[1])
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Room top-down')
    plt.tight_layout()
    plt.savefig(out_dir / "room_2d.png")

    # Compute Room Impulse Response

    room.compute_rir()
    S = len(room.sources)
    for i in range(S):
        rir_list = [room.rir[m][i] for m in range(M)]
        rir = np.asarray(rir_list, dtype=float).T
        np.save(out_dir / f"seat{i:02d}.npy", rir)

    if cfg.debug_data_dir is not None:
        for i,m in tqdm.tqdm(list(itertools.product(range(S), range(M)))):
            rir = np.asarray(room.rir[m][i], dtype=float)
            plot_ir_debug(cfg, room, seat_idx=i, mic_idx=m, debug_dir=debug_out_dir, early_ms=80, headless=True)
    logging.info(f"Done. Files in: {out_dir.resolve()}")

def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = make_config(args)
    return run(cfg)

if __name__ == "__main__":
    sys.exit(main())
