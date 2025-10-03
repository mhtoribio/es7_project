#!/usr/bin/env python3
import numpy as np
import tqdm
import pyroomacoustics as pra
from pathlib import Path
import matplotlib.pyplot as plt
import itertools

# --------------------------- CONFIG (edit me) ---------------------------

GEN_DEBUG_PLOTS = True

# Sample rate and room acoustics
FS   = 16000                 # Hz
ROOM = (5.0, 6.0, 3.0)       # (Lx, Ly, Lz) meters
RT60 = 0.45                  # target T60 (seconds)
MAX_ORDER = 10               # image-source order (10 is fine for small rooms)

WALL = "front"   # choose: "front" (y≈0) or "back" (y≈Ly)
EPS  = 0.05      # small gap from the wall (meters), must be > 0

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

def wall_center_origin(room, M, spacing, wall="front", eps=0.05):
    Lx, Ly, _ = room
    ox = Lx / 2.0           # centered along the wall
    oy = eps if wall == "front" else (Ly - eps)
    return (ox, oy, 0.0)

MIC_POS = linear_bar(
    origin=wall_center_origin(ROOM, M=8, spacing=0.035, wall=WALL, eps=EPS),
    height=1.20,
    M=8,
    spacing=0.035,
    yaw_deg=0.0,            # 0° = array axis along x ⇒ parallel to that wall
)

# Speaker "seats" to render (x,y,z) meters
def seats_along_table(speaker_height=1.30, speakers_per_side=4, center_x=ROOM[0]/2.0, diff_x=2, start_dist_y=2, diff_dist_y=1):
    seats = []
    for i in range(speakers_per_side):
        seats.append([center_x - diff_x/2.0, start_dist_y + i * diff_dist_y, speaker_height])
        seats.append([center_x + diff_x/2.0, start_dist_y + i * diff_dist_y, speaker_height])
    return np.asarray(seats).T

SEATS = seats_along_table()

OUT_DIR = Path("out/rir_static")

def plot_ir_debug(room, FS, seat_idx=0, mic_idx=0, out_dir="out/debug_plots", early_ms=80, headless=True):
    """
    Save two quick plots for a single RIR:
      - Full impulse response
      - Early window (e.g., first 80 ms)
    """
    import numpy as np
    from pathlib import Path
    import matplotlib

    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    debug_out = Path(out_dir); debug_out.mkdir(parents=True, exist_ok=True)

    # Grab RIR
    rir = np.asarray(room.rir[mic_idx][seat_idx], dtype=float)
    t = np.arange(len(rir)) / FS

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
    N_early = int(early_ms * 1e-3 * FS)
    plt.figure(figsize=(8,3))
    plt.plot(t[:N_early], rir[:N_early])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"RIR seat{seat_idx:02d} \u2192 mic{mic_idx:02d} (first {early_ms} ms)")
    plt.tight_layout()
    plt.savefig(debug_out / f"rir_early{early_ms}ms_s{seat_idx:02d}_m{mic_idx:02d}.png", dpi=150)
    plt.close()

# ------------------------------ RIR render ------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Derive uniform absorption from desired RT60 (Sabine)
    absorption = pra.inverse_sabine(RT60, ROOM)[0]

    room = pra.ShoeBox(p=np.array(ROOM, dtype=float),
                       fs=FS,
                       absorption=absorption,
                       max_order=MAX_ORDER)

    # Add microphone array
    room.add_microphone_array(pra.MicrophoneArray(MIC_POS, fs=FS))
    M = room.mic_array.M
    print(f"Room {ROOM} m, RT60≈{RT60}s (absorption={absorption:.3f}), fs={FS} Hz")
    print(f"Mic array: {M} channels")

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
    ax3.set_box_aspect((ROOM[0], ROOM[1], ROOM[2]))  # equal-ish scaling
    plt.tight_layout()
    plt.savefig(OUT_DIR / "room_3d.png")

    # 2d plot
    fig, ax = plt.subplots()
    ax.add_patch(plt.Rectangle((0, 0), ROOM[0], ROOM[1], fill=False, linewidth=1))
    ax.scatter(MIC_POS[0], MIC_POS[1], marker='^', s=50, label='mics')
    ax.scatter(SEATS[0],   SEATS[1],   marker='o', s=40, label='seats')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, ROOM[0]); ax.set_ylim(0, ROOM[1])
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Room top-down')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "room_2d.png")

    # Compute Room Impulse Response

    room.compute_rir()
    S = len(room.sources)
    for i in range(S):
        rir_list = [room.rir[m][i] for m in range(M)]
        rir = np.asarray(rir_list, dtype=float).T
        np.save(OUT_DIR / f"seat{i:02d}.npy", rir)

    if GEN_DEBUG_PLOTS:
        for i,m in tqdm.tqdm(list(itertools.product(range(S), range(M)))):
            rir = np.asarray(room.rir[m][i], dtype=float)
            plot_ir_debug(room, FS, seat_idx=i, mic_idx=m, early_ms=80, headless=True)
    print(f"Done. Files in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
