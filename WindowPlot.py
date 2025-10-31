import numpy as np
import pathlib
import matplotlib.pyplot as plt

# Initialize
y0 = np.zeros((165334, 3))   # total length, 3 windows
fade_len = 3072

# Define filenames (or generate dynamically)
fnames = [
    "win0_0_16000.npy",
    "win1_16000_32000.npy",
    "win2_32000_165334.npy"
]

folder = pathlib.Path(r"C:\code\PlotsforWindow")
wins = []

# Loop over all files
for idx, fname in enumerate(fnames):
    fpath = folder / fname

    # Load window (assuming .npy)
    win = np.load(fpath)
    wins.append(win)

    # Parse start and end indices from filename
    parts = fname.replace(".npy", "").split("_")
    s0 = int(parts[1])
    s1 = int(parts[2])

    # Place window into correct position
    y0[s0:s1+fade_len, idx] = win[:s1-s0+fade_len]

# Plot results
# plt.plot(y0[:, 0], label="win0")
# plt.plot(y0[:, 1], label="win1")
# plt.plot(y0[:, 2], label="win2")
# plt.legend()
# plt.show()

# Second plot: annotate win1
plt.figure()
plt.plot(wins[1], label="win1", color='tab:blue')

# Get info from file name
parts = fnames[1].replace(".npy", "").split("_")
start_sample_i = int(parts[1])       # e.g. 16000
start_sample_next = int(parts[2])    # e.g. 32000

# Plot dashed vertical lines for start/end
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.text(0 + 200, 0.8, "start_sample[i]", va='center', ha='left', fontsize=9, color='black')
plt.text(0 + 900, 1, "Sin Crossfade", va='center', ha='left', fontsize=9, color='black')


plt.axvline(x=(start_sample_next-start_sample_i), color='black', linestyle='--', linewidth=1)
plt.text(start_sample_next-start_sample_i-200, 0.8, "start_sample[i+1]", va='top', ha='right', fontsize=9, color='black')
plt.text(start_sample_next-start_sample_i+fade_len-900, 1, "Sin Crossfade", va='top', ha='right', fontsize=9, color='black')

plt.axvline(x=0+fade_len, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=(start_sample_next-start_sample_i+fade_len), color='gray', linestyle='--', linewidth=1)

# Mark fade region with <---->
xfade_start = 0
xfade_end = fade_len

plt.annotate(
    '', 
    xy=(xfade_end, 0), xycoords='data',
    xytext=(xfade_start, 0), textcoords='data',
    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5)
)
plt.text((xfade_start + xfade_end) / 2, 0.05, "xfade_len", color='red', ha='center', va='bottom')

# (Optional) Also mark end fade region visually
xfade_end_start = len(wins[1]) - fade_len
xfade_end_end = len(wins[1])

plt.annotate(
    '', 
    xy=(xfade_end_end, 0), xycoords='data',
    xytext=(xfade_end_start, 0), textcoords='data',
    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5)
)
plt.text((xfade_end_start + xfade_end_end) / 2, 0.05, "xfade_len", color='red', ha='center', va='bottom')

plt.legend()
plt.title("Window 1 with Crossfade Annotations")
plt.xlabel("Samples (relative to window start)")
plt.ylabel("Amplitude")
plt.show()
