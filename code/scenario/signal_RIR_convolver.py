import argparse
import itertools
import numpy as np
from scipy.signal import decimate, fftconvolve, resample_poly
from scipy.io import wavfile
from math import gcd
import os
from pathlib import Path

def simulate_mic_from_rir(x, rir):
    """
    x: numpy array for input (N,) shape
    rir: numpy array for RIR (N,) shape
    """
    return fftconvolve(x, rir)

def normalize_and_downsample(x, fs_in, fs_out):
    """
    Normalizes x to float64 and downsamples
    x: numpy array
    fs_in: input sample rate
    fs_out: output sample rate
    """
    x_normalized = (0.99 / np.max(np.abs(x)) + 1e-12) * x
    if fs_in == fs_out:
        return x_normalized
    p, q = fs_out, fs_in
    g = gcd(int(p), int(q))
    p //= g; q //= g
    # polyphase lowpass+resample; good quality and no “rate lying”
    return resample_poly(x_normalized, up=p, down=q)

def rir_convolve_file_to_fs(input_file, rir_files: list, out_file, FS=16000):
    # Read speech input
    fs_recording, x_int16 = wavfile.read(input_file)

    # Normalize and convert to float, and downsample input
    x = normalize_and_downsample(x_int16, fs_recording, FS)

    rirs = []
    for rir_file in rir_files:
        # Load RIR
        rir = np.load(rir_file)
        rirs.append(rir)

    rir_max_len = max([len(rir) for rir in rirs])
    y_len = len(x)+rir_max_len-1
    y = np.zeros((y_len, len(rir_files)), np.float64)

    for i, rir in enumerate(rirs):
        # Convolve with RIR to get signal for microphone m
        y_m = simulate_mic_from_rir(x, rir)

        y[:,i] = y_m

    wavfile.write(out_file, FS, y)

def rir_filename(seat, mic):
    return f"seat{seat:02d}_mic{mic:02d}.npy"

def output_filename(input_filename, seat):
    p = Path(input_filename)
    return f"{os.path.splitext(p.name)[0]}_seat{seat}.wav"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rir_dir")
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    parser.add_argument("--FS", default=16000, type=int)
    args = parser.parse_args()

    FS = args.FS
    RIR_DIR = Path(args.rir_dir)
    OUT_DIR = Path(args.output_dir)
    valid_seats = [i for i in range(8)]
    valid_mics  = [i for i in range(8)]

    for seat in valid_seats:
        out_file = OUT_DIR / output_filename(args.input_file, seat)
        print(out_file)
        rir_files = [RIR_DIR / rir_filename(seat, mic) for mic in valid_mics]
        rir_convolve_file_to_fs(args.input_file, rir_files, out_file)

if __name__ == "__main__":
    main()
