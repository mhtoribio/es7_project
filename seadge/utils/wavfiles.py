from typing import Optional
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import wave
import pathlib

from seadge import config
from seadge.utils.log import log

def wavfile_frames(p: pathlib.Path) -> int:
    with wave.open(str(p), "rb") as w:
        frames = w.getnframes()       # frames per channel
    return frames

def wavfile_samplerate(p: pathlib.Path) -> int:
    with wave.open(str(p), "rb") as w:
        fs = w.getframerate()
    return fs

def load_and_resample_source(source_spec: dict) -> np.ndarray:
    """
    Loads, normalizes, and resamples source file according to spec JSON object.
    Assumes spec has been validated.
    """
    fs, x = wavfile.read(config.get().paths.clean_dir / source_spec["wav_path"])
    log.debug(f"Read wavfile with {fs=} and {x.shape=}")
    decimation = source_spec["decimation"]
    interpolation = source_spec["interpolation"]
    x_normalized = (0.99 / (np.max(np.abs(x)) + 1e-12)) * x
    x_resampled = resample_poly(x_normalized, interpolation, decimation)
    log.debug(f"Resampled wavfile with {decimation=} and {interpolation=} from {fs} to {fs*interpolation//decimation} ({x_resampled.shape=})")
    return x_resampled

def load_wav(path: pathlib.Path, expected_fs: Optional[int] = None, expected_ndim: Optional[int] = None, mmap: bool = False) -> np.ndarray:
    """
    Loads wav file with the option to provide expected samplerate and ndim.
    NOTE! Will intentionally crash if samplerate or ndim do not match.
    """
    fs_wav, s = wavfile.read(path, mmap=mmap)
    if expected_fs:
        if fs_wav != expected_fs:
            log.error(f"Mismatch fs for {path}: {fs_wav=} != {expected_fs=}")
            raise ValueError
    if expected_ndim:
        if s.ndim != expected_ndim:
            log.error(f"Mismatch ndim for {path}: {s.ndim=} != {expected_ndim=} ({s.shape=})")
            raise ValueError
    return s

def write_wav(path: pathlib.Path, x: np.ndarray, fs: int):
    cfg = config.get()
    path.parent.mkdir(exist_ok=True)
    log.debug(f"Writing {x.shape} array to wav file {path.relative_to(cfg.paths.output_dir)}")
    wavfile.write(path, fs, x)
