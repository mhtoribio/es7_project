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

def pcm_to_float64(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        return x.astype(np.float64, copy=False)
    elif x.dtype == np.int16:
        return x.astype(np.float64, copy=False) * (1/32768)
    elif x.dtype == np.int32:
        return x.astype(np.float64, copy=False) * (1/2147483648)
    elif x.dtype == np.uint8:
        log.error(f"PCM8 not supported")
        raise ValueError
    else:
        log.error(f"Unknown file format")
        raise ValueError

def pcm_to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        return x.astype(np.float32, copy=False)
    elif x.dtype == np.int16:
        return x.astype(np.float32, copy=False) * (1/32768)
    elif x.dtype == np.int32:
        return x.astype(np.float32, copy=False) * (1/2147483648)
    elif x.dtype == np.uint8:
        log.error(f"PCM8 not supported")
        raise ValueError
    else:
        log.error(f"Unknown file format")
        raise ValueError

def load_wav(path: pathlib.Path, expected_fs: Optional[int] = None, expected_ndim: Optional[int] = None, mmap: bool = False, dtype: Optional[type] = None) -> tuple[int, np.ndarray]:
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
    if dtype:
        if dtype == np.float64:
            s = pcm_to_float64(s)
        elif dtype == np.float32:
            s = pcm_to_float32(s)
        else:
            log.error(f"Invalid dtype {dtype} in load_wav")
            raise ValueError
    log.debug(f"Loaded wavfile {path} with {fs_wav=} and dtype {s.dtype}. {expected_fs=}, {expected_ndim=}, {mmap=}, {dtype=}")
    return fs_wav, s

def write_wav(path: pathlib.Path, x: np.ndarray, fs: int):
    cfg = config.get()
    path.parent.mkdir(exist_ok=True)
    log.debug(f"Writing {x.shape} array to wav file {path.relative_to(cfg.paths.output_dir)}")
    wavfile.write(path, fs, x)
