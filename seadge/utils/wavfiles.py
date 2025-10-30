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

def write_wav(path: pathlib.Path, x: np.ndarray, fs: Optional[int] = None):
    cfg = config.get()
    log.debug(f"Writing wav file {path.relative_to(cfg.paths.output_dir)}")
    if fs is None:
        wavfile.write(path, x, cfg.dsp.datagen_samplerate)
    else:
        wavfile.write(path, x, fs)
