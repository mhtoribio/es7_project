from pathlib import Path
from typing import Any
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import json

from seadge import config
from seadge.utils.wavfiles import wavfile_frames, wavfile_samplerate
from seadge.utils.log import log

def validate_scenario(scen: dict) -> bool:
    cfg = config.get()
    MAX_DURATION_SAMPLES = cfg.dsp.samplerate * 20

    def _file_has_enough_audio(abs_wav_path: Path, needed_samples: int, decimation: int, interpolation: int, expected_fs: int) -> bool:
        try:
            frames = wavfile_frames(abs_wav_path)
            fs = wavfile_samplerate(abs_wav_path)
        except Exception as e:
            log.debug("Failed to read WAV header %s: %s", abs_wav_path, e)
            return False

        resampled_fs = fs * interpolation // decimation
        if resampled_fs != expected_fs:
            log.debug(f"Sample rate mismatch for {abs_wav_path} (clean {fs}, want {expected_fs}; {decimation=} and {interpolation=} resulting in {resampled_fs}")
            return False

        # frames = per-channel sample count; compare directly to needed per-channel samples
        resampled_frames = frames * interpolation // decimation
        if resampled_frames < needed_samples:
            log.debug("WAV too short: need %d samples, file has %d (%d after resampling)", needed_samples, frames, resampled_frames)
            return False

        return True

    def _req(s: dict, key: str, typ: type | tuple[type, ...]) -> Any:
        if key not in s:
            log.debug("Missing key %r in source %r", key, s)
            raise KeyError(key)
        val = s[key]
        if not isinstance(val, typ):
            log.debug("Key %r has wrong type: %r (expected %r)", key, type(val), typ)
            raise TypeError(key)
        return val

    def _validate_source(src: dict, top_dur: int, label: str) -> bool:
        try:
            delay         = _req(src, "delay_samples", int)
            dur           = _req(src, "duration_samples", int)
            seat          = _req(src, "seat", int)
            vol           = _req(src, "volume", (int, float))
            wrel          = _req(src, "wav_path", str)
            decimation    = _req(src, "decimation", int)
            interpolation = _req(src, "interpolation", int)

            if delay < 0:
                log.debug("%s: negative delay (%d)", label, delay)
                return False
            if dur <= 0:
                log.debug("%s: non-positive duration (%d)", label, dur)
                return False
            if delay + dur > top_dur:
                log.debug("%s: delay+duration (%d) exceeds scenario duration (%d)",
                              label, delay + dur, top_dur)
                return False

            # volume in [0.0, 1.0]
            if not (0.0 <= float(vol) <= 1.0):
                log.debug("%s: volume out of range [0,1]: %r", label, vol)
                return False

            # seat index: 0 <= seat < cfg.seats
            if not (0 <= seat < cfg.room.num_seats):
                log.debug("%s: seat %d out of range [0, %d)", label, seat, cfg.room.num_seats)
                return False

            abs_wav = (cfg.paths.clean_dir / wrel).expanduser().resolve()
            if not abs_wav.is_file():
                log.debug("%s: wav_path invalid: %s", label, wrel)
                return False

            # Check the file is long enough for the source’s own duration
            if not _file_has_enough_audio(abs_wav, dur, decimation, interpolation, cfg.dsp.samplerate):
                log.debug("%s: file shorter than requested duration (%d)", label, dur)
                return False

            return True
        except Exception as e:
            log.debug("%s: validation error: %s", label, e)
            return False

    try:
        scenario_id = _req(scen, "scenario_id", str)
        if len(scenario_id) != 32:
            log.debug("Invalid scenario id (%s)", scenario_id)
            return False

        # top-level duration
        if not isinstance(scen.get("duration_samples"), int):
            log.debug("Top-level duration_samples missing or not int")
            return False
        top_dur = scen["duration_samples"]
        if not (top_dur > 0 and top_dur < MAX_DURATION_SAMPLES):
            log.debug("Invalid top-level duration (%d), must be (0, %d)",
                          top_dur, MAX_DURATION_SAMPLES)
            return False

        # target speaker
        target = scen.get("target_speaker")
        if not isinstance(target, dict):
            log.debug("Missing or invalid target_speaker")
            return False
        if not _validate_source(target, top_dur, "target_speaker"):
            return False

        # other sources (optional but if present must be a list of dicts)
        others = scen.get("other_sources", [])
        if others is None:
            others = []
        if not isinstance(others, list):
            log.debug("other_sources must be a list (got %r)", type(others))
            return False
        for i, src in enumerate(others):
            if not isinstance(src, dict):
                log.debug("other_sources[%d] is not a dict", i)
                return False
            if not _validate_source(src, top_dur, f"other_sources[{i}]"):
                return False

        # optional: scenario_type is allowed to be anything stringy; skip strict check
        return True

    except Exception as e:
        log.debug("Scenario validation failed with exception: %s", e)
        return False

def delay_and_scale_source(output_len: int, x: np.ndarray, delay_samples: int, duration_samples: int, volume: float) -> np.ndarray:
    y = np.zeros(output_len, dtype=float)
    log.debug(f"Delaying source with {delay_samples} samples and duration {duration_samples}, and scaling with {volume}")
    y[delay_samples: delay_samples+duration_samples] = x[:duration_samples] * volume
    return y

def load_and_resample_source(source_spec: dict) -> np.ndarray:
    """
    Loads, normalizes, and resamples source file according to spec JSON object.
    Assumes spec has been validated.
    """
    cfg = config.get()
    fs, x = wavfile.read(cfg.paths.clean_dir / source_spec["wav_path"])
    log.debug(f"Read wavfile with {fs=} and {x.shape=}")
    decimation = source_spec["decimation"]
    interpolation = source_spec["interpolation"]
    x_normalized = (0.99 / (np.max(np.abs(x)) + 1e-12)) * x
    x_resampled = resample_poly(x_normalized, interpolation, decimation)
    log.debug(f"Resampled wavfile with {decimation=} and {interpolation=} from {fs} to {fs*interpolation//decimation} ({x_resampled.shape=})")
    return x_resampled

def prepare_source(source_spec: dict, output_len: int) -> tuple[np.ndarray, int]:
    """
    Loads wav source and processes it according to spec.
    Returns: (signal to mix, seat index)
    """
    log.debug(f"Preparing source {source_spec['wav_path']}")
    x = load_and_resample_source(source_spec)
    x_to_mix = delay_and_scale_source(output_len, x, source_spec["delay_samples"], source_spec["duration_samples"], source_spec["volume"])
    return x_to_mix, source_spec["seat"]

def load_scenario(path: Path) -> dict:
    try:
        with open(path) as f:
            scen = json.load(f)
    except ValueError as e:
        raise ValueError("Error parsing scenario file: ", e)
    return scen
