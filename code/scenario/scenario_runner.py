import argparse
import wave
import sys
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import os
from pprint import pprint
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly
from functools import lru_cache

# ---------- CLI ----------

def dir_path(p: str) -> Path:
    pth = Path(p).expanduser().resolve()
    if not pth.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {pth}")
    return pth

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

    # Data dirs
    parser.add_argument("--clean-data-dir",  type=dir_path, required=True,
                        help="Base directory for clean data (wav files)")
    parser.add_argument("--output-data-dir", type=dir_path_allow_new,     required=True,
                        help="Base directory for output data (M-channel wav files)")
    parser.add_argument("--rir-data-dir",    type=dir_path, required=True,
                        help="Base directory for Room Impulse Responses")

    # Number of seats
    parser.add_argument("--seats", type=int, default=8, help="Number of seats (default 8)")
    parser.add_argument("--fs", type=int, default=16000, help="Sample Rate (default 16000)")

    # Debug
    parser.add_argument("--debug-data-dir",  type=dir_path_allow_new, help="Output debug data to this directory")

    # Scenarios
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario-file", type=dir_path, help="Load a single scenario file")
    g.add_argument("--scenario-dir",  type=dir_path, help="Load scenario files from directory")

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v, -vv)")

    args = parser.parse_args(argv)
    return args

# ---------- Config ----------

@dataclass(frozen=True)
class Config:
    clean_data_dir: Path
    debug_data_dir: Path | None
    output_data_dir: Path
    rir_data_dir: Path
    scenario_file: Path | None
    scenario_dir: Path | None
    log_level: int
    seats: int
    fs: int

def make_config(args: argparse.Namespace) -> Config:
    level = logging.WARNING - 10*min(args.verbose, 2)
    return Config(
        clean_data_dir=args.clean_data_dir,
        debug_data_dir=args.debug_data_dir,
        output_data_dir=args.output_data_dir.expanduser().resolve(),
        rir_data_dir=args.rir_data_dir,
        scenario_file=args.scenario_file,
        scenario_dir=args.scenario_dir,
        log_level=level,
        seats=args.seats,
        fs=args.fs
    )

# --- RIR loading & cache ---

from functools import lru_cache

def rir_filename(seat: int) -> str:
    return f"seat{seat:02d}.npy"

@lru_cache(maxsize=4096)
def _load_rir_cached(rir_dir_str: str, seat: int, mmap: bool) -> np.ndarray:
    """Internal: cached loader keyed by dir+seat(+mmap)."""
    path = Path(rir_dir_str) / rir_filename(seat)
    if not path.is_file():
        raise FileNotFoundError(path)
    mmap_mode = 'r' if mmap else None
    return np.load(path, mmap_mode=mmap_mode)  # returns np.memmap if mmap_mode set

def get_rir(cfg: Config, seat: int, *, mmap: bool = True, writeable: bool = False) -> np.ndarray:
    """
    Fetch a RIR array for seat, cached across calls.
    By default returns a read-only memmap (cheap). Set writeable=True to return a copy.
    """
    arr = _load_rir_cached(str(cfg.rir_data_dir), seat, mmap)
    return arr.copy() if writeable else arr

def clear_rir_cache() -> None:
    """If RIR files change on disk, clear the cache before reloading."""
    _load_rir_cached.cache_clear()

# ---------- App logic ----------

def run(cfg: Config) -> int:
    # Ensure output dirs exist
    os.makedirs(cfg.output_data_dir, exist_ok=True)
    if cfg.debug_data_dir is not None:
        os.makedirs(cfg.debug_data_dir, exist_ok=True)

    logging.basicConfig(level=cfg.log_level,
                        format="[%(asctime)s %(levelname)s] %(message)s")

    if cfg.debug_data_dir is not None:
        logging.info("Will write intermediate outputs to debug files")

    if cfg.scenario_file:
        logging.info(f"Running single scenario mode")
        process_one_scenario(cfg, cfg.scenario_file)
    elif cfg.scenario_dir:
        logging.info(f"Running multiple scenario mode")
        for sfile in iter_scenarios(cfg.scenario_dir):
            process_one_scenario(cfg, sfile)

    return 0

def iter_scenarios(scen_dir: Path):
    yield from sorted(scen_dir.glob("*.json"))


def validate_scenario(cfg: Config, scen: dict) -> bool:
    MAX_DURATION_SAMPLES = cfg.fs * 20

    def _wav_frames_and_sr(path: Path) -> tuple[int, int]:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()       # frames per channel
            fs     = w.getframerate()
        return frames, fs

    def _file_has_enough_audio(abs_wav_path: Path, needed_samples: int, decimation: int, interpolation: int, expected_fs: int = 16000) -> bool:
        try:
            frames, fs = _wav_frames_and_sr(abs_wav_path)
        except Exception as e:
            logging.debug("Failed to read WAV header %s: %s", abs_wav_path, e)
            return False

        resampled_fs = fs * interpolation // decimation
        if resampled_fs != expected_fs:
            logging.debug(f"Sample rate mismatch for {abs_wav_path} (clean {fs}, want {expected_fs}; {decimation=} and {interpolation=} resulting in {resampled_fs}")
            return False

        # frames = per-channel sample count; compare directly to needed per-channel samples
        resampled_frames = frames * interpolation // decimation
        if resampled_frames < needed_samples:
            logging.debug("WAV too short: need %d samples, file has %d (%d after resampling)", needed_samples, frames, resampled_frames)
            return False

        return True

    def _req(s: dict, key: str, typ: type | tuple[type, ...]) -> any:
        if key not in s:
            logging.debug("Missing key %r in source %r", key, s)
            raise KeyError(key)
        val = s[key]
        if not isinstance(val, typ):
            logging.debug("Key %r has wrong type: %r (expected %r)", key, type(val), typ)
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
                logging.debug("%s: negative delay (%d)", label, delay)
                return False
            if dur <= 0:
                logging.debug("%s: non-positive duration (%d)", label, dur)
                return False
            if delay + dur > top_dur:
                logging.debug("%s: delay+duration (%d) exceeds scenario duration (%d)",
                              label, delay + dur, top_dur)
                return False

            # volume in [0.0, 1.0]
            if not (0.0 <= float(vol) <= 1.0):
                logging.debug("%s: volume out of range [0,1]: %r", label, vol)
                return False

            # seat index: 0 <= seat < cfg.seats
            if not (0 <= seat < cfg.seats):
                logging.debug("%s: seat %d out of range [0, %d)", label, seat, cfg.seats)
                return False

            abs_wav = (cfg.clean_data_dir / wrel).expanduser().resolve()
            if not abs_wav.is_file():
                logging.debug("%s: wav_path invalid: %s", label, wrel)
                return False

            # Check the file is long enough for the source’s own duration
            if not _file_has_enough_audio(abs_wav, dur, decimation, interpolation, expected_fs=cfg.fs):
                logging.debug("%s: file shorter than requested duration (%d)", label, dur)
                return False

            return True
        except Exception as e:
            logging.debug("%s: validation error: %s", label, e)
            return False

    try:
        scenario_id = _req(scen, "scenario_id", str)
        if len(scenario_id) != 32:
            logging.debug("Invalid scenario id (%s)", scenario_id)
            return False

        # top-level duration
        if not isinstance(scen.get("duration_samples"), int):
            logging.debug("Top-level duration_samples missing or not int")
            return False
        top_dur = scen["duration_samples"]
        if not (top_dur > 0 and top_dur < MAX_DURATION_SAMPLES):
            logging.debug("Invalid top-level duration (%d), must be (0, %d)",
                          top_dur, MAX_DURATION_SAMPLES)
            return False

        # target speaker
        target = scen.get("target_speaker")
        if not isinstance(target, dict):
            logging.debug("Missing or invalid target_speaker")
            return False
        if not _validate_source(target, top_dur, "target_speaker"):
            return False

        # other sources (optional but if present must be a list of dicts)
        others = scen.get("other_sources", [])
        if others is None:
            others = []
        if not isinstance(others, list):
            logging.debug("other_sources must be a list (got %r)", type(others))
            return False
        for i, src in enumerate(others):
            if not isinstance(src, dict):
                logging.debug("other_sources[%d] is not a dict", i)
                return False
            if not _validate_source(src, top_dur, f"other_sources[{i}]"):
                return False

        # optional: scenario_type is allowed to be anything stringy; skip strict check
        return True

    except Exception as e:
        logging.debug("Scenario validation failed with exception: %s", e)
        return False

def delay_and_scale_source(output_len: int, x: np.ndarray, delay_samples: int, duration_samples: int, volume: float) -> np.ndarray:
    y = np.zeros(output_len, dtype=float)
    logging.debug(f"Delaying source with {delay_samples} samples and duration {duration_samples}, and scaling with {volume}")
    y[delay_samples: delay_samples+duration_samples] = x[:duration_samples] * volume
    return y

def load_and_resample_source(cfg: Config, source_spec: dict) -> np.ndarray:
    """
    Loads, normalizes, and resamples source file according to spec JSON object.
    Assumes spec has been validated.
    """
    fs, x = wavfile.read(cfg.clean_data_dir / source_spec["wav_path"])
    logging.debug(f"Read wavfile with {fs=} and {x.shape=}")
    decimation = source_spec["decimation"]
    interpolation = source_spec["interpolation"]
    x_normalized = (0.99 / (np.max(np.abs(x)) + 1e-12)) * x
    x_resampled = resample_poly(x_normalized, interpolation, decimation)
    logging.debug(f"Resampled wavfile with {decimation=} and {interpolation=} from {fs} to {fs*interpolation//decimation} ({x_resampled.shape=})")
    return x_resampled

def prepare_source(cfg: Config, source_spec: dict, output_len: int) -> tuple[np.ndarray, int]:
    """
    Loads wav source and processes it according to spec.
    Returns: (signal to mix, seat index)
    """
    logging.debug(f"Preparing source {source_spec['wav_path']}")
    x = load_and_resample_source(cfg, source_spec)
    x_to_mix = delay_and_scale_source(output_len, x, source_spec["delay_samples"], source_spec["duration_samples"], source_spec["volume"])
    return x_to_mix, source_spec["seat"]

def compute_distant_seat_sources(cfg: Config, clean_seat_ch: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for i, x in enumerate(clean_seat_ch):
        rir = get_rir(cfg, i)
        y = fftconvolve(x[:,None], rir, mode="full")
        logging.debug(f"Convolved seat {i} clean with respective RIR, {x.shape=} {rir.shape=} {y.shape=}")
        out.append(y)
    return out

def process_one_scenario(cfg: Config, scenario: Path):
    logging.info(f"Processing {scenario}")

    with open(scenario) as f:
        scen = json.load(f)

    if not validate_scenario(cfg, scen):
        logging.warning(f"Skipping invalid scenario file {scenario}")
        return 1

    # Create clean seat sources
    clean_seat_ch = [np.zeros(scen["duration_samples"], dtype=float) for _ in range(cfg.seats)]

    # Mix clean seat sources
    logging.debug(f"Mixing target source")
    x, seat = prepare_source(cfg, scen["target_speaker"], scen["duration_samples"])
    clean_seat_ch[seat] += x
    for source in scen["other_sources"]:
        logging.debug(f"Mixing other source")
        x, seat = prepare_source(cfg, source, scen["duration_samples"])
        clean_seat_ch[seat] += x

    # Output debug files
    if cfg.debug_data_dir is not None:
        logging.debug("Writing debug files for clean seat mono audio")
        for i, x in enumerate(clean_seat_ch):
            name = f'{scen["scenario_type"]}_{scen["scenario_id"]}_clean_seat{i}.wav'
            logging.debug(f"Writing debug file {name}")
            wavfile.write(cfg.debug_data_dir / name, cfg.fs, x)

    # Convolve clean seat sources with respective RIR
    logging.debug("Computing distant seat M-channel audio")
    distant_seat_ch = compute_distant_seat_sources(cfg, clean_seat_ch)

    # Output debug files
    if cfg.debug_data_dir is not None:
        logging.debug("Writing debug files for distant seat mono audio")
        for i, x in enumerate(distant_seat_ch):
            name = f'{scen["scenario_type"]}_{scen["scenario_id"]}_distant_seat{i}.wav'
            logging.debug(f"Writing debug file {name}")
            wavfile.write(cfg.debug_data_dir / name, cfg.fs, x)

    # Mix distant seat sources to single file
    logging.debug("Mixing distant sources to single array")
    shapes = [x.shape for x in distant_seat_ch]
    out_len, M = map(max, zip(*shapes))
    y = np.zeros((out_len, M), dtype=float)
    logging.debug(f"Output shape is {y.shape}")
    for i, x in enumerate(distant_seat_ch):
        logging.debug(f"Mixing seat {i}")
        y[:x.shape[0],:] += x

    # Normalize y
    y = (0.99 / (np.max(np.abs(y)) + 1e-12)) * y

    # Save to output wav file
    name = f'{scen["scenario_type"]}_{scen["scenario_id"]}_distant.wav'
    wavfile.write(cfg.output_data_dir / name, cfg.fs, y)

# ---------- Entrypoint ----------

def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = make_config(args)
    return run(cfg)

if __name__ == "__main__":
    sys.exit(main())
