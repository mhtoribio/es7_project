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

    parser.add_argument("--fs", type=int, default=16000, help="Sample Rate (default 16000)")

    # Debug
    parser.add_argument("--debug-data-dir",  type=dir_path_allow_new, help="Output debug data to this directory")

    #scenario
    h = parser.add_mutually_exclusive_group(required=True)
    h.add_argument("--scenario-file", type=dir_path, help="Load a single scenario file")
    h.add_argument("--scenario-dir",  type=dir_path, help="Load scenario files from directory")

    #distant
    d = parser.add_mutually_exclusive_group(required=True)
    d.add_argument("--distant-file", type=dir_path, help="Load a single distant file")
    d.add_argument("--distant-dir",  type=dir_path, help="Load distant files from directory")

    # enhanced speech
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--enhanced-speech-file", type=dir_path, help="Load a single enhanced speech file")
    g.add_argument("--enhanced-speech-dir",  type=dir_path, help="Load enhanced speech files from directory")

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
    scenario_file: Path | None
    scenario_dir: Path | None
    distant_file: Path | None
    distant_dir: Path | None
    enhanced_speech_file: Path | None
    enhanced_speech_dir: Path | None
    log_level: int
    fs: int

def make_config(args: argparse.Namespace) -> Config:
    level = logging.WARNING - 10*min(args.verbose, 2)
    return Config(
        clean_data_dir=args.clean_data_dir,
        debug_data_dir=args.debug_data_dir,
        output_data_dir=args.output_data_dir.expanduser().resolve(),
        scenario_file=args.scenario_file,
        scenario_dir=args.scenario_dir,
        distant_file=args.distant_file,
        distant_dir=args.distant_dir,
        enhanced_speech_file=args.enhanced_speech_file,
        enhanced_speech_dir=args.enhanced_speech_dir,
        log_level=level,
        fs=args.fs
    )

# ------------- app logic ----------------
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
        #add different funciton todo
    elif cfg.scenario_dir:
        logging.info(f"Running multiple scenario mode")
        for sfile in iter_scenarios(cfg.scenario_dir):
            #add different fucntion todo

    if cfg.distant_file:
        logging.info(f"Running single distant mode")
        #add different funciton todo
    elif cfg.distant_dir:
        logging.info(f"Running multiple distant mode")
        for dfile in iter_wav_file(cfg.distant_dir):
            #add different fucntion todo

    if cfg.enhanced_speech_file:
        logging.info(f"Running single enhanced mode")
        #add function todo
    elif cfg.enhanced_speech_dir:
        logging.info(f"Running multiple enhanced speech mode")
        for wfile in iter_wav_file(cfg.enhanced_speech_dir):
            #add function
        
    return 0

def iter_scenarios(scen_dir: Path):
    yield from sorted(scen_dir.glob("*.json"))

def iter_wav_file(wav_dir: Path):
    yield from sorted(wav_dir.glob("*.wav"))

# load clean speech for reference
def load_clean_ref(cfg: Config, scenario: Path):
    logging.info(f"Loading wav files from clean, distant and enhanced speech from {scenario}")
    
    # Load the JSON scenario file
    with open(scenario) as f:
        scen = json.load(f)
    
    # Extract the target speaker spec
    target_spec = scen["target_speaker"]
    scenario_type = scen["scenario_type"]
    scenario_id = scen["scenario_id"]
    wav_path = target_spec["wav_path"]

    # Construct full path to the clean reference file
    logging.debug(f"Processing clean speech ref from {scenario}")
    clean_ref = cfg.clean_data_dir / wav_path

    if not clean_ref.is_file():
        raise FileNotFoundError(clean_ref)

    
    logging.debug(f"Processing enhanced speech from scenario {scenario}")
    distant_ref = cfg.distant_dir / f"{scenario_type}_{scenario_id}_distant.wav"

    logging.debug(f"Processing enhanced speech from scenario {scenario}")
    enhanced_speech_ref = cfg.enhanced_speech_dir / f"{scenario_type}_{scenario_id}_enhanced.wav"
    
    return clean_ref, distant_ref, enhanced_speech_ref

# compute PESQ (how good sound) (with reference)

# compute DNSMOS (ai mos) (no ref)

# compute SRMR (dereverb)

# compute STOI (intelligence how good u can understand)(with ref)

