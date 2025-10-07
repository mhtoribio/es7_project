import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from pystoi import stoi
import sounddevice as sd

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

    #we assume that since only one scenario it is the same for distant and enhanced
    if cfg.scenario_file:
        logging.info(f"Running single scenario mode")
        process_one_scenario(cfg, cfg.scenario_file)
        
    #since multiple scenarios we have to iterate for each of them
    elif cfg.scenario_dir:
        logging.info(f"Running multiple scenario mode")
        for sfile in iter_scenarios(cfg.scenario_dir):
            process_one_scenario(cfg, sfile)
        
    return 0

def iter_scenarios(scen_dir: Path):
    yield from sorted(scen_dir.glob("*.json"))

def iter_wav_file(wav_dir: Path):
    yield from sorted(wav_dir.glob("*.wav"))

# load clean speech for reference
def load_wav_path(cfg: Config, scen: dict):
    
    # Extract the target speaker spec
    scenario_type = scen["scenario_type"]
    scenario_id = scen["scenario_id"]
    #target_spec = scen["target_speaker"]
    #wav_path = target_spec["wav_path"]

    # Construct full path to the clean reference file
    #logging.debug(f"Processing clean speech ref from {scenario}")
    #clean_wav = cfg.clean_data_dir / wav_path

    #if not clean_wav.is_file():
    #    raise FileNotFoundError(clean_wav)

    # Determine distant wav path
    logging.debug(f"Processing enhanced speech from the scenario")
    if cfg.distant_dir is not None:
        distant_wav = cfg.distant_dir / f"{scenario_type}_{scenario_id}_distant.wav"
    elif cfg.distant_file is not None:
        distant_wav = cfg.distant_file
    else:
        raise ValueError("No distant path provided in config")

    # Determine enhanced wav path
    logging.debug(f"Processing enhanced speech from the scenario")
    if cfg.enhanced_speech_dir is not None:
        enhanced_speech_wav = cfg.enhanced_speech_dir / f"{scenario_type}_{scenario_id}_enhanced.wav"
    elif cfg.enhanced_speech_file is not None:
        enhanced_speech_wav = cfg.enhanced_speech_file
    else:
        raise ValueError("No enhanced speech path provided in config")
    
    return distant_wav, enhanced_speech_wav

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

def prepare_wav_files(cfg: Config, source_spec: dict, noisy: Path) -> tuple[np.ndarray, int]:
    """
    Loads wav source and processes it according to spec.
    Returns: (signal to mix, seat index)
    """
    
    # load clean, distant and enhanced
    x = load_and_resample_source(cfg, source_spec)
    fs_noisy, noisy = wavfile.read(noisy)
    logging.info(f'Loading clean={x} and {noisy=} wav file')

    # ensure lengths are the same
    if len(x) != len(noisy):
        logging.debug(f'Ensure length of clean and noisy are the same')        
        x_padded = np.zeros(len(noisy))
        x_padded[:len(x)] = x            # copy original x into the beginning
        x = x_padded

    return x, noisy

# compute STOI (intelligence how good u can understand)(with ref)
def e_stoi(cfg: Config, clean: np.ndarray, noisy: np.ndarray):
    
    # Convert to mono if stereo
    if clean.ndim > 1:
        logging.debug(f'Converting clean stereo to mono')
        clean = clean[:,0] # choose mic 0
    if noisy.ndim > 1:
        logging.debug(f'Converting noisy stereo to mono')        
        noisy = noisy[:,0] # choose mic 0

    # Compute eSTOI
    estoi_score = stoi(clean, noisy, cfg.fs, extended=True)

    return estoi_score

def process_one_scenario(cfg: Config, scenario: Path):
    
    # Load the JSON scenario file
    with open(scenario) as f:
        scen = json.load(f)

    # find wav path for distant and enhanced
    logging.info(f"Loading wav path from distant and enhanced speech from {scenario}")
    distant_path, enhanced_path = load_wav_path(cfg, scen)

    # prepare, clean and noisy from same scenario
    clean_distant_wav, distant_wav   = prepare_wav_files(cfg, scen["target_speaker"], distant_path)
    clean_enhanced_wav, enhanced_wav = prepare_wav_files(cfg, scen["target_speaker"], enhanced_path)
    logging.debug(f'Both distant and enhanced scenario are prepared')

    # E-STOI metric
    logging.info(f'Processing E-STOI for {distant_path=} and {enhanced_path=}')
    estoi_score_distant  = e_stoi(cfg, clean_distant_wav, distant_wav)
    estoi_score_enhanced = e_stoi(cfg, clean_enhanced_wav, enhanced_wav)
    print(f'Distant score is: {estoi_score_distant}')
    print(f'Enhance score is: {estoi_score_enhanced}')

    ''' Playing Sound
    sd.play(clean_distant_wav, cfg.fs)
    sd.wait()  # wait until playback is finished
    sd.play(distant_wav, cfg.fs)
    sd.wait()  # wait until playback is finished
    sd.play(clean_enhanced_wav, cfg.fs)
    sd.wait()  # wait until playback is finished
    sd.play(enhanced_wav, cfg.fs)
    sd.wait()  # wait until playback is finished
    '''

# ---------- Entrypoint ----------

def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = make_config(args)
    return run(cfg)

if __name__ == "__main__":
    sys.exit(main())



