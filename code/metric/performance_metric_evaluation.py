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
import sounddevice as sd
import onnxruntime as ort 
from pystoi import stoi
from srmrpy import srmr
from pesq import pesq
from speechmos import dnsmos




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
    parser.add_argument("--output-data-dir", type=dir_path_allow_new,
                        help="Base directory for output data (M-channel wav files)")

    parser.add_argument("--fs", type=int, default=16000, help="Sample Rate (default 16000)")

    # Txt file

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

    if args.output_data_dir is not None:
        output_dir = args.output_data_dir.expanduser().resolve()
    else:
        output_dir = None

    return Config(
        clean_data_dir=args.clean_data_dir,
        debug_data_dir=args.debug_data_dir,
        output_data_dir=output_dir,
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
    if cfg.output_data_dir is not None:
        logging.info("Will write outputs to .txt files")
        os.makedirs(cfg.output_data_dir, exist_ok=True)

    if cfg.debug_data_dir is not None:
        logging.info("Will write intermediate outputs to debug files")
        os.makedirs(cfg.debug_data_dir, exist_ok=True)

    logging.basicConfig(level=cfg.log_level,
                        format="[%(asctime)s %(levelname)s] %(message)s")

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

def stereo_to_mono(clean: np.ndarray, noisy: np.ndarray):
    # Convert to mono if stereo
    if clean.ndim > 1:
        logging.debug(f'Converting clean stereo to mono')
        clean = clean[:,0] # choose mic 0
    if noisy.ndim > 1:
        logging.debug(f'Converting noisy stereo to mono')        
        noisy = noisy[:,0] # choose mic 0
    
    return clean, noisy

# Convert to .txt file

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

    # stereo to mono
    clean_distant_mono, distant_mono   = stereo_to_mono(clean_distant_wav, distant_wav)
    clean_enhanced_mono, enhanced_mono = stereo_to_mono(clean_enhanced_wav, enhanced_wav)

    # E-STOI metric (range is (0 to 1))
    logging.info(f'Processing E-STOI for {distant_path=} and {enhanced_path=}')
    estoi_score_distant  = stoi(clean_distant_mono, distant_mono, cfg.fs, extended=True)
    estoi_score_enhanced = stoi(clean_enhanced_mono, enhanced_mono, cfg.fs, extended=True)

    # SRMR metric (range is (0 to 20))
    logging.info(f'Processing SRMR for {distant_path=} and {enhanced_path=}')
    srmr_score_distant, _  = srmr(distant_mono, cfg.fs, fast=True)  # We tried with mono and wav, they give same result. Also try with stereo 
    srmr_score_enhanced, _ = srmr(enhanced_mono, cfg.fs, fast=True)

    if cfg.fs == 16000:
        # PESQ metric (range is (-0.5 to 4.64) for 16kHz)
        logging.info(f'Processing PESQ {distant_path=} and {enhanced_path=}')
        pesq_score_distant  = pesq(cfg.fs, clean_distant_mono, distant_mono, 'wb')    # 'wb (16 kHz)' 'nb (8 kHz)'
        pesq_score_enhanced = pesq(cfg.fs, clean_enhanced_mono, enhanced_mono, 'wb')    
    
        # convert int16 to float (DNSMOS is in range (-1,1))
        distant_mono_float = distant_mono.astype(np.float32) / 32768.0
        enhanced_mono_float = enhanced_mono.astype(np.float32) / 32768.0    

        # DNSMOS metric (range is (1 to 5)) (only 8kHz and 16kHz sample rate work for DNSMOS)
        logging.info(f'Processing DNSMOS {distant_path=} and {enhanced_path=}')
        dnsmos_score_distant  = dnsmos.run(distant_mono_float, cfg.fs)
        dnsmos_score_enhanced = dnsmos.run(enhanced_mono_float, cfg.fs)
    
    else:
        logging.warning(f'PESQ and DNSMOS needs 16 kHz sample rate. Therefore, they are set to 0')
        pesq_score_distant    = 0
        pesq_score_enhanced   = 0
        dnsmos_score_distant  = {"ovrl_mos": 0, "sig_mos": 0, "bak_mos": 0}
        dnsmos_score_enhanced = {"ovrl_mos": 0, "sig_mos": 0, "bak_mos": 0}

    if cfg.output_data_dir is None:
        logging.info(f'Printing metric scores')
        # E-STOI
        print(f'Distant E-STOI score is: {estoi_score_distant}')
        print(f'Enhance E-STOI score is: {estoi_score_enhanced}')

        # SRMR
        print(f'Distant SRMR score is: {srmr_score_distant}')
        print(f'Enhance SRMR score is: {srmr_score_enhanced}')

        # PESQ
        print(f'Distant PESQ score is: {pesq_score_distant}')
        print(f'Enhance PESQ score is: {pesq_score_enhanced}')

        # DNSMOS
        print(f'Distant DNSMOS overall score: {dnsmos_score_distant['ovrl_mos']}, speech quality score: {dnsmos_score_distant['sig_mos']}, background noise score {dnsmos_score_distant['bak_mos']}')   # p808 = P.808 mapping   
        print(f'Enhance DNSMOS overall score: {dnsmos_score_enhanced['ovrl_mos']}, speech quality score: {dnsmos_score_enhanced['sig_mos']}, background noise score {dnsmos_score_enhanced['bak_mos']}')
    
    # Save txt file
    else:
        logging.info(f'Making .txt file for metric scores')

        # Prepare metric table
        metrics = [
        ("E-STOI", estoi_score_distant, estoi_score_enhanced),
        ("SRMR", srmr_score_distant, srmr_score_enhanced),
        ("PESQ", pesq_score_distant, pesq_score_enhanced),
        ("DNSMOS_ovrl", dnsmos_score_distant['ovrl_mos'], dnsmos_score_enhanced['ovrl_mos']),
        ("DNSMOS_sig",  dnsmos_score_distant['sig_mos'],  dnsmos_score_enhanced['sig_mos']),
        ("DNSMOS_bak",  dnsmos_score_distant['bak_mos'],  dnsmos_score_enhanced['bak_mos']),
        ]

        # File path: use scenario name as filename
        out_file = cfg.output_data_dir / f"{scenario.stem}_metrics.txt"

        # Write to file
        with open(out_file, "w") as f:
            f.write(f"{'Metric':<15}{'Distant':>12}{'Enhanced':>12}\n")
            f.write("="*39 + "\n")
            for name, distant, enhanced in metrics:
                f.write(f"{name:<15}{distant:>12.4f}{enhanced:>12.4f}\n")

        logging.info(f"Saved metric scores to {out_file}")


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

