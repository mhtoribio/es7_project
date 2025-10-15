import sys
import argparse
import logging
import json
from pathlib import Path
from dataclasses import dataclass


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

    parser = argparse.ArgumentParser("Script for generating json scenario files")
    #input dirs
    parser.add_argument("-c","--clean-data-dir", type=dir_path, required=True,
                        help="Clean data directory for scenarios (REQUIRED)")
    parser.add_argument("-n","--noisy-data-dir", type=dir_path, required=True,
                        help="Noisy data directory for scenarios (REQUIRED)")

    #output dir
    parser.add_argument("-o","--output-data-dir", type=dir_path_allow_new, required=True, 
                        help="Output directory for scenario (REQUIRED)")
    
    #values
    parser.add_argument("--snr", type=int, required=True, 
                        help="SNR value for the scenario (REQUIRED)")

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v, -vv)")

    args = parser.parse_args(argv)
    return args

# ---------- Config ----------

@dataclass(frozen=True)
class Config:


def make_config(args: argparse.Namespace) -> Config:
    level = logging.WARNING - 10*min(args.verbose, 2)
    return Config(


    )