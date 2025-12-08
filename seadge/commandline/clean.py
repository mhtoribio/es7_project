import os
from pathlib import Path
import argparse
import shutil

from seadge.commandline import common
from seadge import config
from seadge.utils.log import log

def _rm(p: Path):
    log.debug(f"Removing {p}")
    shutil.rmtree(p)

p = common.parser_commands.add_parser(
    "clean",
    help = "Run cleanup to start from scratch",
    description = "Run cleanup to start from scratch",
    formatter_class = argparse.RawDescriptionHelpFormatter,
)

def main(args):
    cfg = config.get()

    log.info(f"SEADGE cleanup (rooms, scenarios, generated audio, ml checkpoints, etc.)")
    _rm(cfg.paths.output_dir)
