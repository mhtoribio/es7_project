import argparse
import pathlib
import sys

from seadge import train_psd_model
from seadge.commandline import common
from seadge.utils.log import log

p = common.parser_commands.add_parser(
    "train",
    help = "Train models",
    description = "Train models",
    formatter_class = argparse.RawDescriptionHelpFormatter,
)

def main(args):
    train_psd_model.main()
