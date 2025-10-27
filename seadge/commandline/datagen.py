import argparse
import pathlib
import sys

from seadge import data_download, room_modelling, distant
from seadge.commandline import common
from seadge.utils.log import log

p = common.parser_commands.add_parser(
    "datagen",
    help = "Generating data for training and evaluation.\n\nExample: TODO",
    description = "Generating data for training and evaluation.\n\nExample: TODO",
    formatter_class = argparse.RawDescriptionHelpFormatter,
)

parser_subcommands = p.add_subparsers(dest='subcommand')

subcommands = {
    "rir": room_modelling.main,
    "distant": distant.main,
    "download": data_download.main
}

p_sub_rir = parser_subcommands.add_parser(
    "rir",
    help = "Generating Room Impulse Responses (RIR) based on config.",
    description = "Generating Room Impulse Responses (RIR) based on config.",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )

p_sub_distant = parser_subcommands.add_parser(
    "distant",
    help = "Generate distant speech based on Room Impulse Responses and clean speech",
    description = "Generate distant speech based on Room Impulse Responses and clean speech",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )

p_sub_download = parser_subcommands.add_parser(
    "download",
    help = "Download data from DNS Challenge",
    description = "Download data from DNS Challenge",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )

def main(args):
    if not args.subcommand:
        p.print_usage()
        sys.exit()

    log.info("Running datagen")
    subcommands[args.subcommand]()
