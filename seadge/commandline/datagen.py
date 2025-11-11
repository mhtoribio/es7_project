import argparse
import pathlib
import sys

from seadge import data_download, room_generation, room_modelling, distant, scenario_generation, prepare_psd_tensors
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
    "download": data_download.main,
    "room": room_generation.main,
    "scenario": scenario_generation.main,
    "tensor": prepare_psd_tensors.main,
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

p_sub_rooms = parser_subcommands.add_parser(
    "room",
    help = "Generate rooms",
    description = "Generate rooms",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )

p_sub_scenario = parser_subcommands.add_parser(
    "scenario",
    help = "Generate scenarios",
    description = "Generate scenarios",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )

p_sub_tensor = parser_subcommands.add_parser(
    "tensor",
    help = "Generate tensors for ML training",
    description = "Generate tensors for ML training",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )

def main(args):
    if not args.subcommand:
        p.print_usage()
        sys.exit()

    log.info("Running datagen")
    subcommands[args.subcommand]()
