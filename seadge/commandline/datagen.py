import argparse
import pathlib

from seadge.commandline import common

p = common.parser_commands.add_parser(
    "datagen",
    help = "Generating data for training and evaluation.\n\nExample: TODO",
    description = "Generating data for training and evaluation.\n\nExample: TODO",
    formatter_class = argparse.RawDescriptionHelpFormatter,
)

def main(args):
    print("Running datagen")
