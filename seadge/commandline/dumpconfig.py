import argparse

from seadge.commandline import common
from seadge import config

p = common.parser_commands.add_parser(
    "dumpconfig",
    help = "Dump current configuration",
    description = "Dump current configuration",
    formatter_class = argparse.RawDescriptionHelpFormatter,
)

def main(args):
    print("Current config")
    from pprint import pprint
    pprint(config.as_dict())
