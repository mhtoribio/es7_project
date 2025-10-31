import argparse
import sys

from seadge.commandline import common
from seadge import config
from seadge.utils.log import log

p = common.parser_commands.add_parser(
    "dump",
    help = "Dump values (eg. configuration)",
    description = "Dump values (eg. configuration)",
    formatter_class = argparse.RawDescriptionHelpFormatter,
)

parser_subcommands = p.add_subparsers(dest='subcommand')

subcommands = {
    "config": config.dumpconfig,
}

p_sub_config = parser_subcommands.add_parser(
    "config",
    help = "Dump current configuration as JSON",
    description = "Dump current configuration as JSON",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )

def main(args):
    if not args.subcommand:
        p.print_usage()
        sys.exit()

    subcommands[args.subcommand]()
