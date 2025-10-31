import argparse
import sys

from seadge.commandline import common
from seadge import config
from seadge.utils import scenario
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
    "scenario": scenario.dumpscenario
}

p_sub_config = parser_subcommands.add_parser(
    "config",
    help = "Dump current configuration as JSON",
    description = "Dump current configuration as JSON",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )

p_sub_scen = parser_subcommands.add_parser(
    "scenario",
    help = "Dump values for a scenario id",
    description = "Dump values for a scenario id",
    formatter_class = argparse.RawDescriptionHelpFormatter,
    )
p_sub_scen.add_argument("--room", action="store_true", help="Dump room file associated with scenario")
p_sub_scen.add_argument("--wav", action="store_true", help="Dump distant wav file associated with scenario")
p_sub_scen.add_argument("id", help="Scenario id")

def main(args):
    if not args.subcommand:
        p.print_usage()
        sys.exit()

    subcommands[args.subcommand](args)
