import argparse

from seadge import enhance
from seadge.commandline import common
from seadge.utils.log import log

p = common.parser_commands.add_parser(
    "enhance",
    help = "Enhance generated scenarios",
    description = "Enhance generated scenarios",
    formatter_class = argparse.RawDescriptionHelpFormatter,
)

def main(args):
    log.info("Running enhancement")
    enhance.main()
