import argparse

from seadge import metric
from seadge.commandline import common
from seadge.utils.log import log

p = common.parser_commands.add_parser(
    "metric",
    help = "Compute metrics for algorithms in scenarios",
    description = "Compute metrics for algorithms in scenarios",
    formatter_class = argparse.RawDescriptionHelpFormatter,
)

def main(args):
    log.info("Running metrics")
    metric.main()
