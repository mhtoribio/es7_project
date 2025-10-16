import argparse
import sys
import pathlib
from seadge import config

parser = argparse.ArgumentParser(description='SEADGE Command Line Interface',
                                 prog='seadge')

parser.add_argument("-c", "--config", type=pathlib.Path, help="Config file location (TOML, YAML, JSON)", required=True)
parser_commands = parser.add_subparsers(dest='command')

def entrypoint(commands):
    if len(sys.argv) < 4:
        parser.print_usage()
        sys.exit()
    args = parser.parse_args()
    config.load(path=args.config, create_dirs=True)
    commands[args.command](args)
