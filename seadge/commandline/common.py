import argparse
import sys

parser = argparse.ArgumentParser(description='SEADGE Command Line Interface',
                                 prog='seadge')
parser_commands = parser.add_subparsers(dest='command')

def entrypoint(commands):
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit()
    args = parser.parse_args()
    commands[args.command](args)
