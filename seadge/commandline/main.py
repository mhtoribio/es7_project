from seadge.commandline import common
from seadge.commandline import datagen
from seadge.commandline import dump
from seadge.commandline import train

commands = {
    "datagen": datagen.main,
    "dump": dump.main,
    "train": train.main
}

def main():
    common.entrypoint(commands)

if __name__ == '__main__':
    main()
