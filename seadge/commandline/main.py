from seadge.commandline import common
from seadge.commandline import datagen
from seadge.commandline import dump
from seadge.commandline import train
from seadge.commandline import enhance
from seadge.commandline import clean
from seadge.commandline import metric

commands = {
    "datagen": datagen.main,
    "dump": dump.main,
    "train": train.main,
    "enhance": enhance.main,
    "clean": clean.main,
    "metric": metric.main,
}

def main():
    common.entrypoint(commands)

if __name__ == '__main__':
    main()
