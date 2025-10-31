from seadge.commandline import common
from seadge.commandline import datagen
from seadge.commandline import dump

commands = {
    "datagen": datagen.main,
    "dump": dump.main,
}

def main():
    common.entrypoint(commands)

if __name__ == '__main__':
    main()
