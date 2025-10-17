from seadge.commandline import common
from seadge.commandline import datagen
from seadge.commandline import dumpconfig

commands = {
    "datagen": datagen.main,
    "dumpconfig": dumpconfig.main
}

def main():
    common.entrypoint(commands)

if __name__ == '__main__':
    main()
