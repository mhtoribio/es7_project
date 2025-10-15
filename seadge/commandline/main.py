from seadge.commandline import common
from seadge.commandline import datagen

commands = {
    "datagen": datagen.main
}

def main():
    common.entrypoint(commands)

if __name__ == '__main__':
    main()
