"""Detection-only command line."""
import argparse
from .detection.cli import add_commands, run

def main(argv=None):
    parser = argparse.ArgumentParser(prog="rpad-detection", description="Détection d'événements sur une différence de phase entre deux PMU")
    add_commands(parser.add_subparsers(dest="command", required=True))
    run(parser.parse_args(argv))

if __name__ == "__main__":
    main()
