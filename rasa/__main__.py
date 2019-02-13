# coding: utf8

import argparse
import logging

from rasa import version
from rasa.cli import (scaffold, run, train, configure, interactive, shell, test,
                      show, data)
from rasa_core.cli.arguments import add_logging_option_arguments
from rasa_core.utils import configure_colored_logging

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the training script."""
    parser = argparse.ArgumentParser(
        prog="rasa",
        description='Rasa command line interface. Rasa allows you to build '
                    'your own conversational assistants. The `rasa` command '
                    'allows you to easily run most common commands like '
                    'creating a new bot, training or evaluating models.')

    parser.add_argument("--version", action='store_true',
                        help="Print installed Rasa version.")

    parent_parser = argparse.ArgumentParser(add_help=False)
    add_logging_option_arguments(parent_parser)
    parent_parsers = [parent_parser]

    subparsers = parser.add_subparsers(help='Rasa commands')

    scaffold.add_subparser(subparsers, parents=parent_parsers)
    run.add_subparser(subparsers, parents=parent_parsers)
    shell.add_subparser(subparsers, parents=parent_parsers)
    train.add_subparser(subparsers, parents=parent_parsers)
    #configure.add_subparser(subparsers)
    interactive.add_subparser(subparsers, parents=parent_parsers)
    test.add_subparser(subparsers, parents=parent_parsers)
    show.add_subparser(subparsers, parents=parent_parsers)
    data.add_subparser(subparsers, parents=parent_parsers)

    return parser


def print_version():
    print("Rasa", version.__version__)


def main():
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()

    if hasattr(cmdline_arguments, "func"):
        configure_colored_logging(cmdline_arguments.loglevel)
        cmdline_arguments.func(cmdline_arguments)
    elif cmdline_arguments.version:
        print_version()
    else:
        # user has not provided a subcommand, let's print the help
        logger.error("No command specified.")
        arg_parser.print_help()
        exit(1)


if __name__ == '__main__':
    main()
