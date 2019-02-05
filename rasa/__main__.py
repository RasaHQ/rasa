# coding: utf8

import argparse
import logging

import rasa_core.utils
from rasa.cli import scaffold, run, train, configure

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
        prog="rasa",
        description='Rasa command line interface. Rasa allows you to build '
                    'your own conversational assistants. The `rasa` command '
                    'allows you to easily run most common commands like '
                    'creating a new bot, training or evaluating models.')

    subparsers = parser.add_subparsers(help='Rasa commands')

    scaffold.add_subparser(subparsers)
    run.add_subparser(subparsers)
    train.add_subparser(subparsers)
    configure.add_subparser(subparsers)

    rasa_core.utils.add_logging_option_arguments(parser)
    return parser


def main():
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()

    rasa_core.utils.configure_colored_logging(cmdline_arguments.loglevel)

    if hasattr(cmdline_arguments, "func"):
        cmdline_arguments.func(cmdline_arguments)
    else:
        # user has not provided a subcommand, let's print the help
        logger.error("No command specified.")
        arg_parser.print_help()
        exit(1)


if __name__ == '__main__':
    main()
