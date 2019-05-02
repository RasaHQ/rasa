import argparse
import logging

import rasa.utils.io

from rasa.core.cli.arguments import add_logging_option_arguments
from rasa import version
from rasa.cli import scaffold, run, train, interactive, shell, test, show, data, x
from rasa.cli.utils import parse_last_positional_argument_as_model_path

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
        prog="rasa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rasa command line interface. Rasa allows you to build "
        "your own conversational assistants 🤖. The 'rasa' command "
        "allows you to easily run most common commands like "
        "creating a new bot, training or evaluating models.",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print installed Rasa version",
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    add_logging_option_arguments(parent_parser)
    parent_parsers = [parent_parser]

    subparsers = parser.add_subparsers(help="Rasa commands")

    scaffold.add_subparser(subparsers, parents=parent_parsers)
    run.add_subparser(subparsers, parents=parent_parsers)
    shell.add_subparser(subparsers, parents=parent_parsers)
    train.add_subparser(subparsers, parents=parent_parsers)
    interactive.add_subparser(subparsers, parents=parent_parsers)
    test.add_subparser(subparsers, parents=parent_parsers)
    show.add_subparser(subparsers, parents=parent_parsers)
    data.add_subparser(subparsers, parents=parent_parsers)
    x.add_subparser(subparsers, parents=parent_parsers)

    return parser


def print_version() -> None:
    print ("Rasa", version.__version__)


def main() -> None:
    # Running as standalone python application
    parse_last_positional_argument_as_model_path()
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()

    if hasattr(cmdline_arguments, "func"):
        rasa.utils.io.configure_colored_logging(cmdline_arguments.loglevel)
        cmdline_arguments.func(cmdline_arguments)
    elif hasattr(cmdline_arguments, "version"):
        print_version()
    else:
        # user has not provided a subcommand, let's print the help
        logger.error("No command specified.")
        arg_parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
