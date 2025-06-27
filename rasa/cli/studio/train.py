import argparse
from typing import List

from rasa.cli import SubParsersAction
from rasa.cli.arguments.train import set_train_arguments
from rasa.studio.train import handle_train


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add the studio train parser.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    train_parser = subparsers.add_parser(
        "train",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Trains a Rasa model using Rasa Studio data.",
    )

    train_parser.set_defaults(func=handle_train)
    set_train_arguments(train_parser)
    set_studio_train_arguments(train_parser)


def set_studio_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for running `rasa studio train`."""
    parser.add_argument(
        "assistant_name",
        default=None,
        type=str,
        help="Name of the assistant on Rasa Studio",
    )

    parser.add_argument(
        "--entities",
        default=None,
        nargs="+",
        type=str,
        help="Name of entities to download from Rasa Studio",
    )

    parser.add_argument(
        "--intents",
        default=None,
        nargs="+",
        type=str,
        help="Name of intents to download from Rasa Studio",
    )
