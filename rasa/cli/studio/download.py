import argparse
from typing import List

from rasa.cli import SubParsersAction
from rasa.studio.download import handle_download


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add the studio download parser.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    download_parser = subparsers.add_parser(
        "download",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=(
            "Download data from Rasa Studio including "
            "flows and NLU data depending on the assistant type."
        ),
    )

    download_parser.set_defaults(func=handle_download)
    set_studio_download_arguments(download_parser)


def set_studio_download_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for running `rasa studio download`."""
    parser.add_argument(
        "assistant_name",
        default=None,
        type=str,
        help="Name of the assistant on Rasa Studio",
    )
