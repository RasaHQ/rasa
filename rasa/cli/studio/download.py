import argparse
from typing import List

from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_endpoint_param,
)
from rasa.cli.arguments.train import add_data_param, add_domain_param
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH, DEFAULT_CONFIG_PATH

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
    add_domain_param(parser)
    add_data_param(parser)
    add_config_param(parser, default=DEFAULT_CONFIG_PATH)
    add_endpoint_param(
        parser,
        "Configuration file for the model endpoints.",
        default=DEFAULT_ENDPOINTS_PATH,
    )

    parser.add_argument(
        "assistant_name",
        default=None,
        nargs=1,
        type=str,
        help="Name of the assistant on Rasa Studio",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local data with data from Rasa Studio",
    )
