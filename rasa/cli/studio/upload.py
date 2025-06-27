import argparse
from typing import List

from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_data_param,
    add_domain_param,
    add_endpoint_param,
)
from rasa.studio.upload import handle_upload


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add the upload parser.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    upload_parser = subparsers.add_parser(
        "upload",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Upload primitives to Rasa Studio.",
    )

    upload_parser.set_defaults(func=handle_upload)
    set_upload_arguments(upload_parser)


def set_upload_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for running `rasa upload`."""
    add_data_param(parser, data_type="training")
    add_domain_param(parser)
    add_config_param(parser)
    add_endpoint_param(parser, help_text="Path to the endpoints file.")

    parser.add_argument(
        "--entities",
        default=None,
        nargs="+",
        type=str,
        help="Name of entities to upload to Rasa Studio",
    )

    parser.add_argument(
        "--intents",
        default=None,
        nargs="+",
        type=str,
        help="Name of intents to upload to Rasa Studio",
    )

    parser.add_argument(
        "assistant_name",
        default=None,
        nargs="?",
        type=str,
        help="Name of the assistant on Rasa Studio",
    )
