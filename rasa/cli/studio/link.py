import argparse
from typing import List, Text

from rasa.cli import SubParsersAction
from rasa.cli.studio.upload import (
    add_config_param,
    add_data_param,
    add_domain_param,
    add_endpoint_param,
)
from rasa.shared.constants import (
    DEFAULT_DOMAIN_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.studio.link import handle_link


def add_subparser(
    subparsers: SubParsersAction,
    parents: List[argparse.ArgumentParser],
    domain: Text = DEFAULT_DOMAIN_PATH,
) -> None:
    """Register the `rasa studio link` command with the main CLI.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
        domain: Path to the assistant's domain file.
    """
    link_parser = subparsers.add_parser(
        "link",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Link the current project to an existing Rasa Studio assistant.",
    )

    link_parser.add_argument(
        "assistant_name",
        nargs=1,
        type=str,
        help="Name of the assistant in Rasa Studio.",
    )

    add_domain_param(link_parser, domain)
    add_data_param(link_parser)
    add_config_param(link_parser)
    add_endpoint_param(
        link_parser,
        "Configuration file for the model endpoints.",
        default=DEFAULT_ENDPOINTS_PATH,
    )
    link_parser.set_defaults(func=handle_link)
