import argparse
from typing import List, Text

from rasa.cli import SubParsersAction
from rasa.shared.constants import (
    DEFAULT_DOMAIN_PATH,
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
        type=str,
        help="Name of the assistant in Rasa Studio.",
    )
    link_parser.set_defaults(func=handle_link)
