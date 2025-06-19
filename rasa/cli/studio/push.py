import argparse
from typing import List, Text

from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_data_param,
    add_domain_param,
    add_endpoint_param,
)
from rasa.shared.constants import (
    DEFAULT_DOMAIN_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.studio.push import (
    handle_push,
    handle_push_config,
    handle_push_endpoints,
)


def add_subparser(
    subparsers: SubParsersAction,
    parents: List[argparse.ArgumentParser],
    domain: Text = DEFAULT_DOMAIN_PATH,
) -> None:
    """Register the `rasa studio push` command and its sub-commands.

    Args:
        subparsers: The subparsers to add the command to.
        parents: Parent parsers, needed to ensure tree structure in argparse
        domain: Path to the assistant's domain file.
    """
    push_parser = subparsers.add_parser(
        "push",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Push assistant data to Rasa Studio.",
    )
    push_subparsers = push_parser.add_subparsers()

    # Arguments for pushing configuration only
    config_parser = push_subparsers.add_parser(
        "config",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Push only the assistant configuration file.",
    )
    add_config_param(config_parser)
    config_parser.set_defaults(func=handle_push_config)

    # Arguments for pushing endpoints only
    endpoints_parser = push_subparsers.add_parser(
        "endpoints",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Push ONLY the endpoints configuration file.",
    )
    add_endpoint_param(
        endpoints_parser,
        "Configuration file for the model endpoints.",
        default=DEFAULT_ENDPOINTS_PATH,
    )
    endpoints_parser.set_defaults(func=handle_push_endpoints)

    # Arguments for pushing the whole assistant data
    add_domain_param(push_parser, domain)
    add_data_param(push_parser)
    add_config_param(push_parser)
    add_endpoint_param(
        push_parser,
        "Configuration file for the model endpoints.",
        default=DEFAULT_ENDPOINTS_PATH,
    )
    push_parser.set_defaults(func=handle_push)
