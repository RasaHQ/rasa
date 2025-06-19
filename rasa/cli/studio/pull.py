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
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.studio.pull.pull import (
    handle_pull,
    handle_pull_config,
    handle_pull_endpoints,
)


def add_subparser(
    subparsers: SubParsersAction,
    parents: List[argparse.ArgumentParser],
    domain: Text = DEFAULT_DOMAIN_PATH,
) -> None:
    """Register `rasa studio pull` and its sub-commands.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
        domain: Path to the assistant's domain file.
    """
    pull_parser = subparsers.add_parser(
        "pull",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Pull assistant data from Rasa Studio.",
    )
    pull_subparsers = pull_parser.add_subparsers()

    # Arguments for pulling configuration only
    config_parser = pull_subparsers.add_parser(
        "config",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Pull only the assistant configuration file.",
    )
    add_config_param(config_parser, default=DEFAULT_CONFIG_PATH)
    config_parser.set_defaults(func=handle_pull_config)

    # Arguments for pulling endpoints only
    endpoints_parser = pull_subparsers.add_parser(
        "endpoints",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Pull only the endpoints configuration file.",
    )
    add_endpoint_param(
        endpoints_parser,
        "Configuration file for the model endpoints.",
        default=DEFAULT_ENDPOINTS_PATH,
    )
    endpoints_parser.set_defaults(func=handle_pull_endpoints)

    # Arguments for pulling the whole assistant data
    add_domain_param(pull_parser, domain)
    add_data_param(pull_parser)
    add_config_param(pull_parser, default=DEFAULT_CONFIG_PATH)
    add_endpoint_param(
        pull_parser,
        "Configuration file for the model endpoints.",
        default=DEFAULT_ENDPOINTS_PATH,
    )
    pull_parser.set_defaults(func=handle_pull)
