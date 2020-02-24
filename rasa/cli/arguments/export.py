import argparse

from rasa.cli.arguments import default_arguments
from rasa.constants import DEFAULT_ENDPOINTS_PATH


def set_export_arguments(parser: argparse.ArgumentParser) -> None:
    default_arguments.add_endpoint_param(
        parser,
        default=DEFAULT_ENDPOINTS_PATH,
        help_text=(
            "Endpoint configuration file specifying the tracker store "
            "and event broker."
        ),
    )

    parser.add_argument(
        "--minimum-timestamp",
        type=float,
        help=(
            "Minimum timestamp of events to be exported. The constraint is applied "
            "in a 'greater than or equal' comparison."
        ),
    )

    parser.add_argument(
        "--maximum-timestamp",
        type=float,
        help=(
            "Maximum timestamp of events to be exported. The constraint is "
            "applied in a 'less than' comparison."
        ),
    )

    parser.add_argument(
        "--conversation-ids",
        help=(
            "Comma-separated list of conversation IDs to migrate. If unset, "
            "all available conversation IDs will be exported."
        ),
    )
