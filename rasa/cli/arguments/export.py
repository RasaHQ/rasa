import argparse

from rasa.cli.arguments.default_arguments import add_endpoint_param


def set_export_arguments(parser: argparse.ArgumentParser) -> None:
    add_endpoint_param(
        parser,
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
