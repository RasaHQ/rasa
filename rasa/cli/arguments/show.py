import argparse

from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_domain_param,
    add_stories_param,
)


def set_show_stories_arguments(parser: argparse.ArgumentParser):
    add_domain_param(parser)
    add_stories_param(parser)
    add_config_param(parser)

    add_visualization_arguments(parser)


def add_visualization_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--output",
        default="graph.html",
        type=str,
        help="Filename of the output path, e.g. 'graph.html'.",
    )
    parser.add_argument(
        "--max-history",
        default=2,
        type=int,
        help="Max history to consider when merging paths in the output graph.",
    )
    parser.add_argument(
        "-nlu",
        "--nlu-data",
        default=None,
        type=str,
        help="Path of the Rasa NLU training data, "
        "used to insert example messages into the graph.",
    )
