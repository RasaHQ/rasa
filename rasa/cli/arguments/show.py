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
        "-o",
        "--output",
        default="graph.html",
        type=str,
        help="filename of the output path, e.g. 'graph.html",
    )
    parser.add_argument(
        "-m",
        "--max-history",
        default=2,
        type=int,
        help="max history to consider when merging paths in the output graph",
    )
    parser.add_argument(
        "-nlu",
        "--nlu-data",
        default=None,
        type=str,
        help="path of the Rasa NLU training data, "
        "used to insert example messages into the graph",
    )
