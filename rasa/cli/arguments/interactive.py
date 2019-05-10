import argparse

from rasa.cli.arguments.run import add_run_arguments
from rasa.cli.arguments.default_arguments import add_domain_param, add_stories_param
from rasa.cli.arguments.train import (
    add_force_param,
    add_data_param,
    add_config_param,
    add_out_param,
)


def set_interactive_args(parser):
    add_run_arguments(parser)
    add_config_param(parser)
    add_out_param(parser)
    add_domain_param(parser)
    add_data_param(parser)
    add_force_param(parser)
    add_skip_visualization_param(parser)


def set_interactive_core_args(parser):
    add_config_param(parser)
    add_out_param(parser)
    add_domain_param(parser)
    add_stories_param(parser)
    add_domain_param(parser)
    add_run_arguments(parser)
    add_skip_visualization_param(parser)


def add_skip_visualization_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--skip-visualization",
        default=False,
        action="store_true",
        help="Disables plotting the visualization during interactive learning",
    )
