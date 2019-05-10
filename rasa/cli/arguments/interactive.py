import argparse

from rasa.cli.arguments.run import add_server_arguments
from rasa.cli.arguments.default_arguments import (
    add_domain_param,
    add_stories_param,
    add_model_param,
)
from rasa.cli.arguments.train import (
    add_force_param,
    add_data_param,
    add_config_param,
    add_out_param,
    add_debug_plots_param,
    add_dump_stories_param,
    add_augmentation_param,
)


def set_interactive_arguments(parser: argparse.ArgumentParser):
    add_model_param(parser)
    add_config_param(parser)
    add_domain_param(parser)
    add_data_param(parser)
    add_out_param(parser)

    add_force_param(parser)

    add_skip_visualization_param(parser)

    add_server_arguments(parser)


def set_interactive_core_arguments(parser: argparse.ArgumentParser):
    add_model_param(parser)
    add_config_param(parser)
    add_domain_param(parser)
    add_stories_param(parser)
    add_out_param(parser)

    add_augmentation_param(parser)
    add_debug_plots_param(parser)
    add_dump_stories_param(parser)

    add_skip_visualization_param(parser)

    add_server_arguments(parser)


def add_skip_visualization_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--skip-visualization",
        default=False,
        action="store_true",
        help="Disables plotting the visualization during interactive learning.",
    )
