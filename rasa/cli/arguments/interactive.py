import argparse

from rasa.cli.arguments.default_arguments import (
    add_domain_param,
    add_stories_param,
    add_model_param,
    add_endpoint_param,
)
from rasa.cli.arguments.train import (
    add_force_param,
    add_data_param,
    add_config_param,
    add_out_param,
    add_debug_plots_param,
    add_dump_stories_param,
    add_augmentation_param,
    add_persist_nlu_data_param,
)


def set_interactive_arguments(parser: argparse.ArgumentParser):
    add_model_param(parser, default=None)
    add_data_param(parser)

    add_skip_visualization_param(parser)

    add_endpoint_param(
        parser,
        help_text="Configuration file for the model server and the connectors as a yml file.",
    )

    train_arguments = parser.add_argument_group("Train Arguments")
    add_config_param(train_arguments)
    add_domain_param(train_arguments)
    add_out_param(
        train_arguments, help_text="Directory where your models should be stored."
    )
    add_augmentation_param(train_arguments)
    add_debug_plots_param(train_arguments)
    add_dump_stories_param(train_arguments)
    add_force_param(train_arguments)
    add_persist_nlu_data_param(train_arguments)


def set_interactive_core_arguments(parser: argparse.ArgumentParser):
    add_model_param(parser, model_name="Rasa Core", default=None)
    add_stories_param(parser)

    add_skip_visualization_param(parser)

    add_endpoint_param(
        parser,
        help_text="Configuration file for the model server and the connectors as a yml file.",
    )

    train_arguments = parser.add_argument_group("Train Arguments")
    add_config_param(train_arguments)
    add_domain_param(train_arguments)
    add_out_param(
        train_arguments, help_text="Directory where your models should be stored."
    )
    add_augmentation_param(train_arguments)
    add_debug_plots_param(train_arguments)
    add_dump_stories_param(train_arguments)


def add_skip_visualization_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--skip-visualization",
        default=False,
        action="store_true",
        help="Disable plotting the visualization during interactive learning.",
    )
