import argparse
from typing import Union

from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_stories_param,
    add_nlu_data_param,
    add_out_param,
    add_domain_param,
)
from rasa.constants import DEFAULT_DATA_PATH, DEFAULT_CONFIG_PATH


def set_train_arguments(parser: argparse.ArgumentParser):
    add_data_param(parser)
    add_config_param(parser)
    add_domain_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_augmentation_param(parser)
    add_debug_plots_param(parser)

    add_model_name_param(parser)
    add_persist_nlu_data_param(parser)
    add_force_param(parser)


def set_train_core_arguments(parser: argparse.ArgumentParser):
    add_stories_param(parser)
    add_domain_param(parser)
    add_core_config_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_augmentation_param(parser)
    add_debug_plots_param(parser)

    add_force_param(parser)

    add_model_name_param(parser)

    compare_arguments = parser.add_argument_group("Comparison Arguments")
    add_compare_params(compare_arguments)


def set_train_nlu_arguments(parser: argparse.ArgumentParser):
    add_config_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_nlu_data_param(parser, help_text="File or folder containing your NLU data.")

    add_model_name_param(parser)
    add_persist_nlu_data_param(parser)


def add_force_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]):
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a model training even if the data has not changed.",
    )


def add_data_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]):
    parser.add_argument(
        "--data",
        default=[DEFAULT_DATA_PATH],
        nargs="+",
        help="Paths to the Core and NLU data files.",
    )


def add_core_config_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        default=[DEFAULT_CONFIG_PATH],
        help="The policy and NLU pipeline configuration of your bot. "
        "If multiple configuration files are provided, multiple Rasa Core "
        "models are trained to compare policies.",
    )


def add_compare_params(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
):
    parser.add_argument(
        "--percentages",
        nargs="*",
        type=int,
        default=[0, 25, 50, 75],
        help="Range of exclusion percentages.",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs for experiments."
    )


def add_augmentation_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
):
    parser.add_argument(
        "--augmentation",
        type=int,
        default=50,
        help="How much data augmentation to use during training.",
    )


def add_debug_plots_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
):
    parser.add_argument(
        "--debug-plots",
        default=False,
        action="store_true",
        help="If enabled, will create plots showing checkpoints "
        "and their connections between story blocks in a  "
        "file called `story_blocks_connections.html`.",
    )


def add_model_name_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--fixed-model-name",
        type=str,
        help="If set, the name of the model file/directory will be set to the given "
        "name.",
    )


def add_persist_nlu_data_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
):
    parser.add_argument(
        "--persist-nlu-data",
        action="store_true",
        help="Persist the nlu training data in the saved model.",
    )
