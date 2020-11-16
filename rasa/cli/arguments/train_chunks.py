import argparse
from typing import Union

from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_out_param,
    add_domain_param,
)
from rasa.shared.constants import DEFAULT_DATA_PATH


def set_train_chunk_arguments(parser: argparse.ArgumentParser):
    add_data_param(parser)
    add_config_param(parser)
    add_domain_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_augmentation_param(parser)

    add_num_threads_param(parser)

    add_model_name_param(parser)
    add_force_param(parser)


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


def add_augmentation_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
):
    parser.add_argument(
        "--augmentation",
        type=int,
        default=0,
        help="How much data augmentation to use during training.",
    )


def add_num_threads_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
):
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Maximum amount of threads to use when training.",
    )


def add_model_name_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--fixed-model-name",
        type=str,
        help="If set, the name of the model file/directory will be set to the given "
        "name.",
    )
