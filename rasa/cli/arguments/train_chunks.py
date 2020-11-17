import argparse

from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_out_param,
    add_domain_param,
)
from rasa.cli.arguments.train import (
    add_data_param,
    add_augmentation_param,
    add_num_threads_param,
    add_model_name_param,
    add_force_param,
)


def set_train_in_chunks_arguments(parser: argparse.ArgumentParser) -> None:
    """Set the command line arguments for the command 'rasa train-in-chunks'.

    Args:
        parser: the parser to add the arguments to
    """
    add_data_param(parser)
    add_config_param(parser)
    add_domain_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_augmentation_param(parser)

    add_num_threads_param(parser)

    add_model_name_param(parser)
    add_force_param(parser)


def set_train_in_chunks_core_arguments(parser: argparse.ArgumentParser):
    """Set the command line arguments for the command 'rasa train-in-chunks core'.

    Args:
        parser: the parser to add the arguments to
    """
    add_data_param(parser)
    add_config_param(parser)
    add_domain_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_augmentation_param(parser)

    add_force_param(parser)

    add_model_name_param(parser)


def set_train_in_chunks_nlu_arguments(parser: argparse.ArgumentParser):
    """Set the command line arguments for the command 'rasa train-in-chunks nlu'.

    Args:
        parser: the parser to add the arguments to
    """
    add_data_param(parser)
    add_config_param(parser)
    add_domain_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_num_threads_param(parser)

    add_model_name_param(parser)
