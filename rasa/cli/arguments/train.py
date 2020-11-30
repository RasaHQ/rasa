import argparse
from typing import Union

import rasa.cli.arguments.default_arguments as default_arguments
from rasa.shared.constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH


def set_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets the command line arguments for the command 'rasa train'.

    Args:
        parser: the parser to add the arguments to
    """
    add_data_param(parser)
    default_arguments.add_config_param(parser)
    default_arguments.add_domain_param(parser)
    default_arguments.add_out_param(
        parser, help_text="Directory where your models should be stored."
    )

    add_augmentation_param(parser)
    add_debug_plots_param(parser)

    add_num_threads_param(parser)

    add_model_name_param(parser)
    add_persist_nlu_data_param(parser)
    add_force_param(parser)

    add_arguments_for_training_in_chunks(parser)


def set_train_core_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets the command line arguments for the command 'rasa train core'.

    Args:
        parser: the parser to add the arguments to
    """
    default_arguments.add_stories_param(parser)
    default_arguments.add_domain_param(parser)
    add_core_config_param(parser)
    default_arguments.add_out_param(
        parser, help_text="Directory where your models should be stored."
    )

    add_augmentation_param(parser)
    add_debug_plots_param(parser)

    add_force_param(parser)

    add_model_name_param(parser)

    compare_arguments = parser.add_argument_group("Comparison Arguments")
    add_compare_params(compare_arguments)

    add_arguments_for_training_in_chunks(parser)


def set_train_nlu_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets the command line arguments for the command 'rasa train nlu'.

    Args:
        parser: the parser to add the arguments to
    """
    default_arguments.add_config_param(parser)
    default_arguments.add_domain_param(parser, default=None)
    default_arguments.add_out_param(
        parser, help_text="Directory where your models should be stored."
    )

    default_arguments.add_nlu_data_param(
        parser, help_text="File or folder containing your NLU data."
    )

    add_num_threads_param(parser)

    add_model_name_param(parser)
    add_persist_nlu_data_param(parser)

    add_arguments_for_training_in_chunks(parser)


def add_arguments_for_training_in_chunks(parser: argparse.ArgumentParser) -> None:
    """Adds the arguments for training in chunks.

    Args:
        parser: the parser to add the arguments to
    """
    train_in_chunks_arguments = parser.add_argument_group(
        "Arguments for training in chunks"
    )

    train_in_chunks_arguments.add_argument(
        "--number-of-chunks",
        type=int,
        default=1,
        help="Number of chunks to use. By default the complete dataset is used at "
        "once.",
    )


def add_force_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
) -> None:
    """Adds the parameter '--force'.

    Args:
        parser: the parser to add the arguments to
    """
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a model training even if the data has not changed.",
    )


def add_data_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
) -> None:
    """Adds the parameter '--data'.

    Args:
        parser: the parser to add the arguments to
    """
    parser.add_argument(
        "--data",
        default=[DEFAULT_DATA_PATH],
        nargs="+",
        help="Paths to the Core and NLU data files.",
    )


def add_core_config_param(parser: argparse.ArgumentParser) -> None:
    """Adds the parameter '--config'.

    Args:
        parser: the parser to add the arguments to
    """
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
) -> None:
    """Adds the parameter '--percentages'.

    Args:
        parser: the parser to add the arguments to
    """
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
) -> None:
    """Adds the parameter '--augmentation'.

    Args:
        parser: the parser to add the arguments to
    """
    parser.add_argument(
        "--augmentation",
        type=int,
        default=50,
        help="How much data augmentation to use during training.",
    )


def add_debug_plots_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
) -> None:
    """Adds the parameter '--debug-plots'.

    Args:
        parser: the parser to add the arguments to
    """
    parser.add_argument(
        "--debug-plots",
        default=False,
        action="store_true",
        help="If enabled, will create plots showing checkpoints "
        "and their connections between story blocks in a  "
        "file called `story_blocks_connections.html`.",
    )


def add_num_threads_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
) -> None:
    """Adds the parameter '--num-threads'.

    Args:
        parser: the parser to add the arguments to
    """
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Maximum amount of threads to use when training.",
    )


def add_model_name_param(parser: argparse.ArgumentParser) -> None:
    """Adds the parameter '--fixed-model-name'.

    Args:
        parser: the parser to add the arguments to
    """
    parser.add_argument(
        "--fixed-model-name",
        type=str,
        help="If set, the name of the model file/directory will be set to the given "
        "name.",
    )


def add_persist_nlu_data_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
) -> None:
    """Adds the parameter '--persist-nlu-data'.

    Args:
        parser: the parser to add the arguments to
    """
    parser.add_argument(
        "--persist-nlu-data",
        action="store_true",
        help="Persist the nlu training data in the saved model.",
    )
