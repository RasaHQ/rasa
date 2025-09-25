import argparse
from typing import Union

from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_domain_param,
    add_endpoint_param,
    add_nlu_data_param,
    add_out_param,
    add_remote_root_only_param,
    add_remote_storage_param,
    add_stories_param,
    add_sub_agents_param,
)
from rasa.graph_components.providers.training_tracker_provider import (
    TrainingTrackerProvider,
)
from rasa.shared.constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH

USE_LATEST_MODEL_FOR_FINE_TUNING = True


def set_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Specifies CLI arguments for `rasa train`."""
    add_data_param(parser)
    add_config_param(parser)
    add_domain_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_dry_run_param(parser)
    add_validate_before_train(parser)
    add_augmentation_param(parser)
    add_debug_plots_param(parser)

    _add_num_threads_param(parser)

    _add_model_name_param(parser)
    add_persist_nlu_data_param(parser)
    add_keep_local_model_copy_param(parser)
    add_force_param(parser)
    add_finetune_params(parser)
    add_endpoint_param(
        parser, help_text="Configuration file for the connectors as a yml file."
    )
    add_remote_storage_param(parser)
    add_remote_root_only_param(parser)
    add_sub_agents_param(parser)


def set_train_core_arguments(parser: argparse.ArgumentParser) -> None:
    """Specifies CLI arguments for `rasa train core`."""
    add_stories_param(parser)
    add_domain_param(parser)
    _add_core_config_param(parser)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_augmentation_param(parser)
    add_debug_plots_param(parser)

    add_force_param(parser)

    _add_model_name_param(parser)

    compare_arguments = parser.add_argument_group("Comparison Arguments")
    _add_compare_params(compare_arguments)
    add_finetune_params(parser)


def set_train_nlu_arguments(parser: argparse.ArgumentParser) -> None:
    """Specifies CLI arguments for `rasa train nlu`."""
    add_config_param(parser)
    add_domain_param(parser, default=None)
    add_out_param(parser, help_text="Directory where your models should be stored.")

    add_nlu_data_param(parser, help_text="File or folder containing your NLU data.")

    _add_num_threads_param(parser)

    _add_model_name_param(parser)
    add_persist_nlu_data_param(parser)
    add_finetune_params(parser)


def add_force_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Specifies if the model should be trained from scratch."""
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a model training even if the data has not changed.",
    )


def add_data_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Specifies path to training data."""
    parser.add_argument(
        "--data",
        default=[DEFAULT_DATA_PATH],
        nargs="+",
        help="Paths to the Core and NLU data files.",
    )


def _add_core_config_param(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        default=[DEFAULT_CONFIG_PATH],
        help="The policy and NLU pipeline configuration of your bot. "
        "If multiple configuration files are provided, multiple Rasa Core "
        "models are trained to compare policies.",
    )


def _add_compare_params(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
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


def add_dry_run_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Adds `--dry-run` argument to a specified `parser`.

    Args:
        parser: An instance of `ArgumentParser` or `_ActionsContainer`.
    """
    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="If enabled, no actual training will be performed. Instead, "
        "it will be determined whether a model should be re-trained "
        "and this information will be printed as the output. The return "
        "code is a 4-bit bitmask that can also be used to determine what exactly needs "
        "to be retrained:\n"
        "- 0 means that no extensive training is required (note that the responses "
        "still might require updating by running 'rasa train').\n"
        "- 1 means the model needs to be retrained\n"
        "- 8 means the training was forced (--force argument is specified)",
    )


def add_validate_before_train(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Adds parameters for validating the domain and data files before training.

    Args:
        parser: An instance of `ArgumentParser` or `_ActionsContainer`.
    """
    parser.add_argument(
        "--skip-validation",
        default=False,
        action="store_true",
        help="Skip validation step before training.",
    )

    parser.add_argument(
        "--fail-on-validation-warnings",
        default=False,
        action="store_true",
        help="Fail on validation warnings. "
        "If omitted only errors will exit with a non zero status code",
    )

    parser.add_argument(
        "--validation-max-history",
        type=int,
        default=None,
        help="Number of turns taken into account for story structure validation.",
    )


def add_augmentation_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Sets the augmentation factor for the Core training.

    Args:
        parser: An instance of `ArgumentParser` or `_ActionsContainer`.
    """
    parser.add_argument(
        "--augmentation",
        type=int,
        default=TrainingTrackerProvider.get_default_config()["augmentation_factor"],
        help="How much data augmentation to use during training.",
    )


def add_debug_plots_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Specifies if conversation flow should be visualized."""
    parser.add_argument(
        "--debug-plots",
        default=TrainingTrackerProvider.get_default_config()["debug_plots"],
        action="store_true",
        help="If enabled, will create plots showing checkpoints "
        "and their connections between story blocks in a  "
        "file called `story_blocks_connections.html`.",
    )


def _add_num_threads_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    parser.add_argument(
        "--num-threads",
        type=int,
        help="Maximum amount of threads to use when training.",
    )


def _add_model_name_param(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fixed-model-name",
        type=str,
        help="If set, the name of the model file/directory will be set to the given "
        "name.",
    )


def add_persist_nlu_data_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Adds parameters for persisting the NLU training data with the model."""
    parser.add_argument(
        "--persist-nlu-data",
        action="store_true",
        help="Persist the NLU training data in the saved model.",
    )


def add_keep_local_model_copy_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Adds parameters for keeping a local copy of the model."""
    parser.add_argument(
        "--keep-local-model-copy",
        action="store_true",
        help="Keep a copy of the model in the model directory if remote "
        "model upload is configured. Defaults to `false`, which "
        "deletes the local copy of the model after upload.",
    )


def add_finetune_params(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    """Adds parameters for model finetuning."""
    parser.add_argument(
        "--finetune",
        nargs="?",
        # If the user doesn't specify `--finetune` at all
        default=None,
        # If the user only specifies `--finetune` without an additional path
        const=USE_LATEST_MODEL_FOR_FINE_TUNING,
        help="Fine-tune a previously trained model. If no model path is provided, Rasa "
        "Open Source will try to finetune the latest trained model from the "
        "model directory specified via '--out'.",
    )

    parser.add_argument(
        "--epoch-fraction",
        type=float,
        help="Fraction of epochs which are currently specified in the model "
        "configuration which should be used when finetuning a model.",
    )
