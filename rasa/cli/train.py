import argparse
import logging
import sys
from typing import Dict, List, Optional, Text

from rasa.cli import SubParsersAction
import rasa.cli.arguments.train as train_arguments

import rasa.cli.utils
from rasa.shared.importers.importer import TrainingDataImporter
import rasa.utils.common
from rasa.core.train import do_compare_training
from rasa.plugin import plugin_manager
from rasa.shared.constants import (
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS_NLU,
    CONFIG_MANDATORY_KEYS,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
)

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all training parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    train_parser = subparsers.add_parser(
        "train",
        help="Trains a Rasa model using your NLU data and stories.",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    train_arguments.set_train_arguments(train_parser)

    train_subparsers = train_parser.add_subparsers()
    train_core_parser = train_subparsers.add_parser(
        "core",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Trains a Rasa Core model using your stories.",
    )
    train_core_parser.set_defaults(func=run_core_training)

    train_nlu_parser = train_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Trains a Rasa NLU model using your NLU data.",
    )
    train_nlu_parser.set_defaults(func=run_nlu_training)

    train_parser.set_defaults(func=lambda args: run_training(args, can_exit=True))

    train_arguments.set_train_core_arguments(train_core_parser)
    train_arguments.set_train_nlu_arguments(train_nlu_parser)


def run_training(args: argparse.Namespace, can_exit: bool = False) -> Optional[Text]:
    """Trains a model.

    Args:
        args: Namespace arguments.
        can_exit: If `True`, the operation can send `sys.exit` in the case
            training was not successful.

    Returns:
        Path to a trained model or `None` if training was not successful.
    """
    from rasa import train as train_all

    domain = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )
    config = rasa.cli.utils.get_validated_config(args.config, CONFIG_MANDATORY_KEYS)

    training_files = [
        rasa.cli.utils.get_validated_path(
            f, "data", DEFAULT_DATA_PATH, none_is_valid=True
        )
        for f in args.data
    ]

    if not args.skip_validation:
        logger.info("Started validating domain and training data...")
        importer = TrainingDataImporter.load_from_config(
            domain_path=args.domain, training_data_paths=args.data, config_path=config
        )
        rasa.cli.utils.validate_files(
            args.fail_on_validation_warnings, args.validation_max_history, importer
        )

    training_result = train_all(
        domain=domain,
        config=config,
        training_files=training_files,
        output=args.out,
        dry_run=args.dry_run,
        force_training=args.force,
        fixed_model_name=args.fixed_model_name,
        persist_nlu_training_data=args.persist_nlu_data,
        core_additional_arguments={
            **extract_core_additional_arguments(args),
            **_extract_additional_arguments(args),
        },
        nlu_additional_arguments=extract_nlu_additional_arguments(args),
        model_to_finetune=_model_for_finetuning(args),
        finetuning_epoch_fraction=args.epoch_fraction,
    )
    if training_result.code != 0 and can_exit:
        sys.exit(training_result.code)

    return training_result.model


def _model_for_finetuning(args: argparse.Namespace) -> Optional[Text]:
    if args.finetune == train_arguments.USE_LATEST_MODEL_FOR_FINE_TUNING:
        # We use this constant to signal that the user specified `--finetune` but
        # didn't provide a path to a model. In this case we try to load the latest
        # model from the output directory (that's usually models/).
        return args.out
    else:
        return args.finetune


def run_core_training(args: argparse.Namespace) -> Optional[Text]:
    """Trains a Rasa Core model only.

    Args:
        args: Command-line arguments to configure training.

    Returns:
        Path to a trained model or `None` if training was not successful.
    """
    from rasa.model_training import train_core

    args.domain = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )
    story_file = rasa.cli.utils.get_validated_path(
        args.stories, "stories", DEFAULT_DATA_PATH, none_is_valid=True
    )
    additional_arguments = {
        **extract_core_additional_arguments(args),
        **_extract_additional_arguments(args),
    }

    # Policies might be a list for the compare training. Do normal training
    # if only list item was passed.
    if not isinstance(args.config, list) or len(args.config) == 1:
        if isinstance(args.config, list):
            args.config = args.config[0]

        config = rasa.cli.utils.get_validated_config(
            args.config, CONFIG_MANDATORY_KEYS_CORE
        )

        return train_core(
            domain=args.domain,
            config=config,
            stories=story_file,
            output=args.out,
            fixed_model_name=args.fixed_model_name,
            additional_arguments=additional_arguments,
            model_to_finetune=_model_for_finetuning(args),
            finetuning_epoch_fraction=args.epoch_fraction,
        )
    else:
        do_compare_training(args, story_file, additional_arguments)
        return None


def run_nlu_training(args: argparse.Namespace) -> Optional[Text]:
    """Trains an NLU model.

    Args:
        args: Namespace arguments.

    Returns:
        Path to a trained model or `None` if training was not successful.
    """
    from rasa.model_training import train_nlu

    config = rasa.cli.utils.get_validated_config(args.config, CONFIG_MANDATORY_KEYS_NLU)
    nlu_data = rasa.cli.utils.get_validated_path(
        args.nlu, "nlu", DEFAULT_DATA_PATH, none_is_valid=True
    )

    if args.domain:
        args.domain = rasa.cli.utils.get_validated_path(
            args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
        )

    return train_nlu(
        config=config,
        nlu_data=nlu_data,
        output=args.out,
        fixed_model_name=args.fixed_model_name,
        persist_nlu_training_data=args.persist_nlu_data,
        additional_arguments={
            **extract_nlu_additional_arguments(args),
            **_extract_additional_arguments(args),
        },
        domain=args.domain,
        model_to_finetune=_model_for_finetuning(args),
        finetuning_epoch_fraction=args.epoch_fraction,
    )


def extract_core_additional_arguments(args: argparse.Namespace) -> Dict:
    arguments = {}

    if "augmentation" in args:
        arguments["augmentation_factor"] = args.augmentation
    if "debug_plots" in args:
        arguments["debug_plots"] = args.debug_plots

    return arguments


def extract_nlu_additional_arguments(args: argparse.Namespace) -> Dict:
    arguments = {}

    if "num_threads" in args:
        arguments["num_threads"] = args.num_threads

    return arguments


def _extract_additional_arguments(args: argparse.Namespace) -> Dict:
    space = plugin_manager().hook.handle_space_args(args=args)
    return space or {}
