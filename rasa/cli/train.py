import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Text, Union

import structlog

import rasa.cli.arguments.train as train_arguments
from rasa.cli import SubParsersAction
from rasa.cli.validation.bot_config import validate_files
from rasa.cli.validation.config_path_validation import (
    get_validated_config,
    get_validated_path,
)
from rasa.core.config.configuration import Configuration
from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser
from rasa.core.nlg.generator import NaturalLanguageGenerator
from rasa.core.train import do_compare_training
from rasa.engine.validation import validate_api_type_config_key_usage
from rasa.exceptions import DetailedRasaException
from rasa.shared.constants import (
    CONFIG_MANDATORY_KEYS,
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS_NLU,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATHS,
    LLM_CONFIG_KEY,
)
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.common import display_research_study_prompt

structlogger = structlog.getLogger(__name__)


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


def _check_nlg_endpoint_validity(endpoint: Union[Path, str]) -> None:
    try:
        endpoints = Configuration.initialise_endpoints(
            endpoints_path=endpoint
        ).endpoints
        if endpoints.nlg is not None:
            validate_api_type_config_key_usage(
                endpoints.nlg.kwargs,
                LLM_CONFIG_KEY,
                ContextualResponseRephraser.__name__,
            )
        NaturalLanguageGenerator.create(endpoints.nlg)
    except DetailedRasaException as e:
        structlogger.error(
            e.code,
            event_info=e.info,
            **e.ctx,
        )
        sys.exit(1)
    except Exception as e:
        structlogger.error(
            "cli.train.nlg_failed_to_initialise.validation_error",
            exception=f"{e}",
            event_info=(
                f"The validation failed for NLG configuration defined in "
                f"{endpoint}. Please make sure the NLG configuration is correct."
            ),
        )
        display_research_study_prompt()
        sys.exit(1)


def run_training(args: argparse.Namespace, can_exit: bool = False) -> Optional[Text]:
    """Trains a model.

    Args:
        args: Namespace arguments.
        can_exit: If `True`, the operation can send `sys.exit` in the case
            training was not successful.

    Returns:
        Path to a trained model or `None` if training was not successful.
    """
    from rasa.api import train as train_all

    domain = get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATHS, none_is_valid=True
    )
    config = get_validated_config(args.config, CONFIG_MANDATORY_KEYS)

    # Validates and loads endpoints with proper endpoint file location
    # TODO(Radovan): this should be probably be done in Configuration
    _check_nlg_endpoint_validity(args.endpoints)
    Configuration.initialise_sub_agents(args.sub_agents)

    training_files = [
        get_validated_path(f, "data", DEFAULT_DATA_PATH, none_is_valid=True)
        for f in args.data
    ]

    training_data_importer = TrainingDataImporter.load_from_config(
        domain_path=domain, training_data_paths=args.data, config_path=config
    )

    if not args.skip_validation:
        structlogger.info(
            "cli.train.run_training",
            event_info="Started validating domain and training data...",
        )

        validate_files(
            args.fail_on_validation_warnings,
            args.validation_max_history,
            training_data_importer,
            sub_agents=args.sub_agents,
        )

    training_result = train_all(
        domain=domain,
        config=config,
        endpoints=args.endpoints,
        training_files=training_files,
        output=args.out,
        dry_run=args.dry_run,
        force_training=args.force,
        fixed_model_name=args.fixed_model_name,
        persist_nlu_training_data=args.persist_nlu_data,
        core_additional_arguments={
            **extract_core_additional_arguments(args),
        },
        nlu_additional_arguments=extract_nlu_additional_arguments(args),
        model_to_finetune=_model_for_finetuning(args),
        finetuning_epoch_fraction=args.epoch_fraction,
        remote_storage=args.remote_storage,
        file_importer=training_data_importer,
        keep_local_model_copy=args.keep_local_model_copy,
        remote_root_only=args.remote_root_only,
        sub_agents=args.sub_agents,
    )
    if training_result.code != 0 and can_exit:
        display_research_study_prompt()
        sys.exit(training_result.code)

    return training_result.model


def _model_for_finetuning(args: argparse.Namespace) -> Optional[Text]:
    if args.finetune is not None:
        structlogger.error(
            "cli.train.incremental_training_not_supported",
            event_info=(
                "Incremental training (--finetune) is "
                "not supported in Rasa 3.14.0 onwards. "
                "Please retrain your model from scratch "
                "if you have updated your configuration. "
            ),
        )
        display_research_study_prompt()
        sys.exit(1)
    return None


def run_core_training(args: argparse.Namespace) -> Optional[Text]:
    """Trains a Rasa Core model only.

    Args:
        args: Command-line arguments to configure training.

    Returns:
        Path to a trained model or `None` if training was not successful.
    """
    from rasa.model_training import train_core

    args.domain = get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATHS, none_is_valid=True
    )
    story_file = get_validated_path(
        args.stories, "stories", DEFAULT_DATA_PATH, none_is_valid=True
    )
    additional_arguments = {
        **extract_core_additional_arguments(args),
    }

    # Policies might be a list for the compare training. Do normal training
    # if only list item was passed.
    if not isinstance(args.config, list) or len(args.config) == 1:
        if isinstance(args.config, list):
            args.config = args.config[0]

        config = get_validated_config(args.config, CONFIG_MANDATORY_KEYS_CORE)

        Configuration.initialise_message_processing(
            message_processing_config_path=Path(config)
        ).initialise_empty_endpoints()

        return asyncio.run(
            train_core(
                domain=args.domain,
                config=config,
                stories=story_file,
                output=args.out,
                fixed_model_name=args.fixed_model_name,
                additional_arguments=additional_arguments,
                model_to_finetune=_model_for_finetuning(args),
                finetuning_epoch_fraction=args.epoch_fraction,
                keep_local_model_copy=args.keep_local_model_copy,
            )
        )
    else:
        Configuration.initialise_empty()
        asyncio.run(do_compare_training(args, story_file, additional_arguments))
        return None


def run_nlu_training(args: argparse.Namespace) -> Optional[Text]:
    """Trains an NLU model.

    Args:
        args: Namespace arguments.

    Returns:
        Path to a trained model or `None` if training was not successful.
    """
    from rasa.model_training import train_nlu

    config = get_validated_config(args.config, CONFIG_MANDATORY_KEYS_NLU)
    nlu_data = get_validated_path(
        args.nlu, "nlu", DEFAULT_DATA_PATH, none_is_valid=True
    )

    if args.domain:
        args.domain = get_validated_path(
            args.domain, "domain", DEFAULT_DOMAIN_PATHS, none_is_valid=True
        )

    return asyncio.run(
        train_nlu(
            config=config,
            nlu_data=nlu_data,
            output=args.out,
            fixed_model_name=args.fixed_model_name,
            persist_nlu_training_data=args.persist_nlu_data,
            additional_arguments={
                **extract_nlu_additional_arguments(args),
            },
            domain=args.domain,
            model_to_finetune=_model_for_finetuning(args),
            finetuning_epoch_fraction=args.epoch_fraction,
            keep_local_model_copy=args.keep_local_model_copy,
        )
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
