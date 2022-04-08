import argparse
import asyncio
import logging
import os
from typing import List, Optional, Text, Dict, Union, Any

from rasa.cli import SubParsersAction
import rasa.shared.data
from rasa.shared.exceptions import YamlException
import rasa.shared.utils.io
import rasa.shared.utils.cli
from rasa.cli.arguments import test as arguments
from rasa.core.constants import (
    FAILED_STORIES_FILE,
    SUCCESSFUL_STORIES_FILE,
    STORIES_WITH_WARNINGS_FILE,
)
from rasa.shared.constants import (
    CONFIG_SCHEMA_FILE,
    DEFAULT_E2E_TESTS_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_PATH,
)
import rasa.shared.utils.validation as validation_utils
import rasa.cli.utils
import rasa.utils.common
from rasa.shared.importers.importer import TrainingDataImporter

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all test parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    test_parser = subparsers.add_parser(
        "test",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests Rasa models using your test NLU data and stories.",
    )

    arguments.set_test_arguments(test_parser)

    test_subparsers = test_parser.add_subparsers()
    test_core_parser = test_subparsers.add_parser(
        "core",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests Rasa Core models using your test stories.",
    )
    arguments.set_test_core_arguments(test_core_parser)

    test_nlu_parser = test_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests Rasa NLU models using your test NLU data.",
    )
    arguments.set_test_nlu_arguments(test_nlu_parser)

    test_core_parser.set_defaults(func=run_core_test)
    test_nlu_parser.set_defaults(func=run_nlu_test)
    test_parser.set_defaults(func=test, stories=DEFAULT_E2E_TESTS_PATH)


def _print_core_test_execution_info(args: argparse.Namespace) -> None:
    output = args.out or DEFAULT_RESULTS_PATH

    if args.successes:
        rasa.shared.utils.cli.print_info(
            f"Successful stories written to "
            f"'{os.path.join(output, SUCCESSFUL_STORIES_FILE)}'"
        )
    if not args.no_errors:
        rasa.shared.utils.cli.print_info(
            f"Failed stories written to '{os.path.join(output, FAILED_STORIES_FILE)}'"
        )
    if not args.no_warnings:
        rasa.shared.utils.cli.print_info(
            f"Stories with prediction warnings written to "
            f"'{os.path.join(output, STORIES_WITH_WARNINGS_FILE)}'"
        )


async def run_core_test_async(args: argparse.Namespace) -> None:
    """Run core tests."""
    from rasa.model_testing import (
        test_core_models_in_directory,
        test_core,
        test_core_models,
    )

    stories = rasa.cli.utils.get_validated_path(
        args.stories, "stories", DEFAULT_DATA_PATH
    )

    output = args.out or DEFAULT_RESULTS_PATH
    args.errors = not args.no_errors
    args.warnings = not args.no_warnings

    rasa.shared.utils.io.create_directory(output)

    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]

    if args.model is None:
        rasa.shared.utils.cli.print_error(
            "No model provided. Please make sure to specify "
            "the model to test with '--model'."
        )
        return

    if isinstance(args.model, str):
        model_path = rasa.cli.utils.get_validated_path(
            args.model, "model", DEFAULT_MODELS_PATH
        )

        if args.evaluate_model_directory:
            await test_core_models_in_directory(
                args.model, stories, output, use_conversation_test_files=args.e2e
            )
        else:
            await test_core(
                model=model_path,
                stories=stories,
                output=output,
                additional_arguments=vars(args),
                use_conversation_test_files=args.e2e,
            )

    else:
        await test_core_models(
            args.model, stories, output, use_conversation_test_files=args.e2e
        )

    _print_core_test_execution_info(args)


async def run_nlu_test_async(
    config: Optional[Union[Text, List[Text]]],
    data_path: Text,
    models_path: Text,
    output_dir: Text,
    cross_validation: bool,
    percentages: List[int],
    runs: int,
    no_errors: bool,
    domain_path: Text,
    all_args: Dict[Text, Any],
) -> None:
    """Runs NLU tests.

    Args:
        all_args: all arguments gathered in a Dict so we can pass it as one argument
                  to other functions.
        config: it refers to the model configuration file. It can be a single file or
                a list of multiple files or a folder with multiple config files inside.
        data_path: path for the nlu data.
        models_path: path to a trained Rasa model.
        output_dir: output path for any files created during the evaluation.
        cross_validation: indicates if it should test the model using cross validation
                          or not.
        percentages: defines the exclusion percentage of the training data.
        runs: number of comparison runs to make.
        domain_path: path to domain.
        no_errors: indicates if incorrect predictions should be written to a file
                   or not.
    """
    from rasa.model_testing import (
        compare_nlu_models,
        perform_nlu_cross_validation,
        test_nlu,
    )

    data_path = str(
        rasa.cli.utils.get_validated_path(data_path, "nlu", DEFAULT_DATA_PATH)
    )
    test_data_importer = TrainingDataImporter.load_from_dict(
        training_data_paths=[data_path], domain_path=domain_path
    )
    nlu_data = test_data_importer.get_nlu_data()

    output = output_dir or DEFAULT_RESULTS_PATH
    all_args["errors"] = not no_errors
    rasa.shared.utils.io.create_directory(output)

    if config is not None and len(config) == 1:
        config = os.path.abspath(config[0])
        if os.path.isdir(config):
            config = rasa.shared.utils.io.list_files(config)

    if isinstance(config, list):
        logger.info(
            "Multiple configuration files specified, running nlu comparison mode."
        )

        config_files = []
        for file in config:
            try:
                validation_utils.validate_yaml_schema(
                    rasa.shared.utils.io.read_file(file), CONFIG_SCHEMA_FILE
                )
                config_files.append(file)
            except YamlException:
                rasa.shared.utils.io.raise_warning(
                    f"Ignoring file '{file}' as it is not a valid config file."
                )
                continue
        await compare_nlu_models(
            configs=config_files,
            test_data=nlu_data,
            output=output,
            runs=runs,
            exclusion_percentages=percentages,
        )
    elif cross_validation:
        logger.info("Test model using cross validation.")
        # FIXME: supporting Union[Path, Text] down the chain
        # is the proper fix and needs more work
        config = str(
            rasa.cli.utils.get_validated_path(config, "config", DEFAULT_CONFIG_PATH)
        )
        config_importer = TrainingDataImporter.load_from_dict(config_path=config)

        config_dict = config_importer.get_config()
        await perform_nlu_cross_validation(config_dict, nlu_data, output, all_args)
    else:
        model_path = rasa.cli.utils.get_validated_path(
            models_path, "model", DEFAULT_MODELS_PATH
        )

        await test_nlu(model_path, data_path, output, all_args, domain_path=domain_path)


def run_nlu_test(args: argparse.Namespace) -> None:
    """Runs NLU tests.

    Args:
        args: the parsed CLI arguments for 'rasa test nlu'.
    """
    asyncio.run(
        run_nlu_test_async(
            args.config,
            args.nlu,
            args.model,
            args.out,
            args.cross_validation,
            args.percentages,
            args.runs,
            args.no_errors,
            args.domain,
            vars(args),
        )
    )


def run_core_test(args: argparse.Namespace) -> None:
    """Runs Core tests.

    Args:
        args: the parsed CLI arguments for 'rasa test core'.
    """
    asyncio.run(run_core_test_async(args))


def test(args: argparse.Namespace) -> None:
    """Run end-to-end tests."""
    setattr(args, "e2e", True)
    run_core_test(args)
    run_nlu_test(args)
