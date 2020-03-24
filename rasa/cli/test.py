import argparse
import logging
import os
from typing import List

from rasa.cli.arguments import test as arguments
from rasa.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_E2E_TESTS_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_RESULTS_PATH,
    CONFIG_SCHEMA_FILE,
)
import rasa.utils.validation as validation_utils
import rasa.cli.utils as cli_utils
import rasa.utils.io as io_utils

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    """Adds a test subparser."""
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


def run_core_test(args: argparse.Namespace) -> None:
    """Run core tests."""
    from rasa import data
    from rasa.test import test_core_models_in_directory, test_core, test_core_models

    endpoints = cli_utils.get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    stories = cli_utils.get_validated_path(args.stories, "stories", DEFAULT_DATA_PATH)
    stories = data.get_core_directory(stories)
    output = args.out or DEFAULT_RESULTS_PATH

    io_utils.create_directory(output)

    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]

    if isinstance(args.model, str):
        model_path = cli_utils.get_validated_path(
            args.model, "model", DEFAULT_MODELS_PATH
        )

        if args.evaluate_model_directory:
            test_core_models_in_directory(args.model, stories, output)
        else:
            test_core(
                model=model_path,
                stories=stories,
                endpoints=endpoints,
                output=output,
                additional_arguments=vars(args),
            )

    else:
        test_core_models(args.model, stories, output)


def run_nlu_test(args: argparse.Namespace) -> None:
    """Run NLU tests."""
    from rasa import data
    from rasa.test import compare_nlu_models, perform_nlu_cross_validation, test_nlu

    nlu_data = cli_utils.get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    nlu_data = data.get_nlu_directory(nlu_data)
    output = args.out or DEFAULT_RESULTS_PATH
    args.errors = not args.no_errors

    io_utils.create_directory(output)

    if args.config is not None and len(args.config) == 1:
        args.config = os.path.abspath(args.config[0])
        if os.path.isdir(args.config):
            args.config = io_utils.list_files(args.config)

    if isinstance(args.config, list):
        logger.info(
            "Multiple configuration files specified, running nlu comparison mode."
        )

        config_files = []
        for file in args.config:
            try:
                validation_utils.validate_yaml_schema(
                    io_utils.read_file(file),
                    CONFIG_SCHEMA_FILE,
                    show_validation_errors=False,
                )
                config_files.append(file)
            except validation_utils.InvalidYamlFileError:
                logger.debug(
                    f"Ignoring file '{file}' as it is not a valid config file."
                )
                continue

        compare_nlu_models(
            configs=config_files,
            nlu=nlu_data,
            output=output,
            runs=args.runs,
            exclusion_percentages=args.percentages,
        )
    elif args.cross_validation:
        logger.info("Test model using cross validation.")
        config = cli_utils.get_validated_path(
            args.config, "config", DEFAULT_CONFIG_PATH
        )
        perform_nlu_cross_validation(config, nlu_data, output, vars(args))
    else:
        model_path = cli_utils.get_validated_path(
            args.model, "model", DEFAULT_MODELS_PATH
        )

        test_nlu(model_path, nlu_data, output, vars(args))


def test(args: argparse.Namespace):
    """Run end-to-end tests."""
    setattr(args, "e2e", True)
    run_core_test(args)
    run_nlu_test(args)
