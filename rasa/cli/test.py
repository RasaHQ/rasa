import argparse
import logging
import os
from typing import List

from rasa import data
from rasa.cli.arguments import test as arguments
from rasa.cli.utils import get_validated_path
from rasa.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_RESULTS_PATH,
)
from rasa.test import test_compare

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    test_parser = subparsers.add_parser(
        "test",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests a trained Rasa model using your test NLU data and stories.",
    )

    arguments.set_test_arguments(test_parser)

    test_subparsers = test_parser.add_subparsers()
    test_core_parser = test_subparsers.add_parser(
        "core",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests a trained Rasa Core model using your test stories.",
    )
    arguments.set_test_core_arguments(test_core_parser)

    test_nlu_parser = test_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests a trained Rasa NLU model using your test NLU data.",
    )
    arguments.set_test_nlu_arguments(test_nlu_parser)

    test_core_parser.set_defaults(func=test_core)
    test_nlu_parser.set_defaults(func=test_nlu)
    test_parser.set_defaults(func=test)


def test_core(args: argparse.Namespace) -> None:
    from rasa.test import test_core

    endpoints = get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    stories = get_validated_path(args.stories, "stories", DEFAULT_DATA_PATH)
    stories = data.get_core_directory(stories)
    output = args.out or DEFAULT_RESULTS_PATH

    if not os.path.exists(output):
        os.makedirs(output)

    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]

    if isinstance(args.model, str):
        model_path = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

        test_core(
            model=model_path,
            stories=stories,
            endpoints=endpoints,
            output=output,
            kwargs=vars(args),
        )

    else:
        test_compare(args.model, stories, output)


def test_nlu(args: argparse.Namespace) -> None:
    from rasa.test import test_nlu, test_nlu_with_cross_validation

    nlu_data = get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    nlu_data = data.get_nlu_directory(nlu_data)

    if not args.cross_validation:
        model_path = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)
        test_nlu(model_path, nlu_data, vars(args))
    else:
        print ("No model specified. Model will be trained using cross validation.")
        config = get_validated_path(args.config, "config", DEFAULT_CONFIG_PATH)

        test_nlu_with_cross_validation(config, nlu_data, vars(args))


def test(args: argparse.Namespace):
    test_core(args)
    test_nlu(args)
