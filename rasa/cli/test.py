import argparse
import logging
from typing import List, Optional, Text, Union

from rasa.cli.default_arguments import add_model_param, add_stories_param
from rasa.cli.utils import validate_path
from rasa.constants import (DEFAULT_ENDPOINTS_PATH, DEFAULT_CONFIG_PATH,
                            DEFAULT_NLU_DATA_PATH, DEFAULT_MODELS_PATH)
from rasa.model import get_latest_model, get_model

logger = logging.getLogger(__name__)


def add_subparser(subparsers: argparse._SubParsersAction,
                  parents: List[argparse.ArgumentParser]):
    test_parser = subparsers.add_parser(
        "test",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Test a trained model")

    test_subparsers = test_parser.add_subparsers()
    test_core_parser = test_subparsers.add_parser(
        "core",
        conflict_handler='resolve',
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Test Rasa Core")

    test_nlu_parser = test_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Test Rasa NLU")

    for p in [test_parser, test_core_parser]:
        core_arguments = p.add_argument_group("Core Test arguments")
        _add_core_arguments(core_arguments)
    _add_core_subparser_arguments(test_core_parser)

    for p in [test_parser, test_nlu_parser]:
        nlu_arguments = p.add_argument_group("NLU Test arguments")
        _add_nlu_arguments(nlu_arguments)
    _add_nlu_subparser_arguments(test_nlu_parser)

    test_core_parser.set_defaults(func=test_core)
    test_nlu_parser.set_defaults(func=test_nlu)
    test_parser.set_defaults(func=test)


def _add_core_arguments(parser: Union[argparse.ArgumentParser,
                                      argparse._ActionsContainer]):
    from rasa_core.cli.test import add_evaluation_arguments

    add_evaluation_arguments(parser)
    add_model_param(parser, "Core")
    add_stories_param(parser, "test")

    parser.add_argument(
        '--url',
        type=str,
        help="If supplied, downloads a story file from a URL and "
             "trains on it. Fetches the data by sending a GET request "
             "to the supplied URL.")


def _add_core_subparser_arguments(parser: argparse.ArgumentParser):
    default_path = get_latest_model(DEFAULT_MODELS_PATH)
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=default_path,
        help="Path to a pre-trained model. If it is a directory all models "
             "in this directory will be compared.")


def _add_nlu_arguments(parser: Union[argparse.ArgumentParser,
                                     argparse._ActionsContainer]):
    parser.add_argument('-u', '--nlu',
                        type=str,
                        default="data/nlu",
                        help="file containing training/evaluation data")

    parser.add_argument('-c', '--config',
                        type=str,
                        default="config.yml",
                        help="model configuration file (crossvalidation only)")

    parser.add_argument('-f', '--folds', required=False, default=10,
                        help="number of CV folds (crossvalidation only)")

    parser.add_argument('--report', required=False, nargs='?',
                        const="reports", default=False,
                        help="output path to save the intent/entity"
                             "metrics report")

    parser.add_argument('--successes', required=False, nargs='?',
                        const="successes.json", default=False,
                        help="output path to save successful predictions")

    parser.add_argument('--errors', required=False, default="errors.json",
                        help="output path to save model errors")

    parser.add_argument('--histogram', required=False, default="hist.png",
                        help="output path for the confidence histogram")

    parser.add_argument('--confmat', required=False, default="confmat.png",
                        help="output path for the confusion matrix plot")


def _add_nlu_subparser_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Path to a pre-trained model. If none is given it will "
             "perform crossvalidation.")


def test_core(args: argparse.Namespace, model_path: Optional[Text] = None
              ) -> None:
    from rasa.test import test_core

    validate_path(args, "model", DEFAULT_MODELS_PATH)
    validate_path(args, "endpoints", DEFAULT_ENDPOINTS_PATH, True)
    validate_path(args, "config", DEFAULT_CONFIG_PATH)

    test_core(model_path=model_path, **vars(args))


def test_nlu(args: argparse.Namespace, model_path: Optional[Text] = None
             ) -> None:
    from rasa.test import test_nlu, test_nlu_with_cross_validation

    validate_path(args, "model", DEFAULT_MODELS_PATH)
    validate_path(args, "nlu", DEFAULT_NLU_DATA_PATH)

    model_path = model_path or args.model

    if model_path:
        test_nlu(nlu_data=args.nlu, **vars(args))
    else:
        print("No model specified. Model will be trained using cross "
              "validation.")
        validate_path(args, "config", DEFAULT_CONFIG_PATH)

        test_nlu_with_cross_validation(args.config, args.nlu, args.folds)


def test(args: argparse.Namespace):
    validate_path(args, "model", DEFAULT_MODELS_PATH)
    model_path = get_model(args.model)

    test_core(args, model_path)
    test_nlu(args, model_path)
