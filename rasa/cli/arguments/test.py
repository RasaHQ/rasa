import argparse
from typing import Union

from rasa.constants import DEFAULT_MODELS_PATH, DEFAULT_CONFIG_PATH

from rasa.cli.arguments.default_arguments import (
    add_stories_param,
    add_core_model_param,
    add_model_param,
    add_nlu_data_param,
)
from rasa.model import get_latest_model


def set_test_arguments(parser):
    add_model_param(parser, add_positional_arg=False)

    core_arguments = parser.add_argument_group("Core Test arguments")
    add_test_core_arguments(core_arguments)
    add_stories_param(core_arguments, "test")
    add_url_param(core_arguments)

    nlu_arguments = parser.add_argument_group("NLU Test arguments")
    add_test_nlu_arguments(nlu_arguments)


def set_test_core_arguments(parser: argparse.ArgumentParser):
    core_arguments = parser.add_argument_group("Core Test arguments")
    add_test_core_arguments(core_arguments)
    add_stories_param(core_arguments, "test")
    add_url_param(core_arguments)

    add_test_core_model_param(parser)


def set_test_nlu_arguments(parser: argparse.ArgumentParser):
    add_test_nlu_model_param(parser)

    nlu_arguments = parser.add_argument_group("NLU Test arguments")
    add_test_nlu_arguments(nlu_arguments)


def add_test_core_arguments(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
):
    parser.add_argument(
        "--max-stories", type=int, help="Maximum number of stories to test on."
    )
    add_core_model_param(parser)
    parser.add_argument(
        "-u",
        "--nlu",
        type=str,
        help="NLU model to run with the server. None for regex interpreter.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output path for any files created during the evaluation.",
    )
    parser.add_argument(
        "--e2e",
        "--end-to-end",
        action="store_true",
        help="Run an end-to-end evaluation for combined action and "
        "intent prediction. Requires a story file in end-to-end "
        "format.",
    )
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Configuration file for the connectors as a yml file.",
    )
    parser.add_argument(
        "--fail-on-prediction-errors",
        action="store_true",
        help="If a prediction error is encountered, an exception "
        "is thrown. This can be used to validate stories during "
        "tests, e.g. on travis.",
    )


def add_test_nlu_arguments(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
):
    add_nlu_data_param(parser)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Model configuration file (cross validation only).",
    )
    parser.add_argument(
        "-f",
        "--folds",
        required=False,
        default=10,
        help="Number of cross validation folds (cross validation only).",
    )
    parser.add_argument(
        "--report",
        required=False,
        nargs="?",
        const="reports",
        default=False,
        help="Output path to save the intent/entity metrics report.",
    )
    parser.add_argument(
        "--successes",
        required=False,
        nargs="?",
        const="successes.json",
        default=False,
        help="Output path to save successful predictions.",
    )
    parser.add_argument(
        "--errors",
        required=False,
        default="errors.json",
        help="Output path to save model errors.",
    )
    parser.add_argument(
        "--histogram",
        required=False,
        default="hist.png",
        help="Output path for the confidence histogram.",
    )
    parser.add_argument(
        "--confmat",
        required=False,
        default="confmat.png",
        help="Output path for the confusion matrix plot.",
    )


def add_url_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]):
    parser.add_argument(
        "--url",
        type=str,
        help="If supplied, downloads a story file from a URL and "
        "trains on it. Fetches the data by sending a GET request "
        "to the supplied URL.",
    )


def add_test_nlu_model_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Path to a trained Rasa model. If a directory "
        "is specified, it will use the latest model "
        "in this directory. If none is given it will "
        "perform crossvalidation.",
    )


def add_test_core_model_param(parser: argparse.ArgumentParser):
    default_path = get_latest_model(DEFAULT_MODELS_PATH)
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        default=[default_path],
        help="Path to a pre-trained model. If it is a 'tar.gz' file that model file "
        "will be used. If it is a directory, the latest model in that directory "
        "will be used. If multiple 'tar.gz' files are provided, all those models "
        "will be compared.",
    )
