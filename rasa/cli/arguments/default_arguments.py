import argparse
import logging
from typing import Text, Union

from rasa.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_CONFIG_PATH,
)


def add_model_param(
    parser: argparse.ArgumentParser,
    model_name: Text = "Rasa",
    add_positional_arg: bool = True,
):
    help_text = (
        "Path to a trained {} model. If a directory is specified, it will "
        "use the latest model in this directory.".format(model_name)
    )
    parser.add_argument(
        "-m", "--model", type=str, default=DEFAULT_MODELS_PATH, help=help_text
    )
    if add_positional_arg:
        parser.add_argument(
            "model-as-positional-argument", nargs="?", type=str, help=help_text
        )


def add_stories_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    stories_name: Text = "training",
) -> None:
    parser.add_argument(
        "-s",
        "--stories",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="File or folder containing {} stories.".format(stories_name),
    )


def add_nlu_data_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-u",
        "--nlu",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="File or folder containing your NLU training data.",
    )


def add_domain_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d",
        "--domain",
        type=str,
        default=DEFAULT_DOMAIN_PATH,
        help="Domain specification (yml file).",
    )


def add_config_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="The policy and NLU pipeline configuration of your bot.",
    )


def add_out_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_MODELS_PATH,
        help="Directory where your models should be stored.",
    )


def add_core_model_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--core", type=str, help="Path to a pre-trained Rasa Core model (tar.gz file)."
    )


def add_logging_options(parser: argparse.ArgumentParser):
    """Add options to an argument parser to configure logging levels."""

    logging_arguments = parser.add_argument_group("Python Logging Options")

    # arguments for logging configuration
    logging_arguments.add_argument(
        "-v",
        "--verbose",
        help="Be verbose. Sets logging level to INFO.",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    logging_arguments.add_argument(
        "-vv",
        "--debug",
        help="Print lots of debugging statements. Sets logging level to DEBUG.",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
    )
    logging_arguments.add_argument(
        "--quiet",
        help="Be quiet! Sets logging level to WARNING.",
        action="store_const",
        dest="loglevel",
        const=logging.WARNING,
    )
