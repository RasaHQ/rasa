import argparse
import logging
from typing import Text, Union, Optional

from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENDPOINTS_PATH,
)


def add_model_param(
    parser: argparse.ArgumentParser,
    model_name: Text = "Rasa",
    add_positional_arg: bool = True,
    default: Optional[Text] = DEFAULT_MODELS_PATH,
) -> None:
    help_text = (
        "Path to a trained {} model. If a directory is specified, it will "
        "use the latest model in this directory.".format(model_name)
    )
    parser.add_argument("-m", "--model", type=str, default=default, help=help_text)
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
        help=f"File or folder containing your {stories_name} stories.",
    )


def add_nlu_data_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    help_text: Text,
    default: Optional[Text] = DEFAULT_DATA_PATH,
) -> None:
    parser.add_argument("-u", "--nlu", type=str, default=default, help=help_text)


def add_domain_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    default: Optional[Text] = DEFAULT_DOMAIN_PATH,
) -> None:
    parser.add_argument(
        "-d",
        "--domain",
        type=str,
        default=default,
        help="Domain specification. This can be a single YAML file, or a directory "
        "that contains several files with domain specifications in it. The content "
        "of these files will be read and merged together.",
    )


def add_config_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    default: Optional[Text] = DEFAULT_CONFIG_PATH,
) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=default,
        help="The policy and NLU pipeline configuration of your bot.",
    )


def add_out_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    help_text: Text,
    default: Optional[Text] = DEFAULT_MODELS_PATH,
    required: bool = False,
) -> None:
    parser.add_argument(
        "--out", type=str, default=default, help=help_text, required=required
    )


def add_endpoint_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    help_text: Text,
    default: Optional[Text] = DEFAULT_ENDPOINTS_PATH,
) -> None:
    """Adds an option to an argument parser to configure endpoints path."""
    parser.add_argument("--endpoints", type=str, default=default, help=help_text)


def add_data_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    default: Optional[Text] = DEFAULT_DATA_PATH,
    required: bool = False,
    data_type: Text = "Rasa ",
) -> None:
    parser.add_argument(
        "--data",
        default=default,
        nargs="+",
        type=str,
        help=f"Paths to the files or directories containing {data_type} data.",
        required=required,
    )


def add_logging_options(parser: argparse.ArgumentParser) -> None:
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
