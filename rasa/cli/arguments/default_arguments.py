import argparse
import logging
from enum import Enum
from typing import List, Optional, Text, Union

from rasa.core.constants import DEFAULT_SUB_AGENTS
from rasa.core.persistor import RemoteStorageType, StorageType, parse_remote_storage
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODELS_PATH,
)


def add_model_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
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
    default: Optional[Text] = None,
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
        help="The policy and pipeline configuration of your bot.",
    )


def add_out_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    help_text: Text,
    default: Optional[Text] = DEFAULT_MODELS_PATH,
    required: bool = False,
) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default=default,
        help=help_text,
        # The desired behaviour is that required indicates if this argument must
        # have a value, but argparse interprets it as "must have a value
        # from user input", so we toggle it only if our default is not set
        required=required and default is None,
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
        # The desired behaviour is that required indicates if this argument must
        # have a value, but argparse interprets it as "must have a value
        # from user input", so we toggle it only if our default is not set
        required=required and default is None,
    )


def add_logging_options(parser: argparse.ArgumentParser) -> None:
    """Add options to an argument parser to configure logging levels."""
    logging_arguments = parser.add_argument_group(
        "Python Logging Options",
        "You can control level of log messages printed. "
        "In addition to these arguments, a more fine grained configuration can be "
        "achieved with environment variables. See online documentation for more info.",
    )

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

    logging_arguments.add_argument(
        "--logging-config-file",
        type=str,
        help="If set, the name of the logging configuration file will be set "
        "to the given name.",
    )


def add_remote_storage_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    required: bool = False,
) -> None:
    parser.add_argument(
        "--remote-storage",
        help="Remote storage which should be used to store/load the model. "
        f"Supported storages are: {RemoteStorageType.list()}. "
        "You can also provide your own implementation of the `Persistor` interface.",
        required=required,
        type=parse_remote_storage_arg,
    )


def add_remote_root_only_param(
    parser: argparse.ArgumentParser, required: bool = False
) -> None:
    parser.add_argument(
        "--remote-root-only",
        action="store_true",
        help="If set, models will be stored only at the root directory "
        "of the remote storage.",
        required=required,
    )


def parse_remote_storage_arg(value: str) -> StorageType:
    try:
        return parse_remote_storage(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


class SkipYamlValidation(Enum):
    DOMAIN = "domain"

    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in SkipYamlValidation]


def add_skip_validation_flag(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    parser.add_argument(
        "--skip-yaml-validation",
        default=[],
        choices=SkipYamlValidation.list(),
        action="append",
        help="Skip YAML validation for selected parts of the training data.",
    )


def add_sub_agents_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
) -> None:
    parser.add_argument(
        "--sub-agents",
        type=str,
        default=DEFAULT_SUB_AGENTS,
        help="Directory that specifies sub-agents to use (default: %(default)s).",
    )
