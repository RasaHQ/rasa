import argparse
import os
import sys
from typing import List, Optional, Text, Dict

from rasa.cli import SubParsersAction
import rasa.cli.arguments.train_chunks as train_arguments

import rasa.cli.utils
from rasa.shared.utils.cli import print_error
from rasa.shared.constants import (
    CONFIG_MANDATORY_KEYS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
)

import rasa.utils.common


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all training parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    train_parser = subparsers.add_parser(
        "train-in-chunks",
        help="Trains a Rasa model in smaller chunks using your NLU data and stories.",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    train_arguments.set_train_chunk_arguments(train_parser)
    train_parser.set_defaults(func=train_chunks)


def train_chunks(args: argparse.Namespace) -> Optional[Text]:
    import rasa

    domain = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )

    config = _get_valid_config(args.config, CONFIG_MANDATORY_KEYS)

    training_files = [
        rasa.cli.utils.get_validated_path(
            f, "data", DEFAULT_DATA_PATH, none_is_valid=True
        )
        for f in args.data
    ]

    return rasa.train_in_chunks(
        domain=domain,
        config=config,
        training_files=training_files,
        output=args.out,
        force_training=args.force,
        fixed_model_name=args.fixed_model_name,
        core_additional_arguments=extract_core_additional_arguments(args),
        nlu_additional_arguments=extract_nlu_additional_arguments(args),
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


def _get_valid_config(
    config: Optional[Text],
    mandatory_keys: List[Text],
    default_config: Text = DEFAULT_CONFIG_PATH,
) -> Text:
    """Get a config from a config file and check if it is valid.

    Exit if the config isn't valid.

    Args:
        config: Path to the config file.
        mandatory_keys: The keys that have to be specified in the config file.
        default_config: default config to use if the file at `config` doesn't exist.

    Returns: The path to the config file if the config is valid.
    """
    config = rasa.cli.utils.get_validated_path(config, "config", default_config)

    if not os.path.exists(config):
        print_error(
            "The config file '{}' does not exist. Use '--config' to specify a "
            "valid config file."
            "".format(config)
        )
        sys.exit(1)

    missing_keys = rasa.cli.utils.missing_config_keys(config, mandatory_keys)
    if missing_keys:
        print_error(
            "The config file '{}' is missing mandatory parameters: "
            "'{}'. Add missing parameters to config file and try again."
            "".format(config, "', '".join(missing_keys))
        )
        sys.exit(1)

    return config
