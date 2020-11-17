import argparse
from typing import List, Optional, Text, Dict

from rasa.cli import SubParsersAction
import rasa.cli.arguments.train_chunks as train_arguments

import rasa.cli.utils
import rasa.cli.train
from rasa.shared.constants import (
    CONFIG_MANDATORY_KEYS,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
)


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

    train_arguments.set_train_in_chunks_arguments(train_parser)
    train_parser.set_defaults(func=train_chunks)


def train_chunks(args: argparse.Namespace) -> Optional[Text]:
    import rasa

    domain = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )

    config = rasa.cli.train.get_valid_config(args.config, CONFIG_MANDATORY_KEYS)

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

    return arguments


def extract_nlu_additional_arguments(args: argparse.Namespace) -> Dict:
    arguments = {}

    if "num_threads" in args:
        arguments["num_threads"] = args.num_threads

    return arguments
