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
    """Add all parsers for training in chunks.

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

    train_subparsers = train_parser.add_subparsers()
    train_core_parser = train_subparsers.add_parser(
        "core",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Trains a Rasa Core model in smaller chunks using your stories.",
    )
    train_core_parser.set_defaults(func=train_chunks_core)

    train_nlu_parser = train_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Trains a Rasa NLU model in smaller chunks using your NLU data.",
    )
    train_nlu_parser.set_defaults(func=train_chunks_nlu)

    train_arguments.set_train_in_chunks_core_arguments(train_core_parser)
    train_arguments.set_train_in_chunks_nlu_arguments(train_nlu_parser)
    train_arguments.set_train_in_chunks_arguments(train_parser)

    train_parser.set_defaults(func=train_chunks)


def train_chunks(args: argparse.Namespace) -> Optional[Text]:
    """Train a model using smaller chunks.

    Args:
        args: the command line arguments

    Returns:
        The path where the trained model is stored.
    """
    import rasa.train_chunks

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

    return rasa.train_chunks.train_in_chunks(
        domain=domain,
        config=config,
        training_files=training_files,
        output=args.out,
        force_training=args.force,
        fixed_model_name=args.fixed_model_name,
        core_additional_arguments=_extract_core_additional_arguments(args),
        nlu_additional_arguments=_extract_nlu_additional_arguments(args),
    )


def train_chunks_core(args: argparse.Namespace) -> Optional[Text]:
    """Train a Rasa Core model using smaller chunks.

    Args:
        args: the command line arguments

    Returns:
        The path where the trained model is stored.
    """
    import rasa.train_chunks

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

    return rasa.train_chunks.train_core_in_chunks(
        domain=domain,
        config=config,
        training_files=training_files,
        output=args.out,
        fixed_model_name=args.fixed_model_name,
        additional_arguments=_extract_core_additional_arguments(args),
    )


def train_chunks_nlu(args: argparse.Namespace) -> Optional[Text]:
    """Train a Rasa NLU model using smaller chunks.

    Args:
        args: the command line arguments

    Returns:
        The path where the trained model is stored.
    """
    import rasa.train_chunks

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

    return rasa.train_chunks.train_nlu_in_chunks(
        domain=domain,
        config=config,
        training_files=training_files,
        output=args.out,
        fixed_model_name=args.fixed_model_name,
        additional_arguments=_extract_nlu_additional_arguments(args),
    )


def _extract_core_additional_arguments(args: argparse.Namespace) -> Dict:
    arguments = {}

    if "augmentation" in args:
        arguments["augmentation_factor"] = args.augmentation

    return arguments


def _extract_nlu_additional_arguments(args: argparse.Namespace) -> Dict:
    arguments = {}

    if "num_threads" in args:
        arguments["num_threads"] = args.num_threads

    return arguments
