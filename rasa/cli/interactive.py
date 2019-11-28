import argparse
import os
from typing import List, Text, Optional

from rasa.cli import utils
import rasa.cli.train as train
from rasa.cli.arguments import interactive as arguments
from rasa import data, model


# noinspection PyProtectedMember
from rasa.constants import (
    DEFAULT_MODELS_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.model import get_latest_model


def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    interactive_parser = subparsers.add_parser(
        "interactive",
        conflict_handler="resolve",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Starts an interactive learning session to create new training data for a "
        "Rasa model by chatting.",
    )
    interactive_parser.set_defaults(func=interactive)
    interactive_parser.add_argument(
        "--e2e",
        action="store_true",
        help="Save story files in e2e format. In this format user messages will be included in the stories.",
    )

    interactive_subparsers = interactive_parser.add_subparsers()
    interactive_core_parser = interactive_subparsers.add_parser(
        "core",
        conflict_handler="resolve",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Starts an interactive learning session model to create new training data "
        "for a Rasa Core model by chatting. Uses the 'RegexInterpreter', i.e. "
        "`/<intent>` input format.",
    )
    interactive_core_parser.set_defaults(func=interactive, core_only=True)

    arguments.set_interactive_arguments(interactive_parser)
    arguments.set_interactive_core_arguments(interactive_core_parser)


def interactive(args: argparse.Namespace):
    _set_not_required_args(args)

    if args.model is None:
        zipped_model = train.train_core(args) if args.core_only else train.train(args)
        if not zipped_model:
            utils.print_error_and_exit(
                "Could not train an initial model. Either pass paths "
                "to the relevant training files (`--data`, `--config`, `--domain`), "
                "or use 'rasa train' to train a model."
            )
    else:
        zipped_model = get_provided_model(args.model)
        if not (zipped_model and os.path.exists(zipped_model)):
            utils.print_error_and_exit(
                f"Interactive learning process cannot be started as no initial model was "
                f"found at path '{args.model}'.  Use 'rasa train' to train a model."
            )

    perform_interactive_learning(args, zipped_model)


def _set_not_required_args(args: argparse.Namespace) -> None:
    args.fixed_model_name = None
    args.store_uncompressed = False
    args.core_only = False


def perform_interactive_learning(args, zipped_model) -> None:
    from rasa.core.train import do_interactive_learning

    args.model = zipped_model

    with model.unpack_model(zipped_model) as model_path:
        args.core, args.nlu = model.get_model_subdirectories(model_path)
        stories_directory = data.get_core_directory(args.data)

        args.endpoints = utils.get_validated_path(
            args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
        )

        do_interactive_learning(args, stories_directory)


def get_provided_model(arg_model: Text):
    model_path = utils.get_validated_path(arg_model, "model", DEFAULT_MODELS_PATH)

    if os.path.isdir(model_path):
        model_path = get_latest_model(model_path)

    return model_path
