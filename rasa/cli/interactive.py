import argparse
import os
import shutil
from typing import List, Text

import rasa.cli.train as train
from rasa.cli.arguments import interactive as arguments
from rasa import data, model


# noinspection PyProtectedMember
from rasa.cli.utils import get_validated_path, print_error
from rasa.constants import (
    DEFAULT_DATA_PATH,
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
    interactive_core_parser.set_defaults(func=interactive_core)

    arguments.set_interactive_arguments(interactive_parser)
    arguments.set_interactive_core_arguments(interactive_core_parser)


def interactive(args: argparse.Namespace):
    args.fixed_model_name = None
    args.store_uncompressed = False

    if args.model is None:
        check_training_data(args)
        zipped_model = train.train(args)
    else:
        zipped_model = get_provided_model(args.model)

    perform_interactive_learning(args, zipped_model)


def interactive_core(args: argparse.Namespace):
    args.fixed_model_name = None
    args.store_uncompressed = False

    if args.model is None:
        zipped_model = train.train_core(args)
    else:
        zipped_model = get_provided_model(args.model)

    perform_interactive_learning(args, zipped_model)


def perform_interactive_learning(args, zipped_model):
    from rasa.core.train import do_interactive_learning

    if zipped_model and os.path.exists(zipped_model):
        args.model = zipped_model

        with model.unpack_model(zipped_model) as model_path:
            args.core, args.nlu = model.get_model_subdirectories(model_path)
            stories_directory = data.get_core_directory(args.data)

            args.endpoints = get_validated_path(
                args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
            )

            do_interactive_learning(args, stories_directory)
    else:
        print_error(
            "Interactive learning process cannot be started as no initial model was "
            "found.  Use 'rasa train' to train a model."
        )


def get_provided_model(arg_model: Text):
    model_path = get_validated_path(arg_model, "model", DEFAULT_MODELS_PATH)

    if os.path.isdir(model_path):
        model_path = get_latest_model(model_path)

    return model_path


def check_training_data(args):
    training_files = [
        get_validated_path(f, "data", DEFAULT_DATA_PATH, none_is_valid=True)
        for f in args.data
    ]
    story_directory, nlu_data_directory = data.get_core_nlu_directories(training_files)
    if not os.listdir(story_directory) or not os.listdir(nlu_data_directory):
        print_error(
            "Cannot train initial Rasa model. Please provide NLU and Core data "
            "using the '--data' argument."
        )
        exit(1)
