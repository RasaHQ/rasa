import argparse
import os
import shutil
from typing import List

import rasa.cli.train as train
from rasa.cli.arguments import interactive as arguments
from rasa import data, model


# noinspection PyProtectedMember
from rasa.cli.utils import get_validated_path, print_error, print_warning
from rasa.constants import DEFAULT_DATA_PATH


def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    interactive_parser = subparsers.add_parser(
        "interactive",
        conflict_handler="resolve",
        parents=parents,
        help="Teach the bot with interactive learning",
    )
    interactive_parser.set_defaults(func=interactive)

    arguments.set_interactive_arguments(interactive_parser)

    interactive_subparsers = interactive_parser.add_subparsers()
    interactive_core_parser = interactive_subparsers.add_parser(
        "core",
        conflict_handler="resolve",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Train a Rasa Core model with interactive learning",
    )
    interactive_core_parser.set_defaults(func=interactive_core)

    arguments.set_interactive_core_arguments(interactive_core_parser)


def interactive(args: argparse.Namespace):
    args.finetune = False  # Don't support finetuning

    training_files = [
        get_validated_path(f, "data", DEFAULT_DATA_PATH, none_is_valid=True)
        for f in args.data
    ]
    story_directory, nlu_data_directory = data.get_core_nlu_directories(training_files)

    if not os.listdir(story_directory) or not os.listdir(nlu_data_directory):
        print_error(
            "Cannot train initial Rasa model. Please provide NLU data and Core data."
        )
        exit(1)

    zipped_model = train.train(args)

    perform_interactive_learning(args, zipped_model)


def interactive_core(args: argparse.Namespace):

    args.finetune = False  # Don't support finetuning

    zipped_model = train.train_core(args)

    perform_interactive_learning(args, zipped_model)


def perform_interactive_learning(args, zipped_model):
    from rasa.core.train import do_interactive_learning

    if zipped_model:
        args.model = zipped_model
        model_path = model.unpack_model(zipped_model)
        args.core, args.nlu = model.get_model_subdirectories(model_path)
        stories_directory = data.get_core_directory(args.data)

        do_interactive_learning(args, stories_directory)

        shutil.rmtree(model_path)
    else:
        print_warning("No initial zipped trained model found.")
