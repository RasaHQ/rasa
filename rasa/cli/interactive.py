import argparse
import shutil
from typing import List

import rasa.cli.run as run
import rasa.cli.train as train
from rasa import data, model


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    interactive_parser = subparsers.add_parser(
        "interactive",
        conflict_handler="resolve",
        parents=parents,
        help="Teach the bot with interactive learning",
    )
    run.add_run_arguments(interactive_parser)
    train.add_general_arguments(interactive_parser)
    train.add_domain_param(interactive_parser)
    train.add_joint_parser_arguments(interactive_parser)
    _add_interactive_arguments(interactive_parser)
    interactive_parser.set_defaults(func=interactive)

    interactive_core_subparsers = interactive_parser.add_subparsers()
    interactive_core_parser = interactive_core_subparsers.add_parser(
        "core",
        conflict_handler="resolve",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Train a Rasa Core model with interactive learning",
    )
    interactive_core_parser.set_defaults(func=interactive_core)

    import rasa.core.cli.train as core_cli

    train.add_domain_param(interactive_core_parser)
    core_cli.add_general_args(interactive_core_parser)
    train.add_stories_param(interactive_core_parser)
    train.add_domain_param(interactive_core_parser)
    run.add_run_arguments(interactive_core_parser)
    _add_interactive_arguments(interactive_core_parser)
    train.add_general_arguments(interactive_core_parser)


def _add_interactive_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--skip-visualization",
        default=False,
        action="store_true",
        help="Disables plotting the visualization during interactive learning",
    )


def interactive(args: argparse.Namespace):
    from rasa.core.train import do_interactive_learning

    args.finetune = False  # Don't support finetuning

    zipped_model = train.train(args)

    if zipped_model:
        model_path = model.unpack_model(zipped_model)
        args.core, args.nlu = model.get_model_subdirectories(model_path)
        stories_directory = data.get_core_directory(args.data)

        do_interactive_learning(args, stories_directory)

        shutil.rmtree(model_path)


def interactive_core(args: argparse.Namespace):
    from rasa.core.train import do_interactive_learning

    args.finetune = False  # Don't support finetuning

    zipped_model = train.train_core(args)

    if zipped_model:
        model_path = model.unpack_model(zipped_model)
        args.core, args.nlu = model.get_model_subdirectories(model_path)
        stories_directory = data.get_core_directory(args.data)

        do_interactive_learning(args, stories_directory)

        shutil.rmtree(model_path)
