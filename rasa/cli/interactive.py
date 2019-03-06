import shutil
from argparse import _SubParsersAction, ArgumentParser, Namespace
from typing import List

import rasa.cli.train as train
import rasa.cli.run as run
from rasa import model, data


def add_subparser(subparsers: _SubParsersAction,
                  parents: List[ArgumentParser]):
    interactive_parser = subparsers.add_parser(
        "interactive",
        conflict_handler="resolve",
        parents=parents,
        help="Teach the bot with interactive learning")

    run.add_run_arguments(interactive_parser)
    train.add_general_arguments(interactive_parser)
    train.add_domain_param(interactive_parser)
    train.add_joint_parser_arguments(interactive_parser)
    _add_interactive_arguments(interactive_parser)
    interactive_parser.set_defaults(func=interactive)


def _add_interactive_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--skip_visualization",
        default=False,
        action="store_true",
        help="Disables plotting the visualization during "
             "interactive learning")


def interactive(args: Namespace):
    from rasa_core.train import do_interactive_learning

    args.finetune = False  # Don't support finetuning

    zipped_model = train.train(args)
    model_path = model.unpack_model(zipped_model)
    args.core, args.nlu = model.get_model_subdirectories(model_path)
    stories_directory = data.get_core_directory(args.data)

    do_interactive_learning(args, stories_directory)

    shutil.rmtree(model_path)
