import os
import shutil

import rasa.cli as cli
from rasa.model import unpack_model

from rasa_core.train import do_interactive_learning


def add_subparser(subparsers, parents):
    interactive_parser = subparsers.add_parser(
        "interactive",
        conflict_handler="resolve",
        parents=parents,
        help="Teach the bot with interactive learning")

    cli.run.add_run_arguments(interactive_parser)
    cli.train.add_general_arguments(interactive_parser)
    cli.train.add_core_arguments(interactive_parser)
    cli.train.add_nlu_arguments(interactive_parser)
    _add_interactive_arguments(interactive_parser)
    interactive_parser.set_defaults(func=interactive)


def _add_interactive_arguments(parser):
    parser.add_argument(
        '--finetune',
        default=False,
        action='store_true',
        help="retrain the model immediately based on feedback.")
    parser.add_argument(
        "--skip_visualization",
        default=False,
        action="store_true",
        help="Disables plotting the visualization during "
             "interactive learning")


def interactive(args):
    zipped_model = cli.train.train(args)
    model_path, core_path, nlu_path = unpack_model(zipped_model,
                                                   subdirectories=True)
    args.nlu = nlu_path
    args.core = core_path

    do_interactive_learning(args, args.stories)

    shutil.rmtree(model_path)
