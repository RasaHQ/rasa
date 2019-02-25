import shutil
from argparse import _SubParsersAction, ArgumentParser, Namespace
from typing import List

import rasa.cli.train as train
import rasa.cli.run as run
from rasa.model import unpack_model
import rasa.cli.default_arguments as defaults


def add_subparser(subparsers: _SubParsersAction,
                  parents: List[ArgumentParser]):
    interactive_parser = subparsers.add_parser(
        "interactive",
        conflict_handler="resolve",
        parents=parents,
        help="Teach the bot with interactive learning")

    run.add_run_arguments(interactive_parser)
    train.add_general_arguments(interactive_parser)
    train.add_core_arguments(interactive_parser)
    defaults.add_nlu_data_param(interactive_parser)
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
    model_path, core_path, nlu_path = unpack_model(zipped_model,
                                                   subdirectories=True)
    args.nlu = nlu_path
    args.core = core_path

    do_interactive_learning(args, args.stories)

    shutil.rmtree(model_path)
