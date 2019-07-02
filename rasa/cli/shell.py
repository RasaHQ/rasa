import argparse
import logging
import os

from typing import List

from rasa.cli.arguments import shell as arguments
from rasa.cli.utils import print_error
from rasa.exceptions import ModelNotFound


logger = logging.getLogger(__name__)


# noinspection PyProtectedMember


def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    shell_parser = subparsers.add_parser(
        "shell",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Loads your trained model and lets you talk to your assistant on the command line.",
    )
    shell_parser.set_defaults(func=shell)

    run_subparsers = shell_parser.add_subparsers()
    shell_nlu_subparser = run_subparsers.add_parser(
        "nlu",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Interprets messages on the command line using your NLU model.",
    )
    shell_nlu_subparser.set_defaults(func=shell_nlu)

    arguments.set_shell_arguments(shell_parser)
    arguments.set_shell_nlu_arguments(shell_nlu_subparser)


def shell_nlu(args: argparse.Namespace):
    from rasa.cli.utils import get_validated_path
    from rasa.constants import DEFAULT_MODELS_PATH
    from rasa.model import get_model, get_model_subdirectories
    import rasa.nlu.run

    args.connector = "cmdline"

    model = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

    try:
        model_path = get_model(model)
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `rasa train nlu`."
        )
        return

    _, nlu_model = get_model_subdirectories(model_path)

    if not os.path.exists(nlu_model):
        print_error(
            "No NLU model found. Train a model before running the "
            "server using `rasa train nlu`."
        )
        return

    rasa.nlu.run.run_cmdline(nlu_model)


def shell(args: argparse.Namespace):
    from rasa.cli.utils import get_validated_path
    from rasa.constants import DEFAULT_MODELS_PATH
    from rasa.model import get_model, get_model_subdirectories

    args.connector = "cmdline"

    model = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

    try:
        model_path = get_model(model)
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `rasa train`."
        )
        return

    core_model, nlu_model = get_model_subdirectories(model_path)

    if not os.path.exists(core_model):
        import rasa.nlu.run

        rasa.nlu.run.run_cmdline(nlu_model)
    else:
        import rasa.cli.run

        rasa.cli.run.run(args)
