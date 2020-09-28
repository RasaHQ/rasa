import argparse
import logging
import uuid

from typing import List

from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import shell as arguments
from rasa.shared.utils.cli import print_error
from rasa.exceptions import ModelNotFound

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all shell parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    shell_parser = subparsers.add_parser(
        "shell",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=(
            "Loads your trained model and lets you talk to your "
            "assistant on the command line."
        ),
    )
    shell_parser.set_defaults(func=shell)

    shell_parser.add_argument(
        "--conversation-id",
        default=uuid.uuid4().hex,
        required=False,
        help="Set the conversation ID.",
    )

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
    from rasa.shared.constants import DEFAULT_MODELS_PATH
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

    if not nlu_model:
        print_error(
            "No NLU model found. Train a model before running the "
            "server using `rasa train nlu`."
        )
        return

    telemetry.track_shell_started("nlu")
    rasa.nlu.run.run_cmdline(nlu_model)


def shell(args: argparse.Namespace):
    from rasa.cli.utils import get_validated_path
    from rasa.shared.constants import DEFAULT_MODELS_PATH
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

    if not core_model:
        import rasa.nlu.run

        telemetry.track_shell_started("nlu")

        rasa.nlu.run.run_cmdline(nlu_model)
    else:
        import rasa.cli.run

        telemetry.track_shell_started("rasa")

        rasa.cli.run.run(args)
