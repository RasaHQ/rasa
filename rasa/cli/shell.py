import argparse
import logging
import uuid
from pathlib import Path
from typing import List

from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import shell as arguments
from rasa.cli.arguments.default_arguments import add_sub_agents_param
from rasa.core.config.configuration import Configuration
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.exceptions import ModelNotFound
from rasa.model import get_local_model
from rasa.shared.constants import ASSISTANT_ID_KEY
from rasa.shared.data import TrainingType
from rasa.shared.utils.cli import print_error

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

    add_sub_agents_param(shell_parser)

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


def shell_nlu(args: argparse.Namespace) -> None:
    """Talk with an NLU only bot though the command line."""
    import rasa.nlu.run
    from rasa.cli.validation.config_path_validation import get_validated_path
    from rasa.shared.constants import DEFAULT_MODELS_PATH

    args.connector = "cmdline"

    model_path = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

    try:
        model = get_local_model(model_path)
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `rasa train nlu`."
        )
        return

    metadata = LocalModelStorage.metadata_from_archive(model)

    if metadata.assistant_id is None:
        print_error(
            f"The model metadata does not contain a value for the '{ASSISTANT_ID_KEY}' "
            f"attribute. Check that 'config.yml' file contains a value for "
            f"the '{ASSISTANT_ID_KEY}' key and re-train the model.",
        )

    if metadata.training_type == TrainingType.CORE:
        print_error(
            "No NLU model found. Train a model before running the "
            "server using `rasa train nlu`."
        )
        return

    telemetry.track_shell_started("nlu", metadata.assistant_id)
    rasa.nlu.run.run_cmdline(model)


def shell(args: argparse.Namespace) -> None:
    """Talk with a bot though the command line."""
    from rasa.cli.validation.config_path_validation import get_validated_path
    from rasa.shared.constants import DEFAULT_MODELS_PATH

    args.connector = "cmdline"
    # Load endpoints with proper endpoint file location
    # This will initialise the endpoints singleton properly so that
    # it can be used safely throughout the codebase with
    # `Configuration.get_instance().endpoints`
    Configuration.initialise_endpoints(endpoints_path=Path(args.endpoints))
    Configuration.initialise_sub_agents(args.sub_agents)
    model = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

    try:
        model = get_local_model(model)
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `rasa train`."
        )
        return

    metadata = LocalModelStorage.metadata_from_archive(model)

    if metadata.assistant_id is None:
        print_error(
            f"The model metadata does not contain a value for the '{ASSISTANT_ID_KEY}' "
            f"attribute. Check that 'config.yml' file contains a value for "
            f"the '{ASSISTANT_ID_KEY}' key and re-train the model.",
        )

    if metadata.training_type == TrainingType.NLU:
        import rasa.nlu.run

        telemetry.track_shell_started("nlu", metadata.assistant_id)

        rasa.nlu.run.run_cmdline(model)
    else:
        import rasa.cli.run

        telemetry.track_shell_started("rasa", metadata.assistant_id)

        rasa.cli.run.run(args)
