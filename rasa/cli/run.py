import argparse
import logging
import os
from typing import List, Text

from rasa.cli.arguments import run as arguments
from rasa.cli.utils import get_validated_path, print_error
from rasa.constants import (
    DEFAULT_ACTIONS_PATH,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODELS_PATH,
    DOCS_BASE_URL,
)
from rasa.exceptions import ModelNotFound

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    run_parser = subparsers.add_parser(
        "run",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Starts a Rasa server with your trained model.",
    )
    run_parser.set_defaults(func=run)

    run_subparsers = run_parser.add_subparsers()
    sdk_subparser = run_subparsers.add_parser(
        "actions",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Runs the action server.",
    )
    sdk_subparser.set_defaults(func=run_actions)

    arguments.set_run_arguments(run_parser)
    arguments.set_run_action_arguments(sdk_subparser)


def run_actions(args: argparse.Namespace):
    import rasa_sdk.__main__ as sdk

    args.actions = args.actions or DEFAULT_ACTIONS_PATH

    sdk.main_from_args(args)


def _validate_model_path(model_path: Text, parameter: Text, default: Text):

    if model_path is not None and not os.path.exists(model_path):
        reason_str = f"'{model_path}' not found."
        if model_path is None:
            reason_str = f"Parameter '{parameter}' not set."

        logger.debug(f"{reason_str} Using default location '{default}' instead.")

        os.makedirs(default, exist_ok=True)
        model_path = default

    return model_path


def run(args: argparse.Namespace):
    import rasa.run

    args.endpoints = get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    args.credentials = get_validated_path(
        args.credentials, "credentials", DEFAULT_CREDENTIALS_PATH, True
    )

    if args.enable_api:
        if not args.remote_storage:
            args.model = _validate_model_path(args.model, "model", DEFAULT_MODELS_PATH)
        rasa.run(**vars(args))
        return

    # if the API is not enable you cannot start without a model
    # make sure either a model server, a remote storage, or a local model is
    # configured

    from rasa.model import get_model
    from rasa.core.utils import AvailableEndpoints

    # start server if remote storage is configured
    if args.remote_storage is not None:
        rasa.run(**vars(args))
        return

    # start server if model server is configured
    endpoints = AvailableEndpoints.read_endpoints(args.endpoints)
    model_server = endpoints.model if endpoints and endpoints.model else None
    if model_server is not None:
        rasa.run(**vars(args))
        return

    # start server if local model found
    args.model = _validate_model_path(args.model, "model", DEFAULT_MODELS_PATH)
    local_model_set = True
    try:
        get_model(args.model)
    except ModelNotFound:
        local_model_set = False

    if local_model_set:
        rasa.run(**vars(args))
        return

    print_error(
        "No model found. You have three options to provide a model:\n"
        "1. Configure a model server in the endpoint configuration and provide "
        "the configuration via '--endpoints'.\n"
        "2. Specify a remote storage via '--remote-storage' to load the model "
        "from.\n"
        "3. Train a model before running the server using `rasa train` and "
        "use '--model' to provide the model path.\n"
        "For more information check {}.".format(
            DOCS_BASE_URL + "/user-guide/configuring-http-api/"
        )
    )
