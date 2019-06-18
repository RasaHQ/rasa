import argparse
import logging
import os
from typing import List, Text, Optional

from rasa.cli.arguments import run as arguments
from rasa.cli.utils import get_validated_path, print_error
from rasa.constants import (
    DEFAULT_ACTIONS_PATH,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODELS_PATH,
)

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
    import sys

    args.actions = args.actions or DEFAULT_ACTIONS_PATH

    # insert current path in syspath so module is found
    sys.path.insert(1, os.getcwd())
    path = args.actions.replace(".", os.sep) + ".py"
    _ = get_validated_path(path, "action", DEFAULT_ACTIONS_PATH)

    sdk.main_from_args(args)


def _validate_model_path(model_path: Text, parameter: Text, default: Text):

    if model_path is not None and not os.path.exists(model_path):
        reason_str = "'{}' not found.".format(model_path)
        if model_path is None:
            reason_str = "Parameter '{}' not set.".format(parameter)

        logger.debug(
            "{} Using default location '{}' instead.".format(reason_str, default)
        )

        os.makedirs(default, exist_ok=True)
        model_path = default

    return model_path


def run(args: argparse.Namespace):
    import rasa.run
    # botfront:start
    from botfront.utils import set_endpoints_credentials_args_from_remote
    set_endpoints_credentials_args_from_remote(args)
    # botfront:end

    args.model = _validate_model_path(args.model, "model", DEFAULT_MODELS_PATH)

    if not args.enable_api:
        # if the API is enabled you can start without a model as you can train a
        # model via the API once the server is up and running
        from rasa.model import get_model

        model_path = get_model(args.model)
        if not model_path:
            print_error(
                "No model found. Train a model before running the "
                "server using `rasa train`."
            )
            return

    args.endpoints = get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    args.credentials = get_validated_path(
        args.credentials, "credentials", DEFAULT_CREDENTIALS_PATH, True
    )

    rasa.run(**vars(args))
