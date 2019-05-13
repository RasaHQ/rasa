import argparse
import logging
import os
from typing import List

from rasa.cli.arguments import run as arguments
from rasa.cli.utils import get_validated_path
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
        help="Start a Rasa server which loads a trained model.",
    )
    run_parser.set_defaults(func=run)

    arguments.set_run_arguments(run_parser)

    run_subparsers = run_parser.add_subparsers()
    sdk_subparser = run_subparsers.add_parser(
        "actions",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run the action server",
    )
    sdk_subparser.set_defaults(func=run_actions)

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


def run(args: argparse.Namespace):
    import rasa.run

    args.model = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)
    args.endpoints = get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    args.credentials = get_validated_path(
        args.credentials, "credentials", DEFAULT_CREDENTIALS_PATH, True
    )

    rasa.run(**vars(args))
