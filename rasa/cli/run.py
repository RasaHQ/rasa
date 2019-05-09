import argparse
import logging
import os
from typing import List

from rasa.cli.arguments.default_arguments import add_model_param
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
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run)

    run_subparsers = run_parser.add_subparsers()

    sdk_subparser = run_subparsers.add_parser(
        "actions",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run the action server",
    )
    _adk_sdk_arguments(sdk_subparser)
    sdk_subparser.set_defaults(func=run_actions)


def add_run_arguments(parser: argparse.ArgumentParser):
    from rasa.cli.arguments.run import add_run_arguments
    from rasa.cli.arguments.arguments import add_logging_option_arguments

    add_run_arguments(parser)
    add_model_param(parser)
    add_logging_option_arguments(parser)


def _adk_sdk_arguments(parser: argparse.ArgumentParser):
    import rasa_core_sdk.cli.arguments as sdk

    sdk.add_endpoint_arguments(parser)
    parser.add_argument(
        "--actions",
        type=str,
        default="actions",
        help="name of action package to be loaded",
    )


def run_actions(args: argparse.Namespace):
    import rasa_core_sdk.endpoint as sdk
    import sys

    args.actions = args.actions or DEFAULT_ACTIONS_PATH

    # insert current path in syspath so module is found
    sys.path.insert(1, os.getcwd())
    path = args.actions.replace(".", os.sep) + ".py"
    _ = get_validated_path(path, "action", DEFAULT_ACTIONS_PATH)

    sdk.main(args)


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
