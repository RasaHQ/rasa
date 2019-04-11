import argparse
import logging
import os
import shutil
from typing import List

from rasa import model
from rasa.cli.default_arguments import add_model_param
from rasa.cli.utils import get_validated_path
from rasa.constants import (
    DEFAULT_ACTIONS_PATH,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODELS_PATH,
)
from rasa.model import get_latest_model

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
        help="Start a Rasa server which loads a trained model",
    )
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run)

    run_subparsers = run_parser.add_subparsers()
    run_core_parser = run_subparsers.add_parser(
        "core",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run a trained Core model",
    )
    add_run_arguments(run_core_parser)
    run_core_parser.set_defaults(func=run)

    nlu_subparser = run_subparsers.add_parser(
        "nlu",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run a trained NLU model",
    )

    _add_nlu_arguments(nlu_subparser)
    nlu_subparser.set_defaults(func=run_nlu)

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
    from rasa.core.cli.run import add_run_arguments

    add_run_arguments(parser)
    add_model_param(parser)

    parser.add_argument(
        "--credentials",
        type=str,
        default="credentials.yml",
        help="Authentication credentials for the connector as a yml file",
    )


def _add_nlu_arguments(parser: argparse.ArgumentParser):
    from rasa.nlu.cli.server import add_server_arguments

    add_server_arguments(parser)
    parser.add_argument(
        "--path",
        default=DEFAULT_MODELS_PATH,
        type=str,
        help="Working directory of the server. Models are"
        "loaded from this directory and trained models "
        "will be saved here",
    )

    add_model_param(parser, "NLU")


def _adk_sdk_arguments(parser: argparse.ArgumentParser):
    import rasa_core_sdk.cli.arguments as sdk

    sdk.add_endpoint_arguments(parser)
    parser.add_argument(
        "--actions",
        type=str,
        default="actions",
        help="name of action package to be loaded",
    )


def run_nlu(args: argparse.Namespace):
    import rasa.nlu.server
    import tempfile

    args.model = get_validated_path(args.path, "path", DEFAULT_MODELS_PATH)

    model_archive = get_latest_model(args.model)
    working_directory = tempfile.mkdtemp()
    unpacked_model = model.unpack_model(model_archive, working_directory)
    args.path = os.path.dirname(unpacked_model)

    rasa.nlu.server.main(args)

    shutil.rmtree(unpacked_model)


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
