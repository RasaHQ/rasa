import argparse
import logging
import os
import shutil
from typing import List

from rasa.cli.default_arguments import add_model_param
from rasa.cli.utils import validate, check_path_exists
from rasa.cli.constants import (DEFAULT_ENDPOINTS_PATH,
                                DEFAULT_ACTIONS_PATH, DEFAULT_CREDENTIALS_PATH)
from rasa.model import DEFAULT_MODELS_PATH, get_latest_model, get_model

logger = logging.getLogger(__name__)


def add_subparser(subparsers: argparse._SubParsersAction,
                  parents: List[argparse.ArgumentParser]):
    run_parser = subparsers.add_parser(
        "run",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run a trained model")
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run)

    run_subparsers = run_parser.add_subparsers()
    run_core_parser = run_subparsers.add_parser(
        "core",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run a trained Core model"
    )
    run_core_parser.set_defaults(func=run)

    nlu_subparser = run_subparsers.add_parser(
        "nlu",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run a trained NLU model"
    )

    _add_nlu_arguments(nlu_subparser)
    nlu_subparser.set_defaults(func=run_nlu)

    sdk_subparser = run_subparsers.add_parser(
        "actions",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run the action server"
    )
    _adk_sdk_arguments(sdk_subparser)
    add_run_arguments(run_core_parser)
    sdk_subparser.set_defaults(func=run_actions)


def add_run_arguments(parser: argparse.ArgumentParser):
    from rasa_core.cli.run import add_run_arguments

    add_run_arguments(parser)
    add_model_param(parser)

    parser.add_argument(
        "--credentials",
        type=str,
        default="credentials.yml",
        help="Authentication credentials for the connector as a yml file")


def _add_nlu_arguments(parser: argparse.ArgumentParser):
    from rasa_nlu.cli.server import add_server_arguments

    add_server_arguments(parser)
    parser.add_argument('--path',
                        default=DEFAULT_MODELS_PATH,
                        type=str,
                        help="working directory of the server. Models are"
                             "loaded from this directory and trained models "
                             "will be saved here.")

    add_model_param(parser, "NLU")


def _adk_sdk_arguments(parser: argparse.ArgumentParser):
    import rasa_core_sdk.cli.arguments as sdk

    sdk.add_endpoint_arguments(parser)
    parser.add_argument(
        '--actions',
        type=str,
        default="actions",
        help="name of action package to be loaded")


def run_nlu(args: argparse.Namespace):
    import rasa_nlu.server
    import tempfile

    validate(args, [("path", DEFAULT_MODELS_PATH)])
    args.model = args.path

    model = get_latest_model(args.model)
    working_directory = tempfile.mkdtemp()
    model_path = model.unpack_model(model, working_directory)
    args.path = os.path.dirname(model_path)

    rasa_nlu.server.main(args)

    shutil.rmtree(model_path)


def run_actions(args: argparse.Namespace):
    import rasa_core_sdk.endpoint as sdk
    import sys

    args.actions = args.actions or DEFAULT_ACTIONS_PATH

    # insert current path in syspath so module is found
    sys.path.insert(1, os.getcwd())
    path = args.actions.replace('.', '/') + ".py"
    check_path_exists(path, "action", DEFAULT_ACTIONS_PATH)

    sdk.run(args)


def run(args: argparse.Namespace):
    from rasa_core.broker import PikaProducer
    from rasa_core.interpreter import RasaNLUInterpreter
    import rasa_core.run
    from rasa_core.tracker_store import TrackerStore
    from rasa_core.utils import AvailableEndpoints

    validate(args, [("model", DEFAULT_MODELS_PATH),
                    ("endpoints", DEFAULT_ENDPOINTS_PATH, True),
                    ("credentials", DEFAULT_CREDENTIALS_PATH, True)])

    model_paths = get_model(args.model, subdirectories=True)

    model_path, core_path, nlu_path = model_paths
    _endpoints = AvailableEndpoints.read_endpoints(args.endpoints)

    _interpreter = None
    if os.path.exists(nlu_path):
        _interpreter = RasaNLUInterpreter(model_directory=nlu_path)
    else:
        _interpreter = None
        logging.info("No NLU model found. Running without NLU.")

    _broker = PikaProducer.from_endpoint_config(_endpoints.event_broker)

    _tracker_store = TrackerStore.find_tracker_store(None,
                                                     _endpoints.tracker_store,
                                                     _broker)
    _agent = rasa_core.run.load_agent(core_path,
                                      interpreter=_interpreter,
                                      tracker_store=_tracker_store,
                                      endpoints=_endpoints)

    if not args.connector and not args.credentials:
        channel = "cmdline"
        logger.info("No chat connector configured, falling back to the "
                    "command line. Use `rasa configure channel` to connect"
                    "the bot to e.g. facebook messenger.")
    else:
        channel = args.connector

    rasa_core.run.serve_application(_agent,
                                    channel,
                                    args.port,
                                    args.credentials,
                                    args.cors,
                                    args.auth_token,
                                    args.enable_api,
                                    args.jwt_secret,
                                    args.jwt_method)

    shutil.rmtree(model_path)
