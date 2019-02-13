import argparse
import logging
import os
import shutil

from rasa.cli.default_arguments import add_model_param
from rasa.model import DEFAULT_MODELS_PATH, get_latest_model, get_model

logger = logging.getLogger(__name__)


def add_subparser(subparsers, parents):
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
    add_run_arguments(run_core_parser)
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
        "sdk",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run the action server"
    )

    sdk_subparser.set_defaults(func=run_sdk)


def add_run_arguments(parser):
    from rasa_core.cli.run import add_run_arguments

    add_run_arguments(parser)
    add_model_param(parser)

    parser.add_argument(
        "--credentials",
        default="credentials.yml",
        help="Authentication credentials for the connector as a yml file")


def _add_nlu_arguments(parser):
    from rasa_nlu.cli.server import add_server_arguments

    add_server_arguments(parser)
    parser.add_argument('--path',
                        default=DEFAULT_MODELS_PATH,
                        help="working directory of the server. Models are"
                             "loaded from this directory and trained models "
                             "will be saved here.")

    add_model_param(parser, "NLU")


def run_nlu(args):
    import rasa_nlu.server
    import tempfile

    model = get_latest_model(args.model)
    working_directory = tempfile.mkdtemp()
    model_path = model.unpack_model(model, working_directory)
    args.path = os.path.dirname(model_path)

    rasa_nlu.server.main(args)

    shutil.rmtree(model_path)


def run_sdk(args):
    print("Nothing here yet.")


def run(args):
    from rasa_core.broker import PikaProducer
    from rasa_core.interpreter import RasaNLUInterpreter
    import rasa_core.run
    from rasa_core.tracker_store import TrackerStore
    from rasa_core.utils import AvailableEndpoints

    model_paths = get_model(args.model, subdirectories=True)

    if model_paths is None:
        print("No model found for path '{}'.".format(args.model))

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
