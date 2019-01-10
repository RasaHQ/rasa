import logging
import os
import shutil
import tempfile

from rasa_core.broker import PikaProducer
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.run import load_agent, serve_application, add_run_options
from rasa_core.tracker_store import TrackerStore
from rasa_core.utils import AvailableEndpoints

logger = logging.getLogger(__name__)


def add_subparser(subparsers):
    run_parser = subparsers.add_parser(
        'run',
        help='Run a trained model')
    run_parser.set_defaults(func=run)
    run_parser.add_argument('model',
                            help="path to a trained rasa model")
    add_run_options(run_parser)


def unpack_model(model_file, working_directory):
    import tarfile
    tar = tarfile.open(model_file)
    tar.extractall(working_directory)
    tar.close()
    logger.debug("Extracted model to '{}'".format(working_directory))
    return os.path.join(working_directory, "rasa_model")


def run(args):
    if not os.path.isfile(args.model):
        logger.error("Failed to load model. File '{}' does not exist."
                     "".format(os.path.abspath(args.model)))
        exit(1)

    working_directory = tempfile.mkdtemp()
    model_path = unpack_model(args.model, working_directory)

    _endpoints = AvailableEndpoints.read_endpoints(args.endpoints)
    _interpreter = RasaNLUInterpreter(
        model_directory=os.path.join(model_path, "nlu"))
    _broker = PikaProducer.from_endpoint_config(_endpoints.event_broker)

    _tracker_store = TrackerStore.find_tracker_store(
        None, _endpoints.tracker_store, _broker)
    _agent = load_agent(os.path.join(model_path, "core"),
                        interpreter=_interpreter,
                        tracker_store=_tracker_store,
                        endpoints=_endpoints)

    if not args.credentials:
        channel = "cmdline"
        print("No chat connector configured, falling back to the "
              "command line. Use `rasa configure channel` to connect"
              "the bot to e.g. facebook messenger.")
    else:
        channel = args.connector

    serve_application(_agent,
                      channel,
                      args.port,
                      args.credentials,
                      args.cors,
                      args.auth_token,
                      args.enable_api,
                      args.jwt_secret,
                      args.jwt_method)

    shutil.rmtree(working_directory)
