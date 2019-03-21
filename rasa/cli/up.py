import argparse
import logging
import sys
from multiprocessing import Process
from typing import List, Text

import rasa.cli.run
from rasa.cli.utils import print_error, print_success
from rasa.core import utils, cli
from rasa.core.run import serve_application
from rasa.core.utils import AvailableEndpoints, EndpointConfig
from rasa.utils import configure_colored_logging


# noinspection PyProtectedMember
def add_subparser(subparsers: argparse._SubParsersAction,
                  parents: List[argparse.ArgumentParser]):
    shell_parser = subparsers.add_parser(
        "up",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run the Rasa Interface")

    shell_parser.add_argument(
        "--production",
        action="store_true",
        help="Run Rasa in a production environment")

    shell_parser.add_argument(
        "--auth_token",
        type=str,
        help="Rasa API auth token")

    shell_parser.add_argument(
        "--nlg",
        type=str,
        default="http://localhost:5002/api/nlg",
        help="Rasa NLG endpoint")

    shell_parser.add_argument(
        "--model_endpoint_url",
        type=str,
        default=("http://localhost:5002/api/projects/"
                 "default/models/tags/production"),
        help="Rasa Stack model endpoint URL")

    rasa.cli.run.add_run_arguments(shell_parser)

    shell_parser.set_defaults(func=up)

    cli.arguments.add_logging_option_arguments(shell_parser)


def start_core(args: argparse.Namespace,
               endpoints: AvailableEndpoints = None):
    """Starts the Rasa Core application."""

    if endpoints is None:
        endpoints = AvailableEndpoints.read_endpoints(args.endpoints)

    serve_application(endpoints=endpoints,
                      port=args.port,
                      credentials=args.credentials,
                      cors=args.cors,
                      auth_token=args.auth_token,
                      enable_api=True,
                      jwt_secret=args.jwt_secret,
                      jwt_method=args.jwt_method)


def start_core_for_local_platform(args: argparse.Namespace,
                                  platform_token: Text):
    """Starts the Rasa API with Rasa Core as a background process."""

    endpoints = AvailableEndpoints(
        model=EndpointConfig(args.model_endpoint_url,
                             token=platform_token,
                             wait_time_between_pulls=1),
        event_broker=EndpointConfig(**{"type": "file"}),
        nlg=EndpointConfig(args.nlg, token=platform_token))

    vars(args).update(dict(nlu_model=None,
                           channel="rasa",
                           credentials="credentials.yml",
                           cors="*",
                           auth_token=args.auth_token,
                           enable_api=True,
                           endpoints=endpoints))

    p = Process(target=start_core, args=(args, endpoints))
    p.start()


def up(args: argparse.Namespace):
    logging.getLogger('werkzeug').setLevel(logging.WARN)
    logging.getLogger('engineio').setLevel(logging.WARN)
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger('socketio').setLevel(logging.ERROR)

    configure_colored_logging(args.loglevel)
    utils.configure_file_logging(args.loglevel,
                                 args.log_file)

    if args.production:
        print_success("Starting Rasa Core")
        print_success("have args: {}".format(args))
        start_core(args)
    else:
        try:
            from rasa_platform import config
            from rasa_platform.api.server import main_local
            from rasa_platform.services.event_service import main
        except ImportError:
            print_error("Rasa Platform is not installed. The `rasa up` "
                        "command requires an installation of Rasa Platform.")
            sys.exit()

        print_success("Starting Rasa Event Service")
        print_success("Starting Rasa Core")
        print_success("Starting Rasa Interface")

        p = Process(target=main, args=("rasa_event.log",))
        p.start()

        start_core_for_local_platform(args, config.platform_token)

        main_local(".")
