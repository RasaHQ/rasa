import argparse
import sys
from multiprocessing import Process
from typing import List

import rasa.cli.run
from rasa.cli.utils import print_error, print_success


# noinspection PyProtectedMember
def add_subparser(subparsers: argparse._SubParsersAction,
                  parents: List[argparse.ArgumentParser]):
    shell_parser = subparsers.add_parser(
        "up",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run the Rasa Interface")
    rasa.cli.run.add_run_arguments(shell_parser)
    shell_parser.set_defaults(func=up)


def start_core(platform_token):
    from rasa.core.utils import AvailableEndpoints
    from rasa.core.run import serve_application
    from rasa.core.utils import EndpointConfig

    _endpoints = AvailableEndpoints(
        # TODO: make endpoints more configurable, esp ports
        model=EndpointConfig("http://localhost:5002"
                             "/api/projects/default/models/tags/production",
                             token=platform_token,
                             wait_time_between_pulls=1),
        event_broker=EndpointConfig(**{"type": "file"}),
        nlg=EndpointConfig("http://localhost:5002"
                           "/api/nlg",
                           token=platform_token))

    serve_application("models",
                      nlu_model=None,
                      channel="rasa",
                      credentials_file="credentials.yml",
                      cors="*",
                      auth_token=None,  # TODO: configure auth token
                      enable_api=True,
                      endpoints=_endpoints)


def start_event_service():
    from rasa_platform.services.event_service import main
    main("rasa_event.log")


def up(args: argparse.Namespace):
    try:
        from rasa_platform import config
        from rasa_platform.api.server import main_local
    except ImportError:
        print_error("Rasa Platform is not installed. The `rasa up` command "
                    "requires an installation of Rasa Platform.")
        sys.exit()

    print_success("Starting Rasa Core")

    p = Process(target=start_core, args=(config.platform_token,))
    p.start()

    p = Process(target=start_event_service)
    p.start()

    print_success("Starting Rasa Interface...")

    main_local(".")
