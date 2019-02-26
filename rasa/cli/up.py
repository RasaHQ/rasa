import argparse
from multiprocessing import Process
from threading import Thread
from typing import List

import rasa.cli.run
from rasa_core.utils import EndpointConfig, print_success


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
    import rasa_core
    from rasa_core.utils import AvailableEndpoints
    _endpoints = AvailableEndpoints(
        # TODO: LOCAL make endpoints more configurable, esp ports
        model=EndpointConfig("http://localhost:5002"
                             "/api/projects/default/models/core/tags/production",
                             token=platform_token,
                             wait_time_between_pulls=10),
        event_broker=EndpointConfig(**{"type": "file"}),
        nlg=EndpointConfig("http://localhost:5002"
                           "/api/nlg",
                           token=platform_token))

    from rasa_core.interpreter import NaturalLanguageInterpreter
    from rasa_core import broker
    _broker = broker.from_endpoint_config(_endpoints.event_broker)

    from rasa_core.tracker_store import TrackerStore
    _tracker_store = TrackerStore.find_tracker_store(
        None, _endpoints.tracker_store, _broker)
    from rasa_core.run import load_agent
    _agent = load_agent("models",
                        interpreter=None,
                        tracker_store=_tracker_store,
                        endpoints=_endpoints)
    from rasa_core.run import serve_application
    print_success("About to start core")
    serve_application(_agent,
                      "rasa",
                      5005,
                      "credentials.yml",
                      "*",
                      None,  # TODO: LOCAL configure auth token
                      True)


def start_event_service():
    from rasa_platform.services.event_service import main
    main("rasa_event.log")


def up(args: argparse.Namespace):
    from rasa_core.utils import print_success

    print_success("Starting Rasa Core")
    from rasa_platform import config
    p = Process(target=start_core, args=(config.platform_token,))
    p.start()

    p = Process(target=start_event_service)
    p.start()

    print_success("Starting Rasa Interface...")

    from rasa_platform.api.server import main_local
    main_local(".")
