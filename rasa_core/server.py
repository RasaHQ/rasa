from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging

from builtins import str
from klein import Klein
from typing import Union, Text, Optional

from rasa_core import utils, events
from rasa_core.agent import Agent
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.trackers import DialogueStateTracker
from rasa_core.version import __version__
from rasa_nlu.server import check_cors, requires_auth

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the server script."""

    parser = argparse.ArgumentParser(
            description='starts server to serve an agent')
    parser.add_argument(
            '-d', '--core',
            required=True,
            type=str,
            help="core model to run with the server")
    parser.add_argument(
            '-u', '--nlu',
            type=str,
            help="nlu model to run with the server")
    parser.add_argument(
            '-p', '--port',
            type=int,
            default=5005,
            help="port to run the server at")
    parser.add_argument(
            '--cors',
            nargs='*',
            type=str,
            help="enable CORS for the passed origin. "
                 "Use * to whitelist all origins")
    parser.add_argument(
            '--auth_token',
            type=str,
            help="Enable token based authentication. Requests need to provide "
                 "the token to be accepted.")
    parser.add_argument(
            '-o', '--log_file',
            type=str,
            default="rasa_core.log",
            help="store log file in specified file")

    utils.add_logging_option_arguments(parser)
    return parser


def _configure_logging(loglevel, logfile):
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(loglevel)
        logging.getLogger('').addHandler(fh)
    logging.captureWarnings(True)


class RasaCoreServer(object):
    """Class representing a Rasa Core HTTP server."""

    app = Klein()

    def __init__(self, model_directory,
                 interpreter=None,
                 loglevel="INFO",
                 logfile="rasa_core.log",
                 cors_origins=None,
                 action_factory=None,
                 auth_token=None):

        _configure_logging(loglevel, logfile)

        self.config = {"cors_origins": cors_origins if cors_origins else [],
                       "token": auth_token}
        self.agent = self._create_agent(model_directory, interpreter,
                                        action_factory)

    @staticmethod
    def _create_agent(
            model_directory,  # type: Text
            interpreter,  # type: Union[Text, NaturalLanguageInterpreter]
            action_factory=None  # type: Optional[Text]
    ):
        # type: (...) -> Agent
        return Agent.load(model_directory, interpreter,
                          action_factory=action_factory)

    @app.route("/",
               methods=['GET', 'OPTIONS'])
    @check_cors
    def hello(self, request):
        """Check if the server is running and responds with the version."""
        return "hello from Rasa Core: " + __version__

    @app.route("/conversations/<sender_id>/continue",
               methods=['POST', 'OPTIONS'])
    @check_cors
    @requires_auth
    def continue_predicting(self, request, sender_id):
        """Continue a prediction started with parse.

        Caller should have executed the action returned from the parse
        endpoint. The events returned from that executed action are
        passed to continue which will trigger the next action prediction.

        If continue predicts action listen, the caller should wait for the
        next user message."""

        request.setHeader('Content-Type', 'application/json')
        request_params = json.loads(
                request.content.read().decode('utf-8', 'strict'))
        encoded_events = request_params.get("events", [])
        executed_action = request_params.get("executed_action", None)
        evts = events.deserialise_events(encoded_events, self.agent.domain)
        try:
            response = self.agent.continue_message_handling(sender_id,
                                                            executed_action,
                                                            evts)
        except ValueError as e:
            request.setResponseCode(400)
            return json.dumps({"error": e.message})
        except Exception as e:
            request.setResponseCode(500)
            logger.exception(e)
            return json.dumps({"error": "Server failure. Error: {}".format(e)})
        return json.dumps(response)

    @app.route("/conversations/<sender_id>/tracker/events",
               methods=['POST', 'OPTIONS'])
    @check_cors
    def append_events(self, request, sender_id):
        """Append a list of events to the state of a conversation"""

        request.setHeader('Content-Type', 'application/json')
        request_params = json.loads(
                request.content.read().decode('utf-8', 'strict'))
        evts = events.deserialise_events(request_params, self.agent.domain)
        tracker = self.agent.tracker_store.get_or_create_tracker(sender_id)
        for e in evts:
            tracker.update(e)
        self.agent.tracker_store.save(tracker)
        return json.dumps(tracker.current_state())

    @app.route("/conversations/<sender_id>/tracker",
               methods=['GET', 'OPTIONS'])
    @check_cors
    def retrieve_tracker(self, request, sender_id):
        """Get a dump of a conversations tracker including its events."""

        request.setHeader('Content-Type', 'application/json')
        tracker = self.agent.tracker_store.get_or_create_tracker(sender_id)
        return json.dumps(tracker.current_state(should_include_events=True))

    @app.route("/conversations/<sender_id>/tracker",
               methods=['PUT', 'OPTIONS'])
    @check_cors
    def update_tracker(self, request, sender_id):
        """Use a list of events to set a conversations tracker to a state."""

        request.setHeader('Content-Type', 'application/json')
        request_params = json.loads(
                request.content.read().decode('utf-8', 'strict'))
        tracker = DialogueStateTracker.from_dict(sender_id,
                                                 request_params,
                                                 self.agent.domain)
        self.agent.tracker_store.save(tracker)

        # will override an existing tracker with the same id!
        self.agent.tracker_store.save(tracker)
        return json.dumps(tracker.current_state(should_include_events=True))

    @app.route("/conversations/<sender_id>/parse",
               methods=['GET', 'POST', 'OPTIONS'])
    @check_cors
    @requires_auth
    def parse(self, request, sender_id):
        request.setHeader('Content-Type', 'application/json')
        if request.method.decode('utf-8', 'strict') == 'GET':
            request_params = {
                key.decode('utf-8', 'strict'): value[0].decode('utf-8',
                                                               'strict')
                for key, value in request.args.items()}
        else:
            request_params = json.loads(
                    request.content.read().decode('utf-8', 'strict'))

        if 'query' in request_params:
            message = request_params.pop('query')
        elif 'q' in request_params:
            message = request_params.pop('q')
        else:
            request.setResponseCode(400)
            return json.dumps({"error": "Invalid parse parameter specified"})

        try:
            response = self.agent.start_message_handling(message, sender_id)
            request.setResponseCode(200)
            return json.dumps(response)
        except Exception as e:
            request.setResponseCode(500)
            logger.error("Caught an exception during "
                         "parse: {}".format(e), exc_info=1)
            return json.dumps({"error": "{}".format(e)})

    @app.route("/version",
               methods=['GET', 'OPTIONS'])
    @check_cors
    def version(self, request):
        """Respond with the version number of the installed Rasa Core."""

        request.setHeader('Content-Type', 'application/json')
        return json.dumps({'version': __version__})


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.basicConfig(level=cmdline_args.loglevel)

    rasa = RasaCoreServer(cmdline_args.core,
                          cmdline_args.nlu,
                          cmdline_args.loglevel,
                          cmdline_args.log_file,
                          cmdline_args.cors,
                          auth_token=cmdline_args.auth_token)

    logger.info("Started http server on port %s" % cmdline_args.port)
    rasa.app.run("0.0.0.0", cmdline_args.port)
