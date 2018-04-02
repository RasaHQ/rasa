from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import io
import json
import logging
import os
import tempfile
import zipfile
from functools import wraps

import six
from builtins import str
from klein import Klein
from typing import Union, Text, Optional

from rasa_core import utils, events
from rasa_core.agent import Agent
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.tracker_store import TrackerStore
from rasa_core.trackers import DialogueStateTracker
from rasa_core.version import __version__

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


def ensure_loaded_agent(f):
    """Wraps a request handler ensuring there is a loaded and usable model."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]

        if not self.agent:
            request.setResponseCode(503)
            return ("No agent loaded. To continue processing, a model of a "
                    "trained agent needs to be loaded.")

        return f(*args, **kwargs)

    return decorated


def bool_arg(request, name, default=True):
    # type: (Request, Text, bool) -> bool
    """Return a passed boolean argument of the request or a default.

    Checks the `name` parameter of the request if it contains a valid
    boolean value. If not, `default` is returned."""

    d = [str(default)]
    return request.args.get(name, d)[0].lower() == 'true'


def request_parameters(request):
    if request.method.decode('utf-8', 'strict') == 'GET':
        return {
            key.decode('utf-8', 'strict'): value[0].decode('utf-8',
                                                           'strict')
            for key, value in request.args.items()}
    else:
        content = request.content.read()
        try:
            return json.loads(content.decode('utf-8', 'strict'))
        except ValueError as e:
            logger.error("Failed to decode json during respond request. "
                         "Error: {}. Request content: "
                         "'{}'".format(e, content))
            raise


def check_cors(f):
    """Wraps a request handler with CORS headers checking."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]
        origin = request.getHeader('Origin')

        if origin:
            if '*' in self.config['cors_origins']:
                request.setHeader('Access-Control-Allow-Origin', '*')
            elif origin in self.config['cors_origins']:
                request.setHeader('Access-Control-Allow-Origin', origin)
            else:
                request.setResponseCode(403)
                return 'forbidden'

        if request.method.decode('utf-8', 'strict') == 'OPTIONS':
            return ''  # if this is an options call we skip running `f`
        else:
            return f(*args, **kwargs)

    return decorated


def requires_auth(f):
    """Wraps a request handler with token authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]
        if six.PY3:
            token = request.args.get(b'token', [b''])[0].decode("utf8")
        else:
            token = str(request.args.get('token', [''])[0])
        if self.config['token'] is None or token == self.config['token']:
            return f(*args, **kwargs)
        request.setResponseCode(401)
        return 'unauthorized'

    return decorated


class RasaCoreServer(object):
    """Class representing a Rasa Core HTTP server."""

    app = Klein()

    def __init__(self, model_directory,
                 interpreter=None,
                 loglevel="INFO",
                 logfile="rasa_core.log",
                 cors_origins=None,
                 action_factory=None,
                 auth_token=None,
                 tracker_store=None):

        _configure_logging(loglevel, logfile)

        self.config = {"cors_origins": cors_origins if cors_origins else [],
                       "token": auth_token}
        self.model_directory = model_directory
        self.interpreter = interpreter
        self.tracker_store = tracker_store
        self.action_factory = action_factory
        self.agent = self._create_agent(model_directory, interpreter,
                                        action_factory, tracker_store)

    @staticmethod
    def _create_agent(
            model_directory,  # type: Text
            interpreter,  # type: Union[Text, NaturalLanguageInterpreter]
            action_factory=None,  # type: Optional[Text]
            tracker_store=None  # type: Optional[TrackerStore]
    ):
        # type: (...) -> Optional[Agent]
        try:

            return Agent.load(model_directory, interpreter,
                              tracker_store=tracker_store,
                              action_factory=action_factory)
        except Exception as e:
            logger.warn("Failed to load any agent model. Running "
                        "Rasa Core server with out loaded model now. {}"
                        "".format(e))
            return None

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
    @ensure_loaded_agent
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
        evts = events.deserialise_events(encoded_events)
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
    @ensure_loaded_agent
    def append_events(self, request, sender_id):
        """Append a list of events to the state of a conversation"""

        request.setHeader('Content-Type', 'application/json')
        request_params = json.loads(
                request.content.read().decode('utf-8', 'strict'))
        evts = events.deserialise_events(request_params)
        tracker = self.agent.tracker_store.get_or_create_tracker(sender_id)
        for e in evts:
            tracker.update(e)
        self.agent.tracker_store.save(tracker)
        return json.dumps(tracker.current_state())

    @app.route("/conversations/<sender_id>/tracker",
               methods=['GET', 'OPTIONS'])
    @check_cors
    @ensure_loaded_agent
    def retrieve_tracker(self, request, sender_id):
        """Get a dump of a conversations tracker including its events."""

        # parameters
        use_history = bool_arg(request, 'ignore_restarts', default=False)
        should_include_events = bool_arg(request, 'events', default=True)
        until_time = request.args.get('until', None)

        # retrieve tracker and set to requested state
        tracker = self.agent.tracker_store.get_or_create_tracker(sender_id)
        if until_time is not None:
            tracker = tracker.travel_back_in_time(float(until_time))

        # dump and return tracker
        state = tracker.current_state(
                should_include_events=should_include_events,
                only_events_after_latest_restart=use_history)
        return json.dumps(state)

    @app.route("/conversations/<sender_id>/tracker",
               methods=['PUT', 'OPTIONS'])
    @check_cors
    @ensure_loaded_agent
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
    @ensure_loaded_agent
    def parse(self, request, sender_id):
        request.setHeader('Content-Type', 'application/json')
        request_params = request_parameters(request)

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

    @app.route("/conversations/<sender_id>/respond",
               methods=['GET', 'POST', 'OPTIONS'])
    @check_cors
    @requires_auth
    @ensure_loaded_agent
    def respond(self, request, sender_id):
        request.setHeader('Content-Type', 'application/json')
        request_params = request_parameters(request)

        if 'query' in request_params:
            message = request_params.pop('query')
        elif 'q' in request_params:
            message = request_params.pop('q')
        else:
            request.setResponseCode(400)
            return json.dumps({"error": "Invalid respond parameter specified"})

        try:
            out = CollectingOutputChannel()
            responses = self.agent.handle_message(message,
                                                  output_channel=out,
                                                  sender_id=sender_id)
            request.setResponseCode(200)
            return json.dumps(responses)

        except Exception as e:
            request.setResponseCode(500)
            logger.error("Caught an exception during "
                         "respond: {}".format(e), exc_info=1)
            return json.dumps({"error": "{}".format(e)})

    @app.route("/load", methods=['POST', 'OPTIONS'])
    @check_cors
    def load_model(self, request):
        """Loads a zipped model, replacing the existing one."""

        logger.info("Received new model through REST interface.")
        zipped_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        zipped_path.close()

        with io.open(zipped_path.name, 'wb') as f:
            f.write(request.args[b'model'][0])
        logger.debug("Downloaded model to {}".format(zipped_path.name))

        zip_ref = zipfile.ZipFile(zipped_path.name, 'r')
        zip_ref.extractall(self.model_directory)
        zip_ref.close()
        logger.debug("Unzipped model to {}".format(
                os.path.abspath(self.model_directory)))

        self.agent = self._create_agent(self.model_directory, self.interpreter,
                                        self.action_factory, self.tracker_store)
        logger.debug("Finished loading new agent.")
        return json.dumps({'success': 1})

    @app.route("/version",
               methods=['GET', 'OPTIONS'])
    @check_cors
    def version(self, request):
        """respond with the version number of the installed rasa core."""

        request.setHeader('content-type', 'application/json')
        return json.dumps({'version': __version__})


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    rasa = RasaCoreServer(cmdline_args.core,
                          cmdline_args.nlu,
                          cmdline_args.loglevel,
                          cmdline_args.log_file,
                          cmdline_args.cors,
                          auth_token=cmdline_args.auth_token)

    logger.info("Started http server on port %s" % cmdline_args.port)
    rasa.app.run("0.0.0.0", cmdline_args.port)
