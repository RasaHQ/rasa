from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import tempfile
import zipfile
from functools import wraps

from builtins import str
from flask import Flask, request, abort, Response, jsonify
from flask_cors import CORS, cross_origin
from typing import Union, Text, Optional, List

from rasa_core import events, utils
from rasa_core.actions.action import EndpointConfig
from rasa_core.agent import Agent
from rasa_core.channels import CollectingOutputChannel, InputChannel
from rasa_core.channels import channel
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.tracker_store import TrackerStore
from rasa_core.trackers import DialogueStateTracker
from rasa_core.version import __version__

from typing import Union
import typing

if typing.TYPE_CHECKING:
    from rasa_core.interpreter import NaturalLanguageInterpreter as NLI

logger = logging.getLogger(__name__)


def ensure_loaded_agent(agent):
    """Wraps a request handler ensuring there is a loaded and usable model."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            __agent = agent()
            if not __agent:
                return Response(
                        "No agent loaded. To continue processing, a model "
                        "of a trained agent needs to be loaded.",
                        status=503)

            return f(*args, **kwargs)

        return decorated

    return decorator


def request_parameters():
    if request.method == 'GET':
        return request.args
    else:

        try:
            return request.get_json(force=True)
        except ValueError as e:
            logger.error("Failed to decode json during respond request. "
                         "Error: {}.".format(e))
            raise


def requires_auth(token=None):
    # type: (Optional[Text]) -> function
    """Wraps a request handler with token authentication."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            provided = request.args.get('token')
            if token is None or provided == token:
                return f(*args, **kwargs)
            abort(401)

        return decorated

    return decorator


def _create_agent(
        model_directory,  # type: Text
        interpreter,  # type: Union[Text, NaturalLanguageInterpreter, None]
        action_endpoint=None,  # type: Optional[EndpointConfig]
        tracker_store=None,  # type: Optional[TrackerStore]
        nlg_endpoint=None
):
    # type: (...) -> Optional[Agent]
    try:

        return Agent.load(model_directory, interpreter,
                          tracker_store=tracker_store,
                          action_endpoint=action_endpoint,
                          generator=nlg_endpoint)
    except Exception as e:
        logger.error("Failed to load any agent model. Running "
                     "Rasa Core server with out loaded model now. {}"
                     "".format(e))
        return None


def create_app(model_directory, # type: Text
               interpreter=None, # type: Union[Text, NLI, None]
               input_channels=None,  # type: Optional[List[InputChannel]]
               cors_origins=None, # type: Optional[List[Text]]
               auth_token=None,  # type: Optional[Text]
               tracker_store=None,  # type: Optional[TrackerStore]
               action_endpoint=None,
               nlg_endpoint=None
               ):
    """Class representing a Rasa Core HTTP server."""

    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    if not cors_origins:
        cors_origins = []

    # this needs to be an array, so we can modify it in the nested functions...
    _agent = [_create_agent(model_directory, interpreter, action_endpoint,
                            tracker_store, nlg_endpoint)]

    def agent():
        if _agent and _agent[0]:
            return _agent[0]
        else:
            return None

    def handle_message(text_mesage):
        def noop(text_message):
            logger.info("Ignoring message as there is no agent to handle it.")

        a = agent()
        if a:
            return a.handle_message(text_mesage)
        else:
            return noop(text_mesage)

    @app.route("/",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    def hello():
        """Check if the server is running and responds with the version."""
        return "hello from Rasa Core: " + __version__

    @app.route("/version",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    def version():
        """respond with the version number of the installed rasa core."""

        return jsonify({'version': __version__})

    @app.route("/conversations/<sender_id>/tracker/events",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def append_events(sender_id):
        """Append a list of events to the state of a conversation"""

        request_params = request.get_json(force=True)
        evts = events.deserialise_events(request_params)
        tracker = agent().tracker_store.get_or_create_tracker(sender_id)
        for e in evts:
            tracker.update(e)
        agent().tracker_store.save(tracker)
        return jsonify(tracker.current_state())

    @app.route("/conversations",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def list_trackers():
        return jsonify(list(agent().tracker_store.keys()))

    @app.route("/conversations/<sender_id>/tracker",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def retrieve_tracker(sender_id):
        """Get a dump of a conversations tracker including its events."""

        # parameters
        use_history = utils.bool_arg('ignore_restarts', default=False)
        should_include_events = utils.bool_arg('events', default=True)
        until_time = request.args.get('until', None)

        # retrieve tracker and set to requested state
        tracker = agent().tracker_store.get_or_create_tracker(sender_id)
        if until_time is not None:
            tracker = tracker.travel_back_in_time(float(until_time))

        # dump and return tracker
        state = tracker.current_state(
                should_include_events=should_include_events,
                only_events_after_latest_restart=use_history)
        return jsonify(state)

    @app.route("/conversations/<sender_id>/tracker",
               methods=['PUT', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def update_tracker(sender_id):
        """Use a list of events to set a conversations tracker to a state."""

        request_params = request.get_json(force=True)
        tracker = DialogueStateTracker.from_dict(sender_id,
                                                 request_params,
                                                 agent().domain.slos)
        agent().tracker_store.save(tracker)

        # will override an existing tracker with the same id!
        agent().tracker_store.save(tracker)
        return jsonify(tracker.current_state(should_include_events=True))

    @app.route("/conversations/<sender_id>/respond",
               methods=['GET', 'POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def respond(sender_id):
        request_params = request_parameters()

        if 'query' in request_params:
            message = request_params.pop('query')
        elif 'q' in request_params:
            message = request_params.pop('q')
        else:
            return Response(jsonify(error="Invalid respond parameter "
                                          "specified."),
                            status=400,
                            mimetype="application/json")

        try:
            # Set the output channel
            out = CollectingOutputChannel()
            # Fetches the appropriate bot response in a json format
            responses = agent().handle_message(message,
                                               output_channel=out,
                                               sender_id=sender_id)
            return jsonify(responses)

        except Exception as e:
            logger.exception("Caught an exception during respond.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/load", methods=['POST', 'OPTIONS'])
    @requires_auth(auth_token)
    @cross_origin(origins=cors_origins)
    def load_model():
        """Loads a zipped model, replacing the existing one."""

        if 'model' not in request.files:
            # model file is missing
            abort(400)

        model_file = request.files['model']

        logger.info("Received new model through REST interface.")
        zipped_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        zipped_path.close()

        model_file.save(zipped_path.name)

        logger.debug("Downloaded model to {}".format(zipped_path.name))

        zip_ref = zipfile.ZipFile(zipped_path.name, 'r')
        zip_ref.extractall(model_directory)
        zip_ref.close()
        logger.debug("Unzipped model to {}".format(
                os.path.abspath(model_directory)))

        _agent[0] = _create_agent(model_directory, interpreter, tracker_store)
        logger.debug("Finished loading new agent.")
        return jsonify({'success': 1})

    if input_channels:
        channel.register_blueprints(input_channels, app, handle_message,
                                    route="/webhooks/")
    return app


if __name__ == '__main__':
    from rasa_core import run
    # Running as standalone python application
    arg_parser = run.create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARN)

    # Setting up the color scheme of logger
    utils.configure_colored_logging(cmdline_args.loglevel)
    utils.configure_file_logging(cmdline_args.loglevel,
                                 cmdline_args.log_file)

    logger.info("Rasa process starting")

    # Setting up the rasa_core application framework
    # Running the server at 'this' address with the
    # rasa_core application framework
    run.start_server(cmdline_args.core,
                     cmdline_args.nlu,
                     cmdline_args.connector,
                     cmdline_args.port,
                     cmdline_args.credentials,
                     cmdline_args.cors)
