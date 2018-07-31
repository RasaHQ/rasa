from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import tempfile
import zipfile
from functools import wraps

import typing
from flask import Flask, request, abort, Response, jsonify
from flask_cors import CORS, cross_origin
from typing import Union, Text, Optional, List

from rasa_core import utils, constants
from rasa_core.agent import Agent
from rasa_core.channels import (
    CollectingOutputChannel, InputChannel,
    UserMessage)
from rasa_core.channels import channel
from rasa_core.events import Event
from rasa_core.tracker_store import TrackerStore
from rasa_core.trackers import DialogueStateTracker
from rasa_core.utils import EndpointConfig
from rasa_core.version import __version__

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
        return request.args.to_dict()
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
        interpreter,  # type: Union[Text, NLI, None]
        action_endpoint=None,  # type: Optional[EndpointConfig]
        tracker_store=None,  # type: Optional[TrackerStore]
        generator=None
):
    # type: (...) -> Optional[Agent]
    try:

        return Agent.load(model_directory, interpreter,
                          tracker_store=tracker_store,
                          action_endpoint=action_endpoint,
                          generator=generator)
    except Exception as e:
        logger.error("Failed to load any agent model. Running "
                     "Rasa Core server with out loaded model now. {}"
                     "".format(e))
        return None


def create_app(model_directory=None,  # type: Optional[Text]
               interpreter=None,  # type: Union[Text, NLI, None]
               input_channels=None,  # type: Optional[List[InputChannel]]
               cors_origins=None,  # type: Optional[List[Text]]
               auth_token=None,  # type: Optional[Text]
               tracker_store=None,  # type: Optional[TrackerStore]
               endpoints=None
               ):
    """Class representing a Rasa Core HTTP server."""
    from rasa_core import run

    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    if not cors_origins:
        cors_origins = []

    nlg_endpoint = utils.read_endpoint_config(endpoints, "nlg")

    nlu_endpoint = utils.read_endpoint_config(endpoints, "nlu")

    action_endpoint = utils.read_endpoint_config(endpoints, "action_endpoint")

    _interpreter = run.interpreter_from_args(interpreter, nlu_endpoint)

    # this needs to be an array, so we can modify it in the nested functions...
    _agent = [_create_agent(model_directory, _interpreter, action_endpoint,
                            tracker_store, nlg_endpoint)]

    def agent():
        if _agent and _agent[0]:
            return _agent[0]
        else:
            return None

    def set_agent(a):
        _agent[0] = a

    app.set_agent = set_agent

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

        return jsonify({
            "version": __version__,
            "minimum_compatible_version": constants.MINIMUM_COMPATIBLE_VERSION
        })

    # <sender_id> can be be 'default' if there's only 1 client
    @app.route("/conversations/<sender_id>/execute",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def execute_action(sender_id):
        request_params = request.get_json(force=True)
        action_to_execute = request_params.get("action", None)

        try:
            out = CollectingOutputChannel()
            agent().execute_action(sender_id,
                                   action_to_execute,
                                   out)

            # retrieve tracker and set to requested state
            tracker = agent().tracker_store.get_or_create_tracker(sender_id)
            state = tracker.current_state(should_include_events=True)
            return jsonify({"tracker": state,
                            "messages": out.messages})

        except ValueError as e:
            return Response(jsonify(error=e.message),
                            status=400,
                            content_type="application/json")
        except Exception as e:
            logger.exception(e)
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/conversations/<sender_id>/tracker/events",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def append_event(sender_id):
        """Append a list of events to the state of a conversation"""

        request_params = request.get_json(force=True)
        evt = Event.from_parameters(request_params)
        tracker = agent().tracker_store.get_or_create_tracker(sender_id)
        tracker.update(evt)
        agent().tracker_store.save(tracker)
        return jsonify(tracker.current_state(should_include_events=True))

    @app.route("/conversations/<sender_id>/tracker/events",
               methods=['PUT'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def replace_events(sender_id):
        """Use a list of events to set a conversations tracker to a state."""

        request_params = request.get_json(force=True)
        tracker = DialogueStateTracker.from_dict(sender_id,
                                                 request_params,
                                                 agent().domain.slots)
        # will override an existing tracker with the same id!
        agent().tracker_store.save(tracker)
        return jsonify(tracker.current_state(should_include_events=True))

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

    @app.route("/conversations/<sender_id>/predict",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def predict(sender_id):
        try:
            # Fetches the appropriate bot response in a json format
            responses = agent().predict_next(sender_id)
            return jsonify(responses)

        except Exception as e:
            logger.exception("Caught an exception during prediction.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/conversations/<sender_id>/messages", methods=['POST'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def log_message(sender_id):
        request_params = request.get_json(force=True)
        message = request_params.get("text")
        sender = request_params.get("sender")
        parse_data = request_params.get("parse_data")

        # TODO: TB - implement properly for agent / bot
        if sender != "user":
            return Response(jsonify(error="Currently, onle user messages can "
                                          "be passed to this endpoint. "
                                          "Messages of sender '{}' can not be "
                                          "handled. ".format(sender)),
                            status=500,
                            content_type="application/json")

        try:
            usermsg = UserMessage(message, None, sender_id, parse_data)
            responses = agent().log_message(usermsg)
            return jsonify(responses)

        except Exception as e:
            logger.exception("Caught an exception while logging message.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/model", methods=['POST', 'OPTIONS'])
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

        _agent[0] = _create_agent(model_directory, interpreter, tracker_store,
                                  action_endpoint, nlg_endpoint)
        logger.debug("Finished loading new agent.")
        return jsonify({'success': 1})

    @app.route("/model/domain",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def get_domain():
        """Get current domain in yaml or json format."""
        accepts = request.headers.get("Accept", default="application/json")
        if accepts.endswith("json"):
            domain = agent().domain.as_dict()
            return jsonify(domain)
        elif accepts.endswith("yml"):
            domain_yaml = agent().domain.as_yaml()
            return Response(domain_yaml,
                            status=200,
                            content_type="application/x-yml")
        else:
            return Response(
                    """Invalid accept header. Domain can be provided 
                    as json ("Accept: application/json")  
                    or yml ("Accept: application/x-yml"). 
                    Make sure you've set the appropriate Accept header.""",
                    status=406)

    @app.route("/model/finetune",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def continue_training():
        request.headers.get("Accept")
        epochs = request.args.get("epochs", 30)
        batch_size = request.args.get("batch_size", 5)
        request_params = request.get_json(force=True)
        tracker = DialogueStateTracker.from_dict(UserMessage.DEFAULT_SENDER_ID,
                                                 request_params,
                                                 agent().domain.slots)

        try:
            # Fetches the appropriate bot response in a json format
            responses = agent().continue_training([tracker],
                                                  epochs=epochs,
                                                  batch_size=batch_size)
            return jsonify(responses)

        except Exception as e:
            logger.exception("Caught an exception during prediction.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    if input_channels:
        channel.register_blueprints(input_channels, app, handle_message,
                                    route="/webhooks/")
    return app
