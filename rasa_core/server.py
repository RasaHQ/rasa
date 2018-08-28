from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import tempfile
import zipfile
from flask import Flask, request, abort, Response, jsonify
from flask_cors import CORS, cross_origin
from functools import wraps
from typing import List
from typing import Text, Optional
from typing import Union

from rasa_core import utils, constants
from rasa_core.channels import (
    CollectingOutputChannel, UserMessage)
from rasa_core.events import Event
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.policies import PolicyEnsemble
from rasa_core.trackers import DialogueStateTracker
from rasa_core.version import __version__
from rasa_core.channels import UserMessage


logger = logging.getLogger(__name__)


def ensure_loaded_agent(agent):
    """Wraps a request handler ensuring there is a loaded and usable model."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not agent.is_ready():
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


def create_app(agent,
               cors_origins=None,  # type: Optional[Union[Text, List[Text]]]
               auth_token=None,  # type: Optional[Text]
               ):
    """Class representing a Rasa Core HTTP server."""

    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    cors_origins = cors_origins or []

    if not agent.is_ready():
        logger.info("The loaded agent is not ready to be used yet "
                    "(e.g. only the NLU interpreter is configured, "
                    "but no Core model is loaded). This is NOT AN ISSUE "
                    "some endpoints are not available until the agent "
                    "is ready though.")

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
            agent.execute_action(sender_id,
                                 action_to_execute,
                                 out)

            # retrieve tracker and set to requested state
            tracker = agent.tracker_store.get_or_create_tracker(sender_id)
            state = tracker.current_state(should_include_events=True)
            return jsonify({"tracker": state,
                            "messages": out.messages})

        except ValueError as e:
            return Response(jsonify(error="".format(e)),
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
        tracker = agent.tracker_store.get_or_create_tracker(sender_id)
        if evt:
            tracker.update(evt)
            agent.tracker_store.save(tracker)
        else:
            logger.warning(
                    "Append event called, but could not extract a "
                    "valid event. Request JSON: {}".format(request_params))
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
                                                 agent.domain.slots)
        # will override an existing tracker with the same id!
        agent.tracker_store.save(tracker)
        return jsonify(tracker.current_state(should_include_events=True))

    @app.route("/conversations",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    def list_trackers():
        if agent.tracker_store:
            return jsonify(list(agent.tracker_store.keys()))
        else:
            return jsonify([])

    @app.route("/conversations/<sender_id>/tracker",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    def retrieve_tracker(sender_id):
        """Get a dump of a conversations tracker including its events."""

        if not agent.tracker_store:
            return Response("No tracker store available.",
                            status=503)

        # parameters
        should_ignore_restarts = utils.bool_arg('ignore_restarts',
                                                default=False)
        should_include_events = utils.bool_arg('events',
                                               default=True)
        until_time = request.args.get('until', None)

        # retrieve tracker and set to requested state
        tracker = agent.tracker_store.get_or_create_tracker(sender_id)
        if not tracker:
            return Response("Could not retrieve tracker. Most likely "
                            "because there is no domain set on the agent.",
                            status=503)

        if until_time is not None:
            tracker = tracker.travel_back_in_time(float(until_time))

        # dump and return tracker
        state = tracker.current_state(
                should_include_events=should_include_events,
                should_ignore_restarts=should_ignore_restarts)
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
            responses = agent.handle_text(message,
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
            responses = agent.predict_next(sender_id)
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

        # TODO: implement properly for agent / bot
        if sender != "user":
            return Response(jsonify(error="Currently, only user messages can "
                                          "be passed to this endpoint. "
                                          "Messages of sender '{}' can not be "
                                          "handled. ".format(sender)),
                            status=500,
                            content_type="application/json")

        try:
            usermsg = UserMessage(message, None, sender_id, parse_data)
            responses = agent.log_message(usermsg)
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
        model_directory = tempfile.mkdtemp()

        model_file.save(zipped_path.name)

        logger.debug("Downloaded model to {}".format(zipped_path.name))

        zip_ref = zipfile.ZipFile(zipped_path.name, 'r')
        zip_ref.extractall(model_directory)
        zip_ref.close()
        logger.debug("Unzipped model to {}".format(
                os.path.abspath(model_directory)))

        ensemble = PolicyEnsemble.load(model_directory)
        agent.policy_ensemble = ensemble
        logger.debug("Finished loading new agent.")
        return jsonify({'success': 1})

    @app.route("/domain",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def get_domain():
        """Get current domain in yaml or json format."""

        accepts = request.headers.get("Accept", default="application/json")
        if accepts.endswith("json"):
            domain = agent.domain.as_dict()
            return jsonify(domain)
        elif accepts.endswith("yml"):
            domain_yaml = agent.domain.as_yaml()
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

    @app.route("/finetune",
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
                                                 agent.domain.slots)

        try:
            # Fetches the appropriate bot response in a json format
            agent.continue_training([tracker],
                                    epochs=epochs,
                                    batch_size=batch_size)
            return '', 204

        except Exception as e:
            logger.exception("Caught an exception during prediction.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/status", methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    def status():
        return jsonify({
            "model_fingerprint": agent.fingerprint,
            "is_ready": agent.is_ready()
        })

    @app.route("/predict", methods=['POST'])
    @requires_auth(auth_token)
    @cross_origin(origins=cors_origins)
    @ensure_loaded_agent(agent)
    def tracker_predict():
        """ Given a list of events, predicts the next action"""
        sender_id = UserMessage.DEFAULT_SENDER_ID
        request_params = request.get_json(force=True)
        for param in request_params:
            if param.get('event', None) is None:
                return Response(
                    """Invalid list of events provided.""",
                    status=400)
        tracker = DialogueStateTracker.from_dict(sender_id,
                                                 request_params,
                                                 agent.domain.slots)
        policy_ensemble = agent.policy_ensemble
        probabilities = policy_ensemble.probabilities_using_best_policy(tracker, agent.domain)
        probability_dict = {agent.domain.action_for_index(idx, agent.action_endpoint).name(): probability
                            for idx, probability in enumerate(probabilities)}
        return jsonify(probability_dict)

    return app


if __name__ == '__main__':
    # Running as standalone python application
    from rasa_core import run

    arg_parser = run.create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.getLogger('werkzeug').setLevel(logging.WARN)
    logging.getLogger('matplotlib').setLevel(logging.WARN)

    utils.configure_colored_logging(cmdline_args.loglevel)
    utils.configure_file_logging(cmdline_args.loglevel,
                                 cmdline_args.log_file)

    logger.warning("USING `rasa_core.server` is deprecated and will be "
                   "removed in the future. Use `rasa_core.run --enable_api` "
                   "instead.")

    logger.info("Rasa process starting")

    _endpoints = run.read_endpoints(cmdline_args.endpoints)
    _interpreter = NaturalLanguageInterpreter.create(cmdline_args.nlu,
                                                     _endpoints.nlu)
    _agent = run.load_agent(cmdline_args.core,
                            interpreter=_interpreter,
                            endpoints=_endpoints)

    run.serve_application(_agent,
                          cmdline_args.connector,
                          cmdline_args.port,
                          cmdline_args.credentials,
                          cmdline_args.cors,
                          cmdline_args.auth_token,
                          cmdline_args.enable_api)
