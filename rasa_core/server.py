import logging
import os
import tempfile
import zipfile
from functools import wraps
from typing import List, Text, Optional, Union, Callable, Any

from flask import Flask, request, abort, Response, jsonify, json
from flask_cors import CORS, cross_origin
from flask_jwt_simple import JWTManager, view_decorators

import rasa_nlu
from rasa_core import utils, constants
from rasa_core.channels import CollectingOutputChannel, UserMessage
from rasa_core.evaluate import run_story_evaluation
from rasa_core.events import Event
from rasa_core.domain import Domain
from rasa_core.policies import PolicyEnsemble
from rasa_core.trackers import DialogueStateTracker, EventVerbosity
from rasa_core.version import __version__

logger = logging.getLogger(__name__)


def _docs(sub_url: Text) -> Text:
    """Create a url to a subpart of the docs."""
    return constants.DOCS_BASE_URL + sub_url


def ensure_loaded_agent(agent):
    """Wraps a request handler ensuring there is a loaded and usable model."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not agent.is_ready():
                return error(
                    503,
                    "NoAgent",
                    "No agent loaded. To continue processing, a "
                    "model of a trained agent needs to be loaded.",
                    help_url=_docs("/server.html#running-the-http-server"))

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


def requires_auth(app: Flask,
                  token: Optional[Text] = None
                  ) -> Callable[[Any], Any]:
    """Wraps a request handler with token authentication."""

    def decorator(f: Callable[[Any], Any]) -> Callable[[Any, Any], Any]:
        def sender_id_from_args(f: Callable[[Any], Any],
                                args: Any,
                                kwargs: Any) -> Optional[Text]:
            argnames = utils.arguments_of(f)
            try:
                sender_id_arg_idx = argnames.index("sender_id")
                if "sender_id" in kwargs:  # try to fetch from kwargs first
                    return kwargs["sender_id"]
                if sender_id_arg_idx < len(args):
                    return args[sender_id_arg_idx]
                return None
            except ValueError:
                return None

        def sufficient_scope(*args: Any,
                             **kwargs: Any) -> Optional[bool]:
            jwt_data = view_decorators._decode_jwt_from_headers()
            user = jwt_data.get("user", {})

            username = user.get("username", None)
            role = user.get("role", None)

            if role == "admin":
                return True
            elif role == "user":
                sender_id = sender_id_from_args(f, args, kwargs)
                return sender_id is not None and username == sender_id
            else:
                return False

        @wraps(f)
        def decorated(*args: Any, **kwargs: Any) -> Any:
            provided = request.args.get('token')
            # noinspection PyProtectedMember
            if token is not None and provided == token:
                return f(*args, **kwargs)
            elif app.config.get('JWT_ALGORITHM') is not None:
                if sufficient_scope(*args, **kwargs):
                    return f(*args, **kwargs)
                abort(error(
                    403, "NotAuthorized",
                    "User has insufficient permissions.",
                    help_url=_docs(
                        "/server.html#security-considerations")))
            elif token is None and app.config.get('JWT_ALGORITHM') is None:
                # authentication is disabled
                return f(*args, **kwargs)

            abort(error(
                401, "NotAuthenticated", "User is not authenticated.",
                help_url=_docs("/server.html#security-considerations")))

        return decorated

    return decorator


def error(status, reason, message, details=None, help_url=None):
    return Response(
        json.dumps({
            "version": __version__,
            "status": "failure",
            "message": message,
            "reason": reason,
            "details": details or {},
            "help": help_url,
            "code": status}),
        status=status,
        content_type="application/json")


def event_verbosity_parameter(default_verbosity):
    event_verbosity_str = request.args.get(
        'include_events', default=default_verbosity.name).upper()
    try:
        return EventVerbosity[event_verbosity_str]
    except KeyError:
        enum_values = ", ".join([e.name for e in EventVerbosity])
        abort(error(404, "InvalidParameter",
                    "Invalid parameter value for 'include_events'. "
                    "Should be one of {}".format(enum_values),
                    {"parameter": "include_events", "in": "query"}))


def create_app(agent,
               cors_origins: Optional[Union[Text, List[Text]]] = None,
               auth_token: Optional[Text] = None,
               jwt_secret: Optional[Text] = None,
               jwt_method: Optional[Text] = "HS256",
               ):
    """Class representing a Rasa Core HTTP server."""

    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    cors_origins = cors_origins or []

    # Setup the Flask-JWT-Simple extension
    if jwt_secret and jwt_method:
        # since we only want to check signatures, we don't actually care
        # about the JWT method and set the passed secret as either symmetric
        # or asymmetric key. jwt lib will choose the right one based on method
        app.config['JWT_SECRET_KEY'] = jwt_secret
        app.config['JWT_PUBLIC_KEY'] = jwt_secret
        app.config['JWT_ALGORITHM'] = jwt_method
        JWTManager(app)

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
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(agent)
    def execute_action(sender_id):
        request_params = request.get_json(force=True)

        # we'll accept both parameters to specify the actions name
        action_to_execute = (request_params.get("name") or
                             request_params.get("action"))

        policy = request_params.get("policy", None)
        confidence = request_params.get("confidence", None)
        verbosity = event_verbosity_parameter(EventVerbosity.AFTER_RESTART)

        try:
            out = CollectingOutputChannel()
            agent.execute_action(sender_id,
                                 action_to_execute,
                                 out,
                                 policy,
                                 confidence)

            # retrieve tracker and set to requested state
            tracker = agent.tracker_store.get_or_create_tracker(sender_id)
            state = tracker.current_state(verbosity)
            return jsonify({"tracker": state,
                            "messages": out.messages})

        except ValueError as e:
            return error(400, "ValueError", e)
        except Exception as e:
            logger.error("Encountered an exception while running action '{}'. "
                         "Bot will continue, but the actions events are lost. "
                         "Make sure to fix the exception in your custom "
                         "code.".format(action_to_execute))
            logger.debug(e, exc_info=True)
            return error(500, "ValueError",
                         "Server failure. Error: {}".format(e))

    @app.route("/conversations/<sender_id>/tracker/events",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(agent)
    def append_event(sender_id):
        """Append a list of events to the state of a conversation"""

        request_params = request.get_json(force=True)
        evt = Event.from_parameters(request_params)
        tracker = agent.tracker_store.get_or_create_tracker(sender_id)
        verbosity = event_verbosity_parameter(EventVerbosity.AFTER_RESTART)

        if evt:
            tracker.update(evt)
            agent.tracker_store.save(tracker)
            return jsonify(tracker.current_state(verbosity))
        else:
            logger.warning(
                "Append event called, but could not extract a "
                "valid event. Request JSON: {}".format(request_params))
            return error(400, "InvalidParameter",
                         "Couldn't extract a proper event from the request "
                         "body.",
                         {"parameter": "", "in": "body"})

    @app.route("/conversations/<sender_id>/tracker/events",
               methods=['PUT'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(agent)
    def replace_events(sender_id):
        """Use a list of events to set a conversations tracker to a state."""

        request_params = request.get_json(force=True)
        verbosity = event_verbosity_parameter(EventVerbosity.AFTER_RESTART)

        tracker = DialogueStateTracker.from_dict(sender_id,
                                                 request_params,
                                                 agent.domain.slots)
        # will override an existing tracker with the same id!
        agent.tracker_store.save(tracker)
        return jsonify(tracker.current_state(verbosity))

    @app.route("/conversations",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    def list_trackers():
        if agent.tracker_store:
            return jsonify(list(agent.tracker_store.keys()))
        else:
            return jsonify([])

    @app.route("/conversations/<sender_id>/tracker",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    def retrieve_tracker(sender_id):
        """Get a dump of a conversations tracker including its events."""

        if not agent.tracker_store:
            return error(503, "NoTrackerStore",
                         "No tracker store available. Make sure to configure "
                         "a tracker store when starting the server.")

        # parameters
        default_verbosity = EventVerbosity.AFTER_RESTART

        # this is for backwards compatibility
        if "ignore_restarts" in request.args:
            ignore_restarts = utils.bool_arg('ignore_restarts', default=False)
            if ignore_restarts:
                default_verbosity = EventVerbosity.ALL

        if "events" in request.args:
            include_events = utils.bool_arg('events', default=True)
            if not include_events:
                default_verbosity = EventVerbosity.NONE

        verbosity = event_verbosity_parameter(default_verbosity)

        # retrieve tracker and set to requested state
        tracker = agent.tracker_store.get_or_create_tracker(sender_id)
        if not tracker:
            return error(503,
                         "NoDomain",
                         "Could not retrieve tracker. Most likely "
                         "because there is no domain set on the agent.")

        until_time = utils.float_arg('until')
        if until_time is not None:
            tracker = tracker.travel_back_in_time(until_time)

        # dump and return tracker

        state = tracker.current_state(verbosity)
        return jsonify(state)

    @app.route("/conversations/<sender_id>/story",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    def retrieve_story(sender_id):
        """Get an end-to-end story corresponding to this conversation."""

        if not agent.tracker_store:
            return error(503, "NoTrackerStore",
                         "No tracker store available. Make sure to configure "
                         "a tracker store when starting the server.")

        # retrieve tracker and set to requested state
        tracker = agent.tracker_store.get_or_create_tracker(sender_id)
        if not tracker:
            return error(503,
                         "NoDomain",
                         "Could not retrieve tracker. Most likely "
                         "because there is no domain set on the agent.")

        until_time = utils.float_arg('until')
        if until_time is not None:
            tracker = tracker.travel_back_in_time(until_time)

        # dump and return tracker
        state = tracker.export_stories(e2e=True)
        return state

    @app.route("/conversations/<sender_id>/respond",
               methods=['GET', 'POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(agent)
    def respond(sender_id):
        request_params = request_parameters()

        if 'query' in request_params:
            message = request_params.pop('query')
        elif 'q' in request_params:
            message = request_params.pop('q')
        else:
            return error(400,
                         "InvalidParameter",
                         "Missing the message parameter.",
                         {"parameter": "query", "in": "query"})

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
            return error(500, "ActionException",
                         "Server failure. Error: {}".format(e))

    @app.route("/conversations/<sender_id>/predict",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(agent)
    def predict(sender_id):
        try:
            # Fetches the appropriate bot response in a json format
            responses = agent.predict_next(sender_id)
            return jsonify(responses)

        except Exception as e:
            logger.exception("Caught an exception during prediction.")
            return error(500, "PredictionException",
                         "Server failure. Error: {}".format(e))

    @app.route("/conversations/<sender_id>/messages",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(agent)
    def log_message(sender_id):
        request_params = request.get_json(force=True)
        try:
            message = request_params["message"]
        except KeyError:
            message = request_params.get("text")

        sender = request_params.get("sender")
        parse_data = request_params.get("parse_data")
        verbosity = event_verbosity_parameter(EventVerbosity.AFTER_RESTART)

        # TODO: implement properly for agent / bot
        if sender != "user":
            return error(500,
                         "NotSupported",
                         "Currently, only user messages can be passed "
                         "to this endpoint. Messages of sender '{}' "
                         "can not be handled. ".format(sender),
                         {"parameter": "sender", "in": "body"})

        try:
            usermsg = UserMessage(message, None, sender_id, parse_data)
            tracker = agent.log_message(usermsg)
            return jsonify(tracker.current_state(verbosity))

        except Exception as e:
            logger.exception("Caught an exception while logging message.")
            return error(500, "MessageException",
                         "Server failure. Error: {}".format(e))

    @app.route("/model",
               methods=['POST', 'OPTIONS'])
    @requires_auth(app, auth_token)
    @cross_origin(origins=cors_origins)
    def load_model():
        """Loads a zipped model, replacing the existing one."""

        if 'model' not in request.files:
            # model file is missing
            return error(400, "InvalidParameter",
                         "You did not supply a model as part of your request.",
                         {"parameter": "model", "in": "body"})

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

        domain_path = os.path.join(os.path.abspath(model_directory),
                                   "domain.yml")
        domain = Domain.load(domain_path)
        ensemble = PolicyEnsemble.load(model_directory)
        agent.update_model(domain, ensemble, None)
        logger.debug("Finished loading new agent.")
        return '', 204

    @app.route("/evaluate",
               methods=['POST', 'OPTIONS'])
    @requires_auth(app, auth_token)
    @cross_origin(origins=cors_origins)
    def evaluate_stories():
        """Evaluate stories against the currently loaded model."""
        tmp_file = rasa_nlu.utils.create_temporary_file(request.get_data(),
                                                        mode='w+b')
        use_e2e = utils.bool_arg('e2e', default=False)
        try:
            evaluation = run_story_evaluation(tmp_file, agent, use_e2e=use_e2e)
            return jsonify(evaluation)
        except ValueError as e:
            return error(400, "FailedEvaluation",
                         "Evaluation could not be created. Error: {}"
                         "".format(e))

    @app.route("/domain",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
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
            return error(406,
                         "InvalidHeader",
                         """Invalid accept header. Domain can be provided
                            as json ("Accept: application/json")
                            or yml ("Accept: application/x-yml").
                            Make sure you've set the appropriate Accept
                            header.""")

    @app.route("/finetune",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(agent)
    def continue_training():
        request.headers.get("Accept")
        epochs = request.args.get("epochs", 30)
        batch_size = request.args.get("batch_size", 5)
        request_params = request.get_json(force=True)
        sender_id = UserMessage.DEFAULT_SENDER_ID

        try:
            tracker = DialogueStateTracker.from_dict(sender_id,
                                                     request_params,
                                                     agent.domain.slots)
        except Exception as e:
            return error(400, "InvalidParameter",
                         "Supplied events are not valid. {}".format(e),
                         {"parameter": "", "in": "body"})

        try:
            # Fetches the appropriate bot response in a json format
            agent.continue_training([tracker],
                                    epochs=epochs,
                                    batch_size=batch_size)
            return '', 204

        except Exception as e:
            logger.exception("Caught an exception during prediction.")
            return error(500, "TrainingException",
                         "Server failure. Error: {}".format(e))

    @app.route("/status",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(app, auth_token)
    def status():
        return jsonify({
            "model_fingerprint": agent.fingerprint,
            "is_ready": agent.is_ready()
        })

    @app.route("/predict",
               methods=['POST', 'OPTIONS'])
    @requires_auth(app, auth_token)
    @cross_origin(origins=cors_origins)
    @ensure_loaded_agent(agent)
    def tracker_predict():
        """ Given a list of events, predicts the next action"""

        sender_id = UserMessage.DEFAULT_SENDER_ID
        request_params = request.get_json(force=True)
        verbosity = event_verbosity_parameter(EventVerbosity.AFTER_RESTART)

        try:
            tracker = DialogueStateTracker.from_dict(sender_id,
                                                     request_params,
                                                     agent.domain.slots)
        except Exception as e:
            return error(400, "InvalidParameter",
                         "Supplied events are not valid. {}".format(e),
                         {"parameter": "", "in": "body"})

        policy_ensemble = agent.policy_ensemble
        probabilities, policy = \
            policy_ensemble.probabilities_using_best_policy(tracker,
                                                            agent.domain)

        scores = [{"action": a, "score": p}
                  for a, p in zip(agent.domain.action_names, probabilities)]

        return jsonify({
            "scores": scores,
            "policy": policy,
            "tracker": tracker.current_state(verbosity)
        })

    return app


if __name__ == '__main__':
    raise RuntimeError("Calling `rasa_core.server` directly is "
                       "no longer supported. "
                       "Please use `rasa_core.run --enable_api` instead.")
