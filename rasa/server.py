import logging
import os
import tempfile
import traceback
from functools import wraps
from inspect import isawaitable
from typing import Any, Callable, List, Optional, Text, Union

from sanic import Sanic, response
from sanic.request import Request
from sanic_cors import CORS
from sanic_jwt import Initialize, exceptions

import rasa
import rasa.utils.common
import rasa.utils.endpoints
import rasa.utils.io
from rasa.utils.endpoints import EndpointConfig
from rasa.constants import (
    MINIMUM_COMPATIBLE_VERSION,
    DEFAULT_MODELS_PATH,
    DEFAULT_DOMAIN_PATH,
)
from rasa.core import constants
from rasa.core.agent import load_agent, Agent
from rasa.core.channels import UserMessage, CollectingOutputChannel
from rasa.core.events import Event
from rasa.core.test import test
from rasa.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.core.utils import dump_obj_as_str_to_file
from rasa.model import get_model_subdirectories, fingerprint_from_path
from rasa.nlu.emulators.no_emulator import NoEmulator
from rasa.nlu.test import run_evaluation


logger = logging.getLogger(__name__)


class ErrorResponse(Exception):
    def __init__(self, status, reason, message, details=None, help_url=None):
        self.error_info = {
            "version": rasa.__version__,
            "status": "failure",
            "message": message,
            "reason": reason,
            "details": details or {},
            "help": help_url,
            "code": status,
        }
        self.status = status


def _docs(sub_url: Text) -> Text:
    """Create a url to a subpart of the docs."""
    return constants.DOCS_BASE_URL + sub_url


def ensure_loaded_agent(app: Sanic):
    """Wraps a request handler ensuring there is a loaded and usable agent."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not app.agent or not app.agent.is_ready():
                raise ErrorResponse(
                    409,
                    "Conflict",
                    "No agent loaded. To continue processing, a "
                    "model of a trained agent needs to be loaded.",
                    help_url=_docs("/server.html#running-the-http-server"),
                )

            return f(*args, **kwargs)

        return decorated

    return decorator


def requires_auth(app: Sanic, token: Optional[Text] = None) -> Callable[[Any], Any]:
    """Wraps a request handler with token authentication."""

    def decorator(f: Callable[[Any, Any, Any], Any]) -> Callable[[Any, Any], Any]:
        def conversation_id_from_args(args: Any, kwargs: Any) -> Optional[Text]:
            argnames = rasa.utils.common.arguments_of(f)

            try:
                sender_id_arg_idx = argnames.index("conversation_id")
                if "conversation_id" in kwargs:  # try to fetch from kwargs first
                    return kwargs["conversation_id"]
                if sender_id_arg_idx < len(args):
                    return args[sender_id_arg_idx]
                return None
            except ValueError:
                return None

        def sufficient_scope(request, *args: Any, **kwargs: Any) -> Optional[bool]:
            jwt_data = request.app.auth.extract_payload(request)
            user = jwt_data.get("user", {})

            username = user.get("username", None)
            role = user.get("role", None)

            if role == "admin":
                return True
            elif role == "user":
                conversation_id = conversation_id_from_args(args, kwargs)
                return conversation_id is not None and username == conversation_id
            else:
                return False

        @wraps(f)
        async def decorated(request: Request, *args: Any, **kwargs: Any) -> Any:

            provided = request.args.get("token", None)

            # noinspection PyProtectedMember
            if token is not None and provided == token:
                result = f(request, *args, **kwargs)
                if isawaitable(result):
                    result = await result
                return result
            elif app.config.get("USE_JWT") and request.app.auth.is_authenticated(
                request
            ):
                if sufficient_scope(request, *args, **kwargs):
                    result = f(request, *args, **kwargs)
                    if isawaitable(result):
                        result = await result
                    return result
                raise ErrorResponse(
                    403,
                    "NotAuthorized",
                    "User has insufficient permissions.",
                    help_url=_docs("/server.html#security-considerations"),
                )
            elif token is None and app.config.get("USE_JWT") is None:
                # authentication is disabled
                result = f(request, *args, **kwargs)
                if isawaitable(result):
                    result = await result
                return result
            raise ErrorResponse(
                401,
                "NotAuthenticated",
                "User is not authenticated.",
                help_url=_docs("/server.html#security-considerations"),
            )

        return decorated

    return decorator


def event_verbosity_parameter(
    request: Request, default_verbosity: EventVerbosity
) -> EventVerbosity:
    event_verbosity_str = request.args.get(
        "include_events", default_verbosity.name
    ).upper()
    try:
        return EventVerbosity[event_verbosity_str]
    except KeyError:
        enum_values = ", ".join([e.name for e in EventVerbosity])
        raise ErrorResponse(
            400,
            "BadRequest",
            "Invalid parameter value for 'include_events'. "
            "Should be one of {}".format(enum_values),
            {"parameter": "include_events", "in": "query"},
        )


def obtain_tracker_store(agent: "Agent", conversation_id: Text) -> DialogueStateTracker:
    tracker = agent.tracker_store.get_or_create_tracker(conversation_id)
    if not tracker:
        raise ErrorResponse(
            409,
            "Conflict",
            "Could not retrieve tracker with id '{}'. Most likely "
            "because there is no domain set on the agent.".format(conversation_id),
        )
    return tracker


def validate_request_body(request: Request, error_message: Text):
    if not request.body:
        raise ErrorResponse(400, "BadRequest", error_message)


async def authenticate(request: Request):
    raise exceptions.AuthenticationFailed(
        "Direct JWT authentication not supported. You should already have "
        "a valid JWT from an authentication provider, Rasa will just make "
        "sure that the token is valid, but not issue new tokens."
    )


def _configure_logging(loglevel: Text, logfile: Text):
    logging.basicConfig(filename=logfile, level=loglevel)
    logging.captureWarnings(True)


def _create_emulator(mode: Optional[Text]) -> NoEmulator:
    """Create emulator for specified mode.
    If no emulator is specified, we will use the Rasa NLU format."""

    if mode is None:
        return NoEmulator()
    elif mode.lower() == "wit":
        from rasa.nlu.emulators.wit import WitEmulator

        return WitEmulator()
    elif mode.lower() == "luis":
        from rasa.nlu.emulators.luis import LUISEmulator

        return LUISEmulator()
    elif mode.lower() == "dialogflow":
        from rasa.nlu.emulators.dialogflow import DialogflowEmulator

        return DialogflowEmulator()
    else:
        raise ErrorResponse(
            400,
            "BadRequest",
            "Invalid parameter value for 'emulation_mode'. "
            "Should be one of 'WIT', 'LUIS', 'DIALOGFLOW'.",
            {"parameter": "emulation_mode", "in": "query"},
        )


async def _load_agent(
    model_path: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    remote_storage: Optional[Text] = None,
) -> Agent:
    try:
        loaded_agent = await load_agent(model_path, model_server, remote_storage)
    except Exception as e:
        logger.debug(traceback.format_exc())
        raise ErrorResponse(
            500, "LoadingError", "An unexpected error occurred. Error: {}".format(e)
        )

    if not loaded_agent:
        raise ErrorResponse(
            400,
            "BadRequest",
            "Agent with name '{}' could not be loaded.".format(model_path),
            {"parameter": "model", "in": "query"},
        )

    return loaded_agent


def create_app(
    agent: Optional["Agent"] = None,
    cors_origins: Union[Text, List[Text]] = "*",
    loglevel: Text = "INFO",
    logfile: Optional[Text] = None,
    auth_token: Optional[Text] = None,
    jwt_secret: Optional[Text] = None,
    jwt_method: Text = "HS256",
):
    """Class representing a Rasa HTTP server."""

    app = Sanic(__name__)
    app.config.RESPONSE_TIMEOUT = 60 * 60

    CORS(
        app, resources={r"/*": {"origins": cors_origins or ""}}, automatic_options=True
    )

    _configure_logging(loglevel, logfile)

    # Setup the Sanic-JWT extension
    if jwt_secret and jwt_method:
        # since we only want to check signatures, we don't actually care
        # about the JWT method and set the passed secret as either symmetric
        # or asymmetric key. jwt lib will choose the right one based on method
        app.config["USE_JWT"] = True
        Initialize(
            app,
            secret=jwt_secret,
            authenticate=authenticate,
            algorithm=jwt_method,
            user_id="username",
        )

    app.agent = agent

    @app.exception(ErrorResponse)
    async def handle_error_response(request: Request, exception: ErrorResponse):
        return response.json(exception.error_info, status=exception.status)

    @app.get("/")
    async def hello(request: Request):
        """Check if the server is running and responds with the version."""
        return response.text("Hello from Rasa: " + rasa.__version__)

    @app.get("/version")
    async def version(request: Request):
        """Respond with the version number of the installed Rasa."""

        return response.json(
            {
                "version": rasa.__version__,
                "minimum_compatible_version": MINIMUM_COMPATIBLE_VERSION,
            }
        )

    @app.get("/status")
    @requires_auth(app, auth_token)
    async def status(request: Request):
        """Respond with the model name and the fingerprint of that model."""

        return response.json(
            {
                "model_file": app.agent.model_directory,
                "fingerprint": fingerprint_from_path(app.agent.model_directory),
            }
        )

    @app.get("/conversations/<conversation_id>/tracker")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def retrieve_tracker(request: Request, conversation_id: Text):
        """Get a dump of a conversation's tracker including its events."""
        if not app.agent.tracker_store:
            raise ErrorResponse(
                409,
                "Conflict",
                "No tracker store available. Make sure to "
                "configure a tracker store when starting "
                "the server.",
            )

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        until_time = rasa.utils.endpoints.float_arg(request, "until")

        tracker = obtain_tracker_store(app.agent, conversation_id)

        try:
            if until_time is not None:
                tracker = tracker.travel_back_in_time(until_time)

            state = tracker.current_state(verbosity)
            return response.json(state)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/conversations/<conversation_id>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def append_event(request: Request, conversation_id: Text):
        """Append a list of events to the state of a conversation"""
        validate_request_body(
            request,
            "You must provide events in the request body in order to append them"
            "to the state of a conversation.",
        )

        evt = Event.from_parameters(request.json)
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        tracker = obtain_tracker_store(app.agent, conversation_id)

        if evt:
            try:
                tracker.update(evt)
                app.agent.tracker_store.save(tracker)
                return response.json(tracker.current_state(verbosity))
            except Exception as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    500,
                    "ConversationError",
                    "An unexpected error occurred. Error: {}".format(e),
                )

        logger.warning(
            "Append event called, but could not extract a valid event. "
            "Request JSON: {}".format(request.json)
        )
        raise ErrorResponse(
            400,
            "BadRequest",
            "Couldn't extract a proper event from the request body.",
            {"parameter": "", "in": "body"},
        )

    @app.put("/conversations/<conversation_id>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def replace_events(request: Request, conversation_id: Text):
        """Use a list of events to set a conversations tracker to a state."""
        validate_request_body(
            request,
            "You must provide events in the request body to set the sate of the "
            "conversation tracker.",
        )

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            tracker = DialogueStateTracker.from_dict(
                conversation_id, request.json, app.agent.domain.slots
            )

            # will override an existing tracker with the same id!
            app.agent.tracker_store.save(tracker)
            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.get("/conversations/<conversation_id>/story")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def retrieve_story(request: Request, conversation_id: Text):
        """Get an end-to-end story corresponding to this conversation."""
        if not app.agent.tracker_store:
            raise ErrorResponse(
                409,
                "Conflict",
                "No tracker store available. Make sure to "
                "configure a tracker store when starting "
                "the server.",
            )

        # retrieve tracker and set to requested state
        tracker = obtain_tracker_store(app.agent, conversation_id)

        until_time = rasa.utils.endpoints.float_arg(request, "until")

        try:
            if until_time is not None:
                tracker = tracker.travel_back_in_time(until_time)

            # dump and return tracker
            state = tracker.export_stories(e2e=True)
            return response.text(state)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/conversations/<conversation_id>/execute")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def execute_action(request: Request, conversation_id: Text):
        request_params = request.json

        action_to_execute = request_params.get("name", None)

        if not action_to_execute:
            raise ErrorResponse(
                400,
                "BadRequest",
                "Name of the action not provided in request body.",
                {"parameter": "name", "in": "body"},
            )

        policy = request_params.get("policy", None)
        confidence = request_params.get("confidence", None)

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            out = CollectingOutputChannel()
            await app.agent.execute_action(
                conversation_id, action_to_execute, out, policy, confidence
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

        tracker = obtain_tracker_store(app.agent, conversation_id)
        state = tracker.current_state(verbosity)
        return response.json({"tracker": state, "messages": out.messages})

    @app.post("/conversations/<conversation_id>/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def predict(request: Request, conversation_id: Text):
        try:
            # Fetches the appropriate bot response in a json format
            responses = app.agent.predict_next(conversation_id)
            responses["scores"] = sorted(
                responses["scores"], key=lambda k: (-k["score"], k["action"])
            )
            return response.json(responses)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/conversations/<conversation_id>/messages")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def add_message(request: Request, conversation_id: Text):
        validate_request_body(
            request,
            "No message defined in request body. Add a message to the request body in "
            "order to add it to the tracker.",
        )

        request_params = request.json

        message = request_params.get("text")
        sender = request_params.get("sender")
        parse_data = request_params.get("parse_data")

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        # TODO: implement for agent / bot
        if sender != "user":
            raise ErrorResponse(
                400,
                "BadRequest",
                "Currently, only user messages can be passed to this endpoint. "
                "Messages of sender '{}' cannot be handled.".format(sender),
                {"parameter": "sender", "in": "body"},
            )

        try:
            user_message = UserMessage(message, None, conversation_id, parse_data)
            tracker = await app.agent.log_message(user_message)
            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ConversationError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/model/train")
    @requires_auth(app, auth_token)
    async def train(request: Request):
        """Train a Rasa Model."""
        from rasa.train import train_async

        validate_request_body(
            request,
            "You must provide training data in the request body in order to "
            "train your model.",
        )

        rjs = request.json
        validate_request(rjs)

        # create a temporary directory to store config, domain and
        # training data
        temp_dir = tempfile.mkdtemp()

        config_path = os.path.join(temp_dir, "config.yml")
        dump_obj_as_str_to_file(config_path, rjs["config"])

        if "nlu" in rjs:
            nlu_path = os.path.join(temp_dir, "nlu.md")
            dump_obj_as_str_to_file(nlu_path, rjs["nlu"])

        if "stories" in rjs:
            stories_path = os.path.join(temp_dir, "stories.md")
            dump_obj_as_str_to_file(stories_path, rjs["stories"])

        domain_path = DEFAULT_DOMAIN_PATH
        if "domain" in rjs:
            domain_path = os.path.join(temp_dir, "domain.yml")
            dump_obj_as_str_to_file(domain_path, rjs["domain"])

        try:
            model_path = await train_async(
                domain=domain_path,
                config=config_path,
                training_files=temp_dir,
                output=rjs.get("out", DEFAULT_MODELS_PATH),
                force_training=rjs.get("force", False),
            )

            return await response.file(model_path)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "TrainingError",
                "An unexpected error occurred during training. Error: {}".format(e),
            )

    def validate_request(rjs):
        if "config" not in rjs:
            raise ErrorResponse(
                400,
                "BadRequest",
                "The training request is missing the required key `config`.",
                {"parameter": "config", "in": "body"},
            )

        if "nlu" not in rjs and "stories" not in rjs:
            raise ErrorResponse(
                400,
                "BadRequest",
                "To train a Rasa model you need to specify at least one type of "
                "training data. Add `nlu` and/or `stories` to the request.",
                {"parameters": ["nlu", "stories"], "in": "body"},
            )

        if "stories" in rjs and "domain" not in rjs:
            raise ErrorResponse(
                400,
                "BadRequest",
                "To train a Rasa model with story training data, you also need to "
                "specify the `domain`.",
                {"parameter": "domain", "in": "body"},
            )

    @app.post("/model/test/stories")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def evaluate_stories(request: Request):
        """Evaluate stories against the currently loaded model."""
        validate_request_body(
            request,
            "You must provide some stories in the request body in order to "
            "evaluate your model.",
        )

        stories = rasa.utils.io.create_temporary_file(request.body, mode="w+b")
        use_e2e = rasa.utils.endpoints.bool_arg(request, "e2e", default=False)

        try:
            evaluation = await test(stories, app.agent, e2e=use_e2e)
            return response.json(evaluation)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "TestingError",
                "An unexpected error occurred during evaluation. Error: {}".format(e),
            )

    @app.post("/model/test/intents")
    @requires_auth(app, auth_token)
    async def evaluate_intents(request: Request):
        """Evaluate intents against a Rasa model."""
        validate_request_body(
            request,
            "You must provide some nlu data in the request body in order to "
            "evaluate your model.",
        )

        eval_agent = app.agent

        model_path = request.args.get("model", None)
        if model_path:
            model_server = app.agent.model_server
            if model_server is not None:
                model_server.url = model_path
            eval_agent = await _load_agent(
                model_path, model_server, app.agent.remote_storage
            )

        nlu_data = rasa.utils.io.create_temporary_file(request.body, mode="w+b")
        data_path = os.path.abspath(nlu_data)

        if not os.path.exists(eval_agent.model_directory):
            raise ErrorResponse(409, "Conflict", "Loaded model file not found.")

        model_directory = eval_agent.model_directory
        _, nlu_model = get_model_subdirectories(model_directory)

        try:
            evaluation = run_evaluation(data_path, nlu_model)
            return response.json(evaluation)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "TestingError",
                "An unexpected error occurred during evaluation. Error: {}".format(e),
            )

    @app.post("/model/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def tracker_predict(request: Request):
        """ Given a list of events, predicts the next action"""
        validate_request_body(
            request,
            "No events defined in request_body. Add events to request body in order to "
            "predict the next action.",
        )

        sender_id = UserMessage.DEFAULT_SENDER_ID
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        request_params = request.json

        try:
            tracker = DialogueStateTracker.from_dict(
                sender_id, request_params, app.agent.domain.slots
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                400,
                "BadRequest",
                "Supplied events are not valid. {}".format(e),
                {"parameter": "", "in": "body"},
            )

        try:
            policy_ensemble = app.agent.policy_ensemble
            probabilities, policy = policy_ensemble.probabilities_using_best_policy(
                tracker, app.agent.domain
            )

            scores = [
                {"action": a, "score": p}
                for a, p in zip(app.agent.domain.action_names, probabilities)
            ]

            return response.json(
                {
                    "scores": scores,
                    "policy": policy,
                    "tracker": tracker.current_state(verbosity),
                }
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "PredictionError",
                "An unexpected error occurred. Error: {}".format(e),
            )

    @app.post("/model/parse")
    @requires_auth(app, auth_token)
    async def parse(request: Request):
        validate_request_body(
            request,
            "No text message defined in request_body. Add text message to request body "
            "in order to obtain the intent and extracted entities.",
        )
        emulation_mode = request.args.get("emulation_mode")
        emulator = _create_emulator(emulation_mode)

        try:
            data = emulator.normalise_request_json(request.json)
            parse_data = await app.agent.interpreter.parse(data.get("text"))
            response_data = emulator.normalise_response_json(parse_data)

            return response.json(response_data)

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500, "ParsingError", "An unexpected error occurred. Error: {}".format(e)
            )

    @app.put("/model")
    @requires_auth(app, auth_token)
    async def load_model(request: Request):
        validate_request_body(request, "No path to model file defined in request_body.")

        model_path = request.json.get("model_file", None)
        model_server = request.json.get("model_server", None)
        remote_storage = request.json.get("remote_storage", None)

        app.agent = await _load_agent(model_path, model_server, remote_storage)

        logger.debug("Successfully loaded model '{}'.".format(model_path))
        return response.json(None, status=204)

    @app.delete("/model")
    @requires_auth(app, auth_token)
    async def unload_model(request: Request):
        model_file = app.agent.model_directory

        app.agent = Agent()

        logger.debug("Successfully unload model '{}'.".format(model_file))
        return response.json(None, status=204)

    @app.get("/domain")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def get_domain(request: Request):
        """Get current domain in yaml or json format."""

        accepts = request.headers.get("Accept", default="application/json")
        if accepts.endswith("json"):
            domain = app.agent.domain.as_dict()
            return response.json(domain)
        elif accepts.endswith("yml") or accepts.endswith("yaml"):
            domain_yaml = app.agent.domain.as_yaml()
            return response.text(
                domain_yaml, status=200, content_type="application/x-yml"
            )
        else:
            raise ErrorResponse(
                406,
                "NotAcceptable",
                "Invalid Accept header. Domain can be "
                "provided as "
                'json ("Accept: application/json") or'
                'yml ("Accept: application/x-yml"). '
                "Make sure you've set the appropriate Accept "
                "header.",
            )

    return app
