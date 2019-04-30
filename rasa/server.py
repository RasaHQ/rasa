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
from rasa.constants import MINIMUM_COMPATIBLE_VERSION, DEFAULT_MODELS_PATH
from rasa.core import constants
from rasa.core.channels import UserMessage
from rasa.core.test import test
from rasa.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.core.utils import dump_obj_as_str_to_file
from rasa.model import fingerprint_of_model
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


def ensure_loaded_agent(app):
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
        def sender_id_from_args(args: Any, kwargs: Any) -> Optional[Text]:
            argnames = rasa.utils.common.arguments_of(f)

            try:
                sender_id_arg_idx = argnames.index("sender_id")
                if "sender_id" in kwargs:  # try to fetch from kwargs first
                    return kwargs["sender_id"]
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
                sender_id = sender_id_from_args(args, kwargs)
                return sender_id is not None and username == sender_id
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


def event_verbosity_parameter(request, default_verbosity):
    event_verbosity_str = request.raw_args.get(
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


# noinspection PyUnusedLocal
async def authenticate(request):
    raise exceptions.AuthenticationFailed(
        "Direct JWT authentication not supported. You should already have "
        "a valid JWT from an authentication provider, Rasa will just make "
        "sure that the token is valid, but not issue new tokens."
    )


def _configure_logging(loglevel, logfile):
    logging.basicConfig(filename=logfile, level=loglevel)
    logging.captureWarnings(True)


def create_app(
    model_file: Optional[Text] = None,
    agent: Optional["Agent"] = None,
    data_router: Optional["DataRouter"] = None,
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
    app.data_router = data_router
    app.model_file = model_file

    @app.exception(ErrorResponse)
    async def handle_error_response(request: Request, exception: ErrorResponse):
        return response.json(exception.error_info, status=exception.status)

    @app.get("/")
    async def hello(request: Request):
        """Check if the server is running and responds with the version."""
        return response.text("Hello from Rasa: " + rasa.__version__)

    @app.get("/version")
    async def version(request: Request):
        """respond with the version number of the installed rasa core."""

        return response.json(
            {
                "version": rasa.__version__,
                "minimum_compatible_version": MINIMUM_COMPATIBLE_VERSION,
            }
        )

    @app.get("/status")
    @requires_auth(app, auth_token)
    async def status(request: Request):
        """respond with the model name and the fingerprint of that model."""

        return response.json(
            {
                "model_name": app.model_file,
                "fingerprint": fingerprint_of_model(app.model_file),
            }
        )

    @app.get("/conversations/<conversation_id>/tracker")
    @requires_auth(app, auth_token)
    async def retrieve_tracker(request: Request, conversation_id: Text):
        """Get a dump of a conversation's tracker including its events."""
        pass

    @app.post("/conversations/<conversation_id>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def append_event(request: Request, conversation_id: Text):
        """Append a list of events to the state of a conversation"""
        pass

    @app.put("/conversations/<conversation_id>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def replace_events(request: Request, conversation_id: Text):
        """Use a list of events to set a conversations tracker to a state."""
        pass

    @app.get("/conversations/<conversation_id>/story")
    @requires_auth(app, auth_token)
    async def retrieve_story(request: Request, conversation_id: Text):
        """Get an end-to-end story corresponding to this conversation."""
        pass

    @app.post("/conversations/<conversation_id>/execute")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def execute_action(request: Request, conversation_id: Text):
        pass

    @app.post("/conversations/<conversation_id>/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def predict(request: Request, conversation_id: Text):
        pass

    @app.post("/conversations/<conversation_id>/messages")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def add_message(request: Request, conversation_id: Text):
        pass

    @app.post("/model/train")
    @requires_auth(app, auth_token)
    async def train(request: Request):
        """Train a Rasa Model."""
        from rasa.train import train_async

        rjs = request.json

        # create a temporary directory to store config, domain and
        # training data
        temp_dir = tempfile.mkdtemp()

        try:
            config_path = os.path.join(temp_dir, "config.yml")
            dump_obj_as_str_to_file(config_path, rjs["config"])

            domain_path = os.path.join(temp_dir, "domain.yml")
            dump_obj_as_str_to_file(domain_path, rjs["domain"])
        except KeyError:
            raise ErrorResponse(
                400,
                "BadRequest",
                "The training request is missing a key. The required keys are "
                "`config` and `domain`.",
            )

        if "nlu" not in rjs and "stories" not in rjs:
            raise ErrorResponse(
                400,
                "BadRequest",
                "To train a Rasa model you need to specify at least one type of "
                "training data. Add `nlu` and/or `stories` to the request.",
            )

        if "nlu" in rjs:
            nlu_path = os.path.join(temp_dir, "nlu.md")
            dump_obj_as_str_to_file(nlu_path, rjs["nlu"])

        if "stories" in rjs:
            stories_path = os.path.join(temp_dir, "stories.md")
            dump_obj_as_str_to_file(stories_path, rjs["stories"])

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
                "ServerError",
                "An unexpected error occurred during training. Error: {}".format(e),
            )

    @app.post("/model/evaluate/stories")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def evaluate_stories(request: Request):
        """Evaluate stories against the currently loaded model."""
        if not request.body:
            raise ErrorResponse(
                400,
                "BadRequest",
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
                "ServerError",
                "An unexpected error occurred during evaluation. Error: {}".format(e),
            )

    @app.post("/model/evaluate/intents")
    @requires_auth(app, auth_token)
    async def evaluate_intents(request: Request):
        """Evaluate intents against a Rasa model."""
        if not request.body:
            raise ErrorResponse(
                400,
                "BadRequest",
                "You must provide some nlu data in the request body in order to "
                "evaluate your model.",
            )

        nlu_data = rasa.utils.io.create_temporary_file(request.body, mode="w+b")
        data_path = os.path.abspath(nlu_data)

        try:
            evaluation = run_evaluation(data_path, app.model_file)
            return response.json(evaluation)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                500,
                "ServerError",
                "An unexpected error occurred during evaluation. Error: {}".format(e),
            )

    @app.post("/model/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def tracker_predict(request: Request):
        """ Given a list of events, predicts the next action"""

        sender_id = UserMessage.DEFAULT_SENDER_ID
        request_params = request.json
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            tracker = DialogueStateTracker.from_dict(
                sender_id, request_params, app.agent.domain.slots
            )
        except Exception as e:
            raise ErrorResponse(
                400,
                "InvalidParameter",
                "Supplied events are not valid. {}".format(e),
                {"parameter": "", "in": "body"},
            )

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

    @app.post("/model/parse")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def parse(request: Request):
        pass

    @app.post("/model")
    @requires_auth(app, auth_token)
    async def load_model(request: Request):
        pass

    @app.delete("/model")
    @requires_auth(app, auth_token)
    async def unload_model(request: Request):
        pass

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
                "InvalidHeader",
                "Invalid Accept header. Domain can be "
                "provided as "
                'json ("Accept: application/json") or'
                'yml ("Accept: application/x-yml"). '
                "Make sure you've set the appropriate Accept "
                "header.",
            )

    return app
