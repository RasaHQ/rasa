import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import tempfile
import traceback
from collections import defaultdict
from functools import reduce, wraps
from inspect import isawaitable
from pathlib import Path
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    DefaultDict,
    List,
    Optional,
    Text,
    Union,
    Dict,
    TYPE_CHECKING,
    NoReturn,
    Coroutine,
)

import aiohttp
import jsonschema
from sanic import Sanic, response
from sanic.request import Request
from sanic.response import HTTPResponse
from sanic_cors import CORS
from sanic_jwt import Initialize, exceptions

import rasa
import rasa.core.utils
from rasa.nlu.emulators.emulator import Emulator
import rasa.utils.common
import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.shared.utils.validation
import rasa.shared.nlu.training_data.schemas.data_schema
import rasa.utils.endpoints
import rasa.utils.io
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa.core.constants import DEFAULT_RESPONSE_TIMEOUT
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.shared.constants import (
    DOCS_URL_TRAINING_DATA,
    DOCS_BASE_URL,
    DEFAULT_SENDER_ID,
    DEFAULT_MODELS_PATH,
    TEST_STORIES_FILE_PREFIX,
)
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.core.agent import Agent
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    OutputChannel,
    UserMessage,
)
import rasa.shared.core.events
from rasa.shared.core.events import Event
from rasa.core.test import test
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    EventVerbosity,
)
from rasa.core.utils import AvailableEndpoints
from rasa.nlu.emulators.no_emulator import NoEmulator
import rasa.nlu.test
from rasa.nlu.test import CVEvaluationResult
from rasa.shared.utils.schemas.events import EVENTS_SCHEMA
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from ssl import SSLContext  # noqa: F401
    from rasa.core.processor import MessageProcessor
    from mypy_extensions import Arg, VarArg, KwArg

    SanicResponse = Union[
        response.HTTPResponse, Coroutine[Any, Any, response.HTTPResponse]
    ]
    SanicView = Callable[
        [Arg(Request, "request"), VarArg(), KwArg()],  # noqa: F821
        Coroutine[Any, Any, SanicResponse],
    ]


logger = logging.getLogger(__name__)

JSON_CONTENT_TYPE = "application/json"
YAML_CONTENT_TYPE = "application/x-yaml"

OUTPUT_CHANNEL_QUERY_KEY = "output_channel"
USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL = "latest"
EXECUTE_SIDE_EFFECTS_QUERY_KEY = "execute_side_effects"


class ErrorResponse(Exception):
    """Common exception to handle failing API requests."""

    def __init__(
        self,
        status: Union[int, HTTPStatus],
        reason: Text,
        message: Text,
        details: Any = None,
        help_url: Optional[Text] = None,
    ) -> None:
        """Creates error.

        Args:
            status: The HTTP status code to return.
            reason: Short summary of the error.
            message: Detailed explanation of the error.
            details: Additional details which describe the error. Must be serializable.
            help_url: URL where users can get further help (e.g. docs).
        """
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
        logger.error(message)
        super(ErrorResponse, self).__init__()


def _docs(sub_url: Text) -> Text:
    """Create a url to a subpart of the docs."""
    return DOCS_BASE_URL + sub_url


def ensure_loaded_agent(
    app: Sanic, require_core_is_ready: bool = False
) -> Callable[[Callable], Callable[..., Any]]:
    """Wraps a request handler ensuring there is a loaded and usable agent.

    Require the agent to have a loaded Core model if `require_core_is_ready` is
    `True`.
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args: Any, **kwargs: Any) -> Any:
            # noinspection PyUnresolvedReferences
            if not app.ctx.agent or not app.ctx.agent.is_ready():
                raise ErrorResponse(
                    HTTPStatus.CONFLICT,
                    "Conflict",
                    "No agent loaded. To continue processing, a "
                    "model of a trained agent needs to be loaded.",
                    help_url=_docs("/user-guide/configuring-http-api/"),
                )

            return f(*args, **kwargs)

        return decorated

    return decorator


def ensure_conversation_exists() -> Callable[["SanicView"], "SanicView"]:
    """Wraps a request handler ensuring the conversation exists."""

    def decorator(f: "SanicView") -> "SanicView":
        @wraps(f)
        async def decorated(
            request: Request, *args: Any, **kwargs: Any
        ) -> "SanicResponse":
            conversation_id = kwargs["conversation_id"]
            if await request.app.ctx.agent.tracker_store.exists(conversation_id):
                return await f(request, *args, **kwargs)
            else:
                raise ErrorResponse(
                    HTTPStatus.NOT_FOUND, "Not found", "Conversation ID not found."
                )

        return decorated

    return decorator


def requires_auth(
    app: Sanic, token: Optional[Text] = None
) -> Callable[["SanicView"], "SanicView"]:
    """Wraps a request handler with token authentication."""

    def decorator(f: "SanicView") -> "SanicView":
        def conversation_id_from_args(args: Any, kwargs: Any) -> Optional[Text]:
            argnames = rasa.shared.utils.common.arguments_of(f)

            try:
                sender_id_arg_idx = argnames.index("conversation_id")
                if "conversation_id" in kwargs:  # try to fetch from kwargs first
                    return kwargs["conversation_id"]
                if sender_id_arg_idx < len(args):
                    return args[sender_id_arg_idx]
                return None
            except ValueError:
                return None

        async def sufficient_scope(
            request: Request, *args: Any, **kwargs: Any
        ) -> Optional[bool]:
            # This is a coroutine since `sanic-jwt==1.6`
            jwt_data = await rasa.utils.common.call_potential_coroutine(
                request.app.ctx.auth.extract_payload(request)
            )

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
        async def decorated(
            request: Request, *args: Any, **kwargs: Any
        ) -> response.HTTPResponse:

            provided = request.args.get("token", None)

            # noinspection PyProtectedMember
            if token is not None and provided == token:
                result = f(request, *args, **kwargs)
                return await result if isawaitable(result) else result
            elif app.config.get(
                "USE_JWT"
            ) and await rasa.utils.common.call_potential_coroutine(
                # This is a coroutine since `sanic-jwt==1.6`
                request.app.ctx.auth.is_authenticated(request)
            ):
                if await sufficient_scope(request, *args, **kwargs):
                    result = f(request, *args, **kwargs)
                    return await result if isawaitable(result) else result
                raise ErrorResponse(
                    HTTPStatus.FORBIDDEN,
                    "NotAuthorized",
                    "User has insufficient permissions.",
                    help_url=_docs(
                        "/user-guide/configuring-http-api/#security-considerations"
                    ),
                )
            elif token is None and app.config.get("USE_JWT") is None:
                # authentication is disabled
                result = f(request, *args, **kwargs)
                return await result if isawaitable(result) else result
            raise ErrorResponse(
                HTTPStatus.UNAUTHORIZED,
                "NotAuthenticated",
                "User is not authenticated.",
                help_url=_docs(
                    "/user-guide/configuring-http-api/#security-considerations"
                ),
            )

        return decorated

    return decorator


def event_verbosity_parameter(
    request: Request, default_verbosity: EventVerbosity
) -> EventVerbosity:
    """Create `EventVerbosity` object using request params if present."""
    event_verbosity_str = request.args.get(
        "include_events", default_verbosity.name
    ).upper()
    try:
        return EventVerbosity[event_verbosity_str]
    except KeyError:
        enum_values = ", ".join([e.name for e in EventVerbosity])
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            "Invalid parameter value for 'include_events'. "
            "Should be one of {}".format(enum_values),
            {"parameter": "include_events", "in": "query"},
        )


async def get_test_stories(
    processor: "MessageProcessor",
    conversation_id: Text,
    until_time: Optional[float],
    fetch_all_sessions: bool = False,
) -> Text:
    """Retrieves test stories from `processor` for all conversation sessions for
    `conversation_id`.

    Args:
        processor: An instance of `MessageProcessor`.
        conversation_id: Conversation ID to fetch stories for.
        until_time: Timestamp up to which to include events.
        fetch_all_sessions: Whether to fetch stories for all conversation sessions.
            If `False`, only the last conversation session is retrieved.

    Returns:
        The stories for `conversation_id` in test format.
    """
    if fetch_all_sessions:
        trackers = await processor.get_trackers_for_all_conversation_sessions(
            conversation_id
        )
    else:
        trackers = [await processor.get_tracker(conversation_id)]

    if until_time is not None:
        trackers = [tracker.travel_back_in_time(until_time) for tracker in trackers]
        # keep only non-empty trackers
        trackers = [tracker for tracker in trackers if len(tracker.events)]

    logger.debug(
        f"Fetched trackers for {len(trackers)} conversation sessions "
        f"for conversation ID {conversation_id}."
    )

    story_steps = []

    more_than_one_story = len(trackers) > 1

    for i, tracker in enumerate(trackers, 1):
        tracker.sender_id = conversation_id

        if more_than_one_story:
            tracker.sender_id += f", story {i}"

        story_steps += tracker.as_story().story_steps

    return YAMLStoryWriter().dumps(story_steps, is_test_story=True)


async def update_conversation_with_events(
    conversation_id: Text,
    processor: "MessageProcessor",
    domain: Domain,
    events: List[Event],
) -> DialogueStateTracker:
    """Fetches or creates a tracker for `conversation_id` and appends `events` to it.

    Args:
        conversation_id: The ID of the conversation to update the tracker for.
        processor: An instance of `MessageProcessor`.
        domain: The domain associated with the current `Agent`.
        events: The events to append to the tracker.

    Returns:
        The tracker for `conversation_id` with the updated events.
    """
    if rasa.shared.core.events.do_events_begin_with_session_start(events):
        tracker = await processor.get_tracker(conversation_id)
    else:
        tracker = await processor.fetch_tracker_with_initial_session(conversation_id)

    for event in events:
        tracker.update(event, domain)

    return tracker


def validate_request_body(request: Request, error_message: Text) -> None:
    """Check if `request` has a body."""
    if not request.body:
        raise ErrorResponse(HTTPStatus.BAD_REQUEST, "BadRequest", error_message)


def validate_events_in_request_body(request: Request) -> None:
    """Validates events format in request body."""
    if not isinstance(request.json, list):
        events = [request.json]
    else:
        events = request.json

    try:
        jsonschema.validate(events, EVENTS_SCHEMA)
    except jsonschema.ValidationError as error:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            f"Failed to validate the events format. "
            f"For more information about the format visit the docs. Error: {error}",
            help_url=_docs("/pages/http-api"),
        ) from error


async def authenticate(_: Request) -> NoReturn:
    """Callback for authentication failed."""
    raise exceptions.AuthenticationFailed(
        "Direct JWT authentication not supported. You should already have "
        "a valid JWT from an authentication provider, Rasa will just make "
        "sure that the token is valid, but not issue new tokens."
    )


def create_ssl_context(
    ssl_certificate: Optional[Text],
    ssl_keyfile: Optional[Text],
    ssl_ca_file: Optional[Text] = None,
    ssl_password: Optional[Text] = None,
) -> Optional["SSLContext"]:
    """Create an SSL context if a proper certificate is passed.

    Args:
        ssl_certificate: path to the SSL client certificate
        ssl_keyfile: path to the SSL key file
        ssl_ca_file: path to the SSL CA file for verification (optional)
        ssl_password: SSL private key password (optional)

    Returns:
        SSL context if a valid certificate chain can be loaded, `None` otherwise.

    """

    if ssl_certificate:
        import ssl

        ssl_context = ssl.create_default_context(
            purpose=ssl.Purpose.CLIENT_AUTH, cafile=ssl_ca_file
        )
        ssl_context.load_cert_chain(
            ssl_certificate, keyfile=ssl_keyfile, password=ssl_password
        )
        return ssl_context
    else:
        return None


def _create_emulator(mode: Optional[Text]) -> Emulator:
    """Create emulator for specified mode.

    If no emulator is specified, we will use the Rasa NLU format.
    """
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
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            "Invalid parameter value for 'emulation_mode'. "
            "Should be one of 'WIT', 'LUIS', 'DIALOGFLOW'.",
            {"parameter": "emulation_mode", "in": "query"},
        )


async def _load_agent(
    model_path: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    remote_storage: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
) -> Agent:
    try:
        loaded_agent = await rasa.core.agent.load_agent(
            model_path=model_path,
            model_server=model_server,
            remote_storage=remote_storage,
            endpoints=endpoints,
        )
    except Exception as e:
        logger.debug(traceback.format_exc())
        raise ErrorResponse(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "LoadingError",
            f"An unexpected error occurred. Error: {e}",
        )

    if not loaded_agent.is_ready():
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            f"Agent with name '{model_path}' could not be loaded.",
            {"parameter": "model", "in": "query"},
        )

    return loaded_agent


def configure_cors(
    app: Sanic, cors_origins: Union[Text, List[Text], None] = ""
) -> None:
    """Configure CORS origins for the given app."""

    # Workaround so that socketio works with requests from other origins.
    # https://github.com/miguelgrinberg/python-socketio/issues/205#issuecomment-493769183
    app.config.CORS_AUTOMATIC_OPTIONS = True
    app.config.CORS_SUPPORTS_CREDENTIALS = True
    app.config.CORS_EXPOSE_HEADERS = "filename"

    CORS(
        app, resources={r"/*": {"origins": cors_origins or ""}}, automatic_options=True
    )


def add_root_route(app: Sanic) -> None:
    """Add '/' route to return hello."""

    @app.get("/")
    async def hello(request: Request) -> HTTPResponse:
        """Check if the server is running and responds with the version."""
        return response.text("Hello from Rasa: " + rasa.__version__)


def async_if_callback_url(f: Callable[..., Coroutine]) -> Callable:
    """Decorator to enable async request handling.

    If the incoming HTTP request specified a `callback_url` query parameter, the request
    will return immediately with a 204 while the actual request response will
    be sent to the `callback_url`. If an error happens, the error payload will also
    be sent to the `callback_url`.

    Args:
        f: The request handler function which should be decorated.

    Returns:
        The decorated function.
    """

    @wraps(f)
    async def decorated_function(
        request: Request, *args: Any, **kwargs: Any
    ) -> HTTPResponse:
        callback_url = request.args.get("callback_url")
        # Only process request asynchronously if the user specified a `callback_url`
        # query parameter.
        if not callback_url:
            return await f(request, *args, **kwargs)

        async def wrapped() -> None:
            try:
                result: HTTPResponse = await f(request, *args, **kwargs)
                payload: Dict[Text, Any] = dict(
                    data=result.body, headers={"Content-Type": result.content_type}
                )
                logger.debug(
                    "Asynchronous processing of request was successful. "
                    "Sending result to callback URL."
                )

            except Exception as e:
                if not isinstance(e, ErrorResponse):
                    logger.error(e)
                    e = ErrorResponse(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "UnexpectedError",
                        f"An unexpected error occurred. Error: {e}",
                    )
                # If an error happens, we send the error payload to the `callback_url`
                payload = dict(json=e.error_info)
                logger.error(
                    "Error happened when processing request asynchronously. "
                    "Sending error to callback URL."
                )
            async with aiohttp.ClientSession() as session:
                await session.post(callback_url, raise_for_status=True, **payload)

        # Run the request in the background on the event loop
        request.app.add_task(wrapped())

        # The incoming request will return immediately with a 204
        return response.empty()

    return decorated_function


def run_in_thread(f: Callable[..., Coroutine]) -> Callable:
    """Decorator which runs request on a separate thread.

    Some requests (e.g. training or cross-validation) are computional intense requests.
    This means that they will block the event loop and hence the processing of other
    requests. This decorator can be used to process these requests on a separate thread
    to avoid blocking the processing of incoming requests.

    Args:
        f: The request handler function which should be decorated.

    Returns:
        The decorated function.
    """

    @wraps(f)
    async def decorated_function(
        request: Request, *args: Any, **kwargs: Any
    ) -> HTTPResponse:
        # Use a sync wrapper for our `async` function as `run_in_executor` only supports
        # sync functions
        def run() -> HTTPResponse:
            return asyncio.run(f(request, *args, **kwargs))

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await request.app.loop.run_in_executor(pool, run)

    return decorated_function


def inject_temp_dir(f: Callable[..., Coroutine]) -> Callable:
    """Decorator to inject a temporary directory before a request and clean up after.

    Args:
        f: The request handler function which should be decorated.

    Returns:
        The decorated function.
    """

    @wraps(f)
    async def decorated_function(*args: Any, **kwargs: Any) -> HTTPResponse:
        with tempfile.TemporaryDirectory() as directory:
            # Decorated request handles need to have a parameter `temporary_directory`
            return await f(*args, temporary_directory=Path(directory), **kwargs)

    return decorated_function


def create_app(
    agent: Optional["Agent"] = None,
    cors_origins: Union[Text, List[Text], None] = "*",
    auth_token: Optional[Text] = None,
    response_timeout: int = DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_private_key: Optional[Text] = None,
    jwt_method: Text = "HS256",
    endpoints: Optional[AvailableEndpoints] = None,
) -> Sanic:
    """Class representing a Rasa HTTP server."""
    app = Sanic("rasa_server")
    app.config.RESPONSE_TIMEOUT = response_timeout
    configure_cors(app, cors_origins)

    # Set up the Sanic-JWT extension
    if jwt_secret and jwt_method:
        # `sanic-jwt` depends on having an available event loop when making the call to
        # `Initialize`. If there is none, the server startup will fail with
        # `There is no current event loop in thread 'MainThread'`.
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        # since we only want to check signatures, we don't actually care
        # about the JWT method and set the passed secret as either symmetric
        # or asymmetric key. jwt lib will choose the right one based on method
        app.config["USE_JWT"] = True
        Initialize(
            app,
            secret=jwt_secret,
            private_key=jwt_private_key,
            authenticate=authenticate,
            algorithm=jwt_method,
            user_id="username",
        )

    app.ctx.agent = agent
    # Initialize shared object of type unsigned int for tracking
    # the number of active training processes
    app.ctx.active_training_processes = multiprocessing.Value("I", 0)

    @app.exception(ErrorResponse)
    async def handle_error_response(
        request: Request, exception: ErrorResponse
    ) -> HTTPResponse:
        return response.json(exception.error_info, status=exception.status)

    add_root_route(app)

    @app.get("/version")
    async def version(request: Request) -> HTTPResponse:
        """Respond with the version number of the installed Rasa."""

        return response.json(
            {
                "version": rasa.__version__,
                "minimum_compatible_version": MINIMUM_COMPATIBLE_VERSION,
            }
        )

    @app.get("/status")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def status(request: Request) -> HTTPResponse:
        """Respond with the model name and the fingerprint of that model."""
        return response.json(
            {
                "model_file": app.ctx.agent.processor.model_filename,
                "model_id": app.ctx.agent.model_id,
                "num_active_training_jobs": app.ctx.active_training_processes.value,
            }
        )

    @app.get("/conversations/<conversation_id:path>/tracker")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def retrieve_tracker(request: Request, conversation_id: Text) -> HTTPResponse:
        """Get a dump of a conversation's tracker including its events."""
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        until_time = rasa.utils.endpoints.float_arg(request, "until")

        tracker = await app.ctx.agent.processor.fetch_tracker_with_initial_session(
            conversation_id
        )

        try:
            if until_time is not None:
                tracker = tracker.travel_back_in_time(until_time)

            state = tracker.current_state(verbosity)
            return response.json(state)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/conversations/<conversation_id:path>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def append_events(request: Request, conversation_id: Text) -> HTTPResponse:
        """Append a list of events to the state of a conversation."""
        validate_events_in_request_body(request)

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.ctx.agent.lock_store.lock(conversation_id):
                processor = app.ctx.agent.processor
                events = _get_events_from_request_body(request)

                tracker = await update_conversation_with_events(
                    conversation_id, processor, app.ctx.agent.domain, events
                )

                output_channel = _get_output_channel(request, tracker)

                if rasa.utils.endpoints.bool_arg(
                    request, EXECUTE_SIDE_EFFECTS_QUERY_KEY, False
                ):
                    await processor.execute_side_effects(
                        events, tracker, output_channel
                    )
                await app.ctx.agent.tracker_store.save(tracker)
            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    def _get_events_from_request_body(request: Request) -> List[Event]:
        events = request.json

        if not isinstance(events, list):
            events = [events]

        events = [Event.from_parameters(event) for event in events]
        events = [event for event in events if event]

        if not events:
            rasa.shared.utils.io.raise_warning(
                f"Append event called, but could not extract a valid event. "
                f"Request JSON: {request.json}"
            )
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Couldn't extract a proper event from the request body.",
                {"parameter": "", "in": "body"},
            )

        return events

    @app.put("/conversations/<conversation_id:path>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def replace_events(request: Request, conversation_id: Text) -> HTTPResponse:
        """Use a list of events to set a conversations tracker to a state."""
        validate_events_in_request_body(request)

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.ctx.agent.lock_store.lock(conversation_id):
                tracker = DialogueStateTracker.from_dict(
                    conversation_id, request.json, app.ctx.agent.domain.slots
                )

                # will override an existing tracker with the same id!
                await app.ctx.agent.tracker_store.save(tracker)

            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.get("/conversations/<conversation_id:path>/story")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    @ensure_conversation_exists()
    async def retrieve_story(request: Request, conversation_id: Text) -> HTTPResponse:
        """Get an end-to-end story corresponding to this conversation."""
        until_time = rasa.utils.endpoints.float_arg(request, "until")
        fetch_all_sessions = rasa.utils.endpoints.bool_arg(
            request, "all_sessions", default=False
        )

        try:
            stories = await get_test_stories(
                app.ctx.agent.processor,
                conversation_id,
                until_time,
                fetch_all_sessions=fetch_all_sessions,
            )
            return response.text(stories)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/conversations/<conversation_id:path>/execute")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    @ensure_conversation_exists()
    async def execute_action(request: Request, conversation_id: Text) -> HTTPResponse:
        rasa.shared.utils.io.raise_warning(
            'The "POST /conversations/<conversation_id>/execute"'
            " endpoint is deprecated. Inserting actions to the tracker externally"
            " should be avoided. Actions should be predicted by the policies only.",
            category=FutureWarning,
        )
        request_params = request.json

        action_to_execute = request_params.get("name", None)

        if not action_to_execute:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Name of the action not provided in request body.",
                {"parameter": "name", "in": "body"},
            )

        policy = request_params.get("policy", None)
        confidence = request_params.get("confidence", None)
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.ctx.agent.lock_store.lock(conversation_id):
                tracker = await (
                    app.ctx.agent.processor.fetch_tracker_and_update_session(
                        conversation_id
                    )
                )

                output_channel = _get_output_channel(request, tracker)
                await app.ctx.agent.execute_action(
                    conversation_id,
                    action_to_execute,
                    output_channel,
                    policy,
                    confidence,
                )

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

        state = tracker.current_state(verbosity)

        response_body: Dict[Text, Any] = {"tracker": state}

        if isinstance(output_channel, CollectingOutputChannel):
            response_body["messages"] = output_channel.messages

        return response.json(response_body)

    @app.post("/conversations/<conversation_id:path>/trigger_intent")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def trigger_intent(request: Request, conversation_id: Text) -> HTTPResponse:
        request_params = request.json

        intent_to_trigger = request_params.get("name")
        entities = request_params.get("entities", [])

        if not intent_to_trigger:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Name of the intent not provided in request body.",
                {"parameter": "name", "in": "body"},
            )

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.ctx.agent.lock_store.lock(conversation_id):
                tracker = await (
                    app.ctx.agent.processor.fetch_tracker_and_update_session(
                        conversation_id
                    )
                )
                output_channel = _get_output_channel(request, tracker)
                if intent_to_trigger not in app.ctx.agent.domain.intents:
                    raise ErrorResponse(
                        HTTPStatus.NOT_FOUND,
                        "NotFound",
                        f"The intent {trigger_intent} does not exist in the domain.",
                    )
                await app.ctx.agent.trigger_intent(
                    intent_name=intent_to_trigger,
                    entities=entities,
                    output_channel=output_channel,
                    tracker=tracker,
                )
        except ErrorResponse:
            raise
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

        state = tracker.current_state(verbosity)

        response_body: Dict[Text, Any] = {"tracker": state}

        if isinstance(output_channel, CollectingOutputChannel):
            response_body["messages"] = output_channel.messages

        return response.json(response_body)

    @app.post("/conversations/<conversation_id:path>/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    @ensure_conversation_exists()
    async def predict(request: Request, conversation_id: Text) -> HTTPResponse:
        try:
            # Fetches the appropriate bot response in a json format
            responses = await app.ctx.agent.predict_next_for_sender_id(conversation_id)
            responses["scores"] = sorted(
                responses["scores"], key=lambda k: (-k["score"], k["action"])
            )
            return response.json(responses)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/conversations/<conversation_id:path>/messages")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def add_message(request: Request, conversation_id: Text) -> HTTPResponse:
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
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Currently, only user messages can be passed to this endpoint. "
                "Messages of sender '{}' cannot be handled.".format(sender),
                {"parameter": "sender", "in": "body"},
            )

        user_message = UserMessage(message, None, conversation_id, parse_data)

        try:
            async with app.ctx.agent.lock_store.lock(conversation_id):
                # cf. processor.handle_message (ignoring prediction loop run)
                tracker = await app.ctx.agent.processor.log_message(
                    user_message, should_save_tracker=False
                )
                tracker = await app.ctx.agent.processor.run_action_extract_slots(
                    user_message.output_channel, tracker
                )
                await app.ctx.agent.processor.save_tracker(tracker)

            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/model/train")
    @requires_auth(app, auth_token)
    @async_if_callback_url
    @run_in_thread
    @inject_temp_dir
    async def train(request: Request, temporary_directory: Path) -> HTTPResponse:
        validate_request_body(
            request,
            "You must provide training data in the request body in order to "
            "train your model.",
        )

        training_payload = _training_payload_from_yaml(request, temporary_directory)

        try:
            with app.ctx.active_training_processes.get_lock():
                app.ctx.active_training_processes.value += 1

            from rasa.model_training import train

            # pass `None` to run in default executor
            training_result = train(**training_payload)

            if training_result.model:
                filename = os.path.basename(training_result.model)

                return await response.file(
                    training_result.model,
                    filename=filename,
                    headers={"filename": filename},
                )
            else:
                raise ErrorResponse(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "TrainingError",
                    "Ran training, but it finished without a trained model.",
                )
        except ErrorResponse as e:
            raise e
        except InvalidDomain as e:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "InvalidDomainError",
                f"Provided domain file is invalid. Error: {e}",
            )
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TrainingError",
                f"An unexpected error occurred during training. Error: {e}",
            )
        finally:
            with app.ctx.active_training_processes.get_lock():
                app.ctx.active_training_processes.value -= 1

    @app.post("/model/test/stories")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app, require_core_is_ready=True)
    @inject_temp_dir
    async def evaluate_stories(
        request: Request, temporary_directory: Path
    ) -> HTTPResponse:
        """Evaluate stories against the currently loaded model."""
        validate_request_body(
            request,
            "You must provide some stories in the request body in order to "
            "evaluate your model.",
        )

        test_data = _test_data_file_from_payload(request, temporary_directory)

        e2e = rasa.utils.endpoints.bool_arg(request, "e2e", default=False)

        try:
            evaluation = await test(
                test_data, app.ctx.agent, e2e=e2e, disable_plotting=True
            )
            return response.json(evaluation)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TestingError",
                f"An unexpected error occurred during evaluation. Error: {e}",
            )

    @app.post("/model/test/intents")
    @requires_auth(app, auth_token)
    @async_if_callback_url
    @run_in_thread
    @inject_temp_dir
    async def evaluate_intents(
        request: Request, temporary_directory: Path
    ) -> HTTPResponse:
        """Evaluate intents against a Rasa model."""
        validate_request_body(
            request,
            "You must provide some nlu data in the request body in order to "
            "evaluate your model.",
        )

        cross_validation_folds = request.args.get("cross_validation_folds")
        is_yaml_payload = request.headers.get("Content-type") == YAML_CONTENT_TYPE
        test_coroutine = None

        if is_yaml_payload:
            payload = _training_payload_from_yaml(request, temporary_directory)
            config_file = payload.get("config")
            test_data = payload.get("training_files")

            if cross_validation_folds:
                test_coroutine = _cross_validate(
                    test_data, config_file, int(cross_validation_folds)
                )
        else:
            payload = _nlu_training_payload_from_json(request, temporary_directory)
            test_data = payload.get("training_files")

            if cross_validation_folds:
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "TestingError",
                    "Cross-validation is only supported for YAML data.",
                )

        if not cross_validation_folds:
            test_coroutine = _evaluate_model_using_test_set(
                request.args.get("model"), test_data
            )

        try:
            if test_coroutine is not None:
                evaluation = await test_coroutine
            return response.json(evaluation)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TestingError",
                f"An unexpected error occurred during evaluation. Error: {e}",
            )

    async def _evaluate_model_using_test_set(
        model_path: Text, test_data_file: Text
    ) -> Dict:
        logger.info("Starting model evaluation using test set.")

        eval_agent = app.ctx.agent

        if model_path:
            model_server = app.ctx.agent.model_server
            if model_server is not None:
                model_server = model_server.copy()
                model_server.url = model_path
                # Set wait time between pulls to `0` so that the agent does not schedule
                # a job to pull the model from the server
                model_server.kwargs["wait_time_between_pulls"] = 0
            eval_agent = await _load_agent(
                model_path=model_path,
                model_server=model_server,
                remote_storage=app.ctx.agent.remote_storage,
            )

        data_path = os.path.abspath(test_data_file)

        if not eval_agent.is_ready():
            raise ErrorResponse(
                HTTPStatus.CONFLICT, "Conflict", "Loaded model file not found."
            )

        return await rasa.nlu.test.run_evaluation(
            data_path, eval_agent.processor, disable_plotting=True, report_as_dict=True
        )

    async def _cross_validate(data_file: Text, config_file: Text, folds: int) -> Dict:
        logger.info(f"Starting cross-validation with {folds} folds.")
        importer = TrainingDataImporter.load_from_dict(
            config=None, config_path=config_file, training_data_paths=[data_file]
        )
        config = importer.get_config()
        nlu_data = importer.get_nlu_data()

        evaluations = await rasa.nlu.test.cross_validate(
            data=nlu_data,
            n_folds=folds,
            nlu_config=config,
            disable_plotting=True,
            errors=True,
            report_as_dict=True,
        )
        evaluation_results = _get_evaluation_results(*evaluations)

        return evaluation_results

    def _get_evaluation_results(
        intent_report: CVEvaluationResult,
        entity_report: CVEvaluationResult,
        response_selector_report: CVEvaluationResult,
    ) -> Dict[Text, Any]:
        eval_name_mapping = {
            "intent_evaluation": intent_report,
            "entity_evaluation": entity_report,
            "response_selection_evaluation": response_selector_report,
        }

        result: DefaultDict[Text, Any] = defaultdict(dict)
        for evaluation_name, evaluation in eval_name_mapping.items():
            report = evaluation.evaluation.get("report", {})
            averages = report.get("weighted avg", {})
            result[evaluation_name]["report"] = report
            result[evaluation_name]["precision"] = averages.get("precision")
            result[evaluation_name]["f1_score"] = averages.get("1-score")
            result[evaluation_name]["errors"] = evaluation.evaluation.get("errors", [])

        return result

    @app.post("/model/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app, require_core_is_ready=True)
    async def tracker_predict(request: Request) -> HTTPResponse:
        """Given a list of events, predicts the next action."""
        validate_events_in_request_body(request)

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        request_params = request.json
        try:
            tracker = DialogueStateTracker.from_dict(
                DEFAULT_SENDER_ID, request_params, app.ctx.agent.domain.slots
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                f"Supplied events are not valid. {e}",
                {"parameter": "", "in": "body"},
            )

        try:
            result = app.ctx.agent.predict_next_with_tracker(tracker, verbosity)

            return response.json(result)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "PredictionError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/model/parse")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def parse(request: Request) -> HTTPResponse:
        validate_request_body(
            request,
            "No text message defined in request_body. Add text message to request body "
            "in order to obtain the intent and extracted entities.",
        )
        emulation_mode = request.args.get("emulation_mode")
        emulator = _create_emulator(emulation_mode)

        try:
            data = emulator.normalise_request_json(request.json)
            try:
                parsed_data = await app.ctx.agent.parse_message(data.get("text"))
            except Exception as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "ParsingError",
                    f"An unexpected error occurred. Error: {e}",
                )
            response_data = emulator.normalise_response_json(parsed_data)

            return response.json(response_data)

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ParsingError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.put("/model")
    @requires_auth(app, auth_token)
    async def load_model(request: Request) -> HTTPResponse:
        validate_request_body(request, "No path to model file defined in request_body.")

        model_path = request.json.get("model_file", None)
        model_server = request.json.get("model_server", None)
        remote_storage = request.json.get("remote_storage", None)

        if model_server:
            try:
                model_server = EndpointConfig.from_dict(model_server)
            except TypeError as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "BadRequest",
                    f"Supplied 'model_server' is not valid. Error: {e}",
                    {"parameter": "model_server", "in": "body"},
                )

        new_agent = await _load_agent(
            model_path=model_path,
            model_server=model_server,
            remote_storage=remote_storage,
            endpoints=endpoints,
        )
        new_agent.lock_store = app.ctx.agent.lock_store
        app.ctx.agent = new_agent

        logger.debug(f"Successfully loaded model '{model_path}'.")
        return response.json(None, status=HTTPStatus.NO_CONTENT)

    @app.delete("/model")
    @requires_auth(app, auth_token)
    async def unload_model(request: Request) -> HTTPResponse:
        model_file = app.ctx.agent.model_name

        app.ctx.agent = Agent(lock_store=app.ctx.agent.lock_store)

        logger.debug(f"Successfully unloaded model '{model_file}'.")
        return response.json(None, status=HTTPStatus.NO_CONTENT)

    @app.get("/domain")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def get_domain(request: Request) -> HTTPResponse:
        """Get current domain in yaml or json format."""
        # FIXME: this is a false positive mypy error after upgrading to 0.931
        accepts = request.headers.get("Accept", default=JSON_CONTENT_TYPE)
        if accepts.endswith("json"):
            domain = app.ctx.agent.domain.as_dict()
            return response.json(domain)
        elif accepts.endswith("yml") or accepts.endswith("yaml"):
            domain_yaml = app.ctx.agent.domain.as_yaml()
            return response.text(
                domain_yaml, status=HTTPStatus.OK, content_type=YAML_CONTENT_TYPE
            )
        else:
            raise ErrorResponse(
                HTTPStatus.NOT_ACCEPTABLE,
                "NotAcceptable",
                f"Invalid Accept header. Domain can be "
                f"provided as "
                f'json ("Accept: {JSON_CONTENT_TYPE}") or'
                f'yml ("Accept: {YAML_CONTENT_TYPE}"). '
                f"Make sure you've set the appropriate Accept "
                f"header.",
            )

    return app


def _get_output_channel(
    request: Request, tracker: Optional[DialogueStateTracker]
) -> OutputChannel:
    """Returns the `OutputChannel` which should be used for the bot's responses.

    Args:
        request: HTTP request whose query parameters can specify which `OutputChannel`
                 should be used.
        tracker: Tracker for the conversation. Used to get the latest input channel.

    Returns:
        `OutputChannel` which should be used to return the bot's responses to.
    """
    requested_output_channel = request.args.get(OUTPUT_CHANNEL_QUERY_KEY)

    if (
        requested_output_channel == USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL
        and tracker
    ):
        requested_output_channel = tracker.get_latest_input_channel()

    # Interactive training does not set `input_channels`, hence we have to be cautious
    registered_input_channels = getattr(request.app.ctx, "input_channels", None) or []
    matching_channels = [
        channel
        for channel in registered_input_channels
        if channel.name() == requested_output_channel
    ]

    # Check if matching channels can provide a valid output channel,
    # otherwise use `CollectingOutputChannel`
    return reduce(
        lambda output_channel_created_so_far, input_channel: (
            input_channel.get_output_channel() or output_channel_created_so_far
        ),
        matching_channels,
        CollectingOutputChannel(),
    )


def _test_data_file_from_payload(request: Request, temporary_directory: Path) -> Text:
    return str(
        _training_payload_from_yaml(
            request,
            temporary_directory,
            # test stories have to prefixed with `test_`
            file_name=f"{TEST_STORIES_FILE_PREFIX}data.yml",
        )["training_files"]
    )


def _training_payload_from_yaml(
    request: Request, temp_dir: Path, file_name: Text = "data.yml"
) -> Dict[Text, Any]:
    logger.debug("Extracting YAML training data from request body.")

    decoded = request.body.decode(rasa.shared.utils.io.DEFAULT_ENCODING)
    _validate_yaml_training_payload(decoded)

    training_data = temp_dir / file_name
    rasa.shared.utils.io.write_text_file(decoded, training_data)

    model_output_directory = str(temp_dir)
    if rasa.utils.endpoints.bool_arg(request, "save_to_default_model_directory", True):
        model_output_directory = DEFAULT_MODELS_PATH

    return dict(
        domain=str(training_data),
        config=str(training_data),
        training_files=str(temp_dir),
        output=model_output_directory,
        force_training=rasa.utils.endpoints.bool_arg(request, "force_training", False),
        core_additional_arguments=_extract_core_additional_arguments(request),
        nlu_additional_arguments=_extract_nlu_additional_arguments(request),
    )


def _nlu_training_payload_from_json(
    request: Request, temp_dir: Path, file_name: Text = "data.json"
) -> Dict[Text, Any]:
    logger.debug("Extracting JSON training data from request body.")

    rasa.shared.utils.validation.validate_training_data(
        request.json,
        rasa.shared.nlu.training_data.schemas.data_schema.rasa_nlu_data_schema(),
    )
    training_data = temp_dir / file_name
    rasa.shared.utils.io.dump_obj_as_json_to_file(training_data, request.json)

    model_output_directory = str(temp_dir)
    if rasa.utils.endpoints.bool_arg(request, "save_to_default_model_directory", True):
        model_output_directory = DEFAULT_MODELS_PATH

    return dict(
        domain=str(training_data),
        config=str(training_data),
        training_files=str(temp_dir),
        output=model_output_directory,
        force_training=rasa.utils.endpoints.bool_arg(request, "force_training", False),
        core_additional_arguments=_extract_core_additional_arguments(request),
        nlu_additional_arguments=_extract_nlu_additional_arguments(request),
    )


def _validate_yaml_training_payload(yaml_text: Text) -> None:
    try:
        RasaYAMLReader().validate(yaml_text)
    except Exception as e:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            f"The request body does not contain valid YAML. Error: {e}",
            help_url=DOCS_URL_TRAINING_DATA,
        )


def _extract_core_additional_arguments(request: Request) -> Dict[Text, Any]:
    return {
        "augmentation_factor": rasa.utils.endpoints.int_arg(request, "augmentation", 50)
    }


def _extract_nlu_additional_arguments(request: Request) -> Dict[Text, Any]:
    return {"num_threads": rasa.utils.endpoints.int_arg(request, "num_threads", 1)}
