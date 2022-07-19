import asyncio
import logging
import uuid
import os
import shutil
from functools import partial
from typing import Any, List, Optional, Text, Union, Dict

import rasa.core.utils
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.common
import rasa.utils
import rasa.utils.common
import rasa.utils.io
from rasa import model, server, telemetry
from rasa.constants import ENV_SANIC_BACKLOG
from rasa.core import agent, channels, constants
from rasa.core.agent import Agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels import console
from rasa.core.channels.channel import InputChannel
import rasa.core.interpreter
from rasa.core.lock_store import LockStore
from rasa.core.tracker_store import TrackerStore
from rasa.core.utils import AvailableEndpoints
import rasa.shared.utils.io
from sanic import Sanic
from asyncio import AbstractEventLoop

logger = logging.getLogger()  # get the root logger


def create_http_input_channels(
    channel: Optional[Text], credentials_file: Optional[Text]
) -> List["InputChannel"]:
    """Instantiate the chosen input channel."""

    if credentials_file:
        all_credentials = rasa.shared.utils.io.read_config_file(credentials_file)
    else:
        all_credentials = {}

    if channel:
        if len(all_credentials) > 1:
            logger.info(
                "Connecting to channel '{}' which was specified by the "
                "'--connector' argument. Any other channels will be ignored. "
                "To connect to all given channels, omit the '--connector' "
                "argument.".format(channel)
            )
        return [_create_single_channel(channel, all_credentials.get(channel))]
    else:
        return [_create_single_channel(c, k) for c, k in all_credentials.items()]


def _create_single_channel(channel: Text, credentials: Dict[Text, Any]) -> Any:
    from rasa.core.channels import BUILTIN_CHANNELS

    if channel in BUILTIN_CHANNELS:
        return BUILTIN_CHANNELS[channel].from_credentials(credentials)
    else:
        # try to load channel based on class name
        try:
            input_channel_class = rasa.shared.utils.common.class_from_module_path(
                channel
            )
            return input_channel_class.from_credentials(credentials)
        except (AttributeError, ImportError):
            raise RasaException(
                f"Failed to find input channel class for '{channel}'. Unknown "
                f"input channel. Check your credentials configuration to "
                f"make sure the mentioned channel is not misspelled. "
                f"If you are creating your own channel, make sure it "
                f"is a proper name of a class in a module."
            )


def _create_app_without_api(cors: Optional[Union[Text, List[Text]]] = None) -> Sanic:
    app = Sanic("rasa_core_no_api", configure_logging=False, register=False)
    server.add_root_route(app)
    server.configure_cors(app, cors)
    return app


def configure_app(
    input_channels: Optional[List["InputChannel"]] = None,
    cors: Optional[Union[Text, List[Text], None]] = None,
    auth_token: Optional[Text] = None,
    enable_api: bool = True,
    response_timeout: int = constants.DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    route: Optional[Text] = "/webhooks/",
    port: int = constants.DEFAULT_SERVER_PORT,
    endpoints: Optional[AvailableEndpoints] = None,
    log_file: Optional[Text] = None,
    conversation_id: Optional[Text] = uuid.uuid4().hex,
) -> Sanic:
    """Run the agent."""

    rasa.core.utils.configure_file_logging(logger, log_file)

    if enable_api:
        app = server.create_app(
            cors_origins=cors,
            auth_token=auth_token,
            response_timeout=response_timeout,
            jwt_secret=jwt_secret,
            jwt_method=jwt_method,
            endpoints=endpoints,
        )
    else:
        app = _create_app_without_api(cors)

    if input_channels:
        channels.channel.register(input_channels, app, route=route)
    else:
        input_channels = []

    if logger.isEnabledFor(logging.DEBUG):
        rasa.core.utils.list_routes(app)

    async def configure_async_logging() -> None:
        if logger.isEnabledFor(logging.DEBUG):
            rasa.utils.io.enable_async_loop_debugging(asyncio.get_event_loop())

    app.add_task(configure_async_logging)

    if "cmdline" in {c.name() for c in input_channels}:

        async def run_cmdline_io(running_app: Sanic) -> None:
            """Small wrapper to shut down the server once cmd io is done."""
            await asyncio.sleep(1)  # allow server to start

            await console.record_messages(
                server_url=constants.DEFAULT_SERVER_FORMAT.format("http", port),
                sender_id=conversation_id,
            )

            logger.info("Killing Sanic server now.")
            running_app.stop()  # kill the sanic server

        app.add_task(run_cmdline_io)

    return app


def serve_application(
    model_path: Optional[Text] = None,
    channel: Optional[Text] = None,
    port: int = constants.DEFAULT_SERVER_PORT,
    credentials: Optional[Text] = None,
    cors: Optional[Union[Text, List[Text]]] = None,
    auth_token: Optional[Text] = None,
    enable_api: bool = True,
    response_timeout: int = constants.DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
    remote_storage: Optional[Text] = None,
    log_file: Optional[Text] = None,
    ssl_certificate: Optional[Text] = None,
    ssl_keyfile: Optional[Text] = None,
    ssl_ca_file: Optional[Text] = None,
    ssl_password: Optional[Text] = None,
    conversation_id: Optional[Text] = uuid.uuid4().hex,
) -> None:
    """Run the API entrypoint."""

    if not channel and not credentials:
        channel = "cmdline"

    input_channels = create_http_input_channels(channel, credentials)

    app = configure_app(
        input_channels,
        cors,
        auth_token,
        enable_api,
        response_timeout,
        jwt_secret,
        jwt_method,
        port=port,
        endpoints=endpoints,
        log_file=log_file,
        conversation_id=conversation_id,
    )

    ssl_context = server.create_ssl_context(
        ssl_certificate, ssl_keyfile, ssl_ca_file, ssl_password
    )
    protocol = "https" if ssl_context else "http"

    logger.info(
        f"Starting Rasa server on "
        f"{constants.DEFAULT_SERVER_FORMAT.format(protocol, port)}"
    )

    app.register_listener(
        partial(load_agent_on_start, model_path, endpoints, remote_storage),
        "before_server_start",
    )
    app.register_listener(close_resources, "after_server_stop")

    # noinspection PyUnresolvedReferences
    async def clear_model_files(_app: Sanic, _loop: Text) -> None:
        if app.ctx.agent.model_directory:
            shutil.rmtree(_app.ctx.agent.model_directory)

    number_of_workers = rasa.core.utils.number_of_sanic_workers(
        endpoints.lock_store if endpoints else None
    )

    telemetry.track_server_start(
        input_channels, endpoints, model_path, number_of_workers, enable_api
    )

    app.register_listener(clear_model_files, "after_server_stop")

    rasa.utils.common.update_sanic_log_level(log_file)
    app.run(
        host="0.0.0.0",
        port=port,
        ssl=ssl_context,
        backlog=int(os.environ.get(ENV_SANIC_BACKLOG, "100")),
        workers=number_of_workers,
    )


# noinspection PyUnusedLocal
async def load_agent_on_start(
    model_path: Text,
    endpoints: AvailableEndpoints,
    remote_storage: Optional[Text],
    app: Sanic,
    loop: AbstractEventLoop,
) -> Agent:
    """Load an agent.

    Used to be scheduled on server start
    (hence the `app` and `loop` arguments).
    """
    # noinspection PyBroadException
    try:
        with model.get_model(model_path) as unpacked_model:
            _, nlu_model = model.get_model_subdirectories(unpacked_model)
            _interpreter = rasa.core.interpreter.create_interpreter(
                endpoints.nlu or nlu_model
            )
    except Exception:
        logger.debug(f"Could not load interpreter from '{model_path}'.")
        _interpreter = None

    _broker = await EventBroker.create(endpoints.event_broker, loop=loop)
    _tracker_store = TrackerStore.create(endpoints.tracker_store, event_broker=_broker)
    _lock_store = LockStore.create(endpoints.lock_store)

    model_server = endpoints.model if endpoints and endpoints.model else None

    try:
        app.ctx.agent = await agent.load_agent(
            model_path,
            model_server=model_server,
            remote_storage=remote_storage,
            interpreter=_interpreter,
            generator=endpoints.nlg,
            tracker_store=_tracker_store,
            lock_store=_lock_store,
            action_endpoint=endpoints.action,
        )
    except Exception as e:
        rasa.shared.utils.io.raise_warning(
            f"The model at '{model_path}' could not be loaded. "
            f"Error: {type(e)}: {e}"
        )
        app.ctx.agent = None

    if not app.ctx.agent:
        rasa.shared.utils.io.raise_warning(
            "Agent could not be loaded with the provided configuration. "
            "Load default agent without any model."
        )
        app.ctx.agent = Agent(
            interpreter=_interpreter,
            generator=endpoints.nlg,
            tracker_store=_tracker_store,
            action_endpoint=endpoints.action,
            model_server=model_server,
            remote_storage=remote_storage,
        )

    logger.info("Rasa server is up and running.")
    return app.ctx.agent


async def close_resources(app: Sanic, _: AbstractEventLoop) -> None:
    """Gracefully closes resources when shutting down server.

    Args:
        app: The Sanic application.
        _: The current Sanic worker event loop.
    """
    current_agent = getattr(app.ctx, "agent", None)
    if not current_agent:
        logger.debug("No agent found when shutting down server.")
        return

    event_broker = current_agent.tracker_store.event_broker
    if event_broker:
        if not asyncio.iscoroutinefunction(event_broker.close):
            rasa.shared.utils.io.raise_deprecation_warning(
                f"The method '{EventBroker.__name__}.{EventBroker.close.__name__}' was "
                f"changed to be asynchronous. Please adapt your custom event broker "
                f"accordingly. Support for synchronous implementations will be removed "
                f"in Rasa Open Source 3.0.0."
            )
            event_broker.close()
        else:
            await event_broker.close()
