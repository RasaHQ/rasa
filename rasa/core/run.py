import asyncio
import logging
import uuid
import os
from functools import partial
from typing import Any, List, Optional, TYPE_CHECKING, Text, Union, Dict

import rasa.core.utils
from rasa.plugin import plugin_manager
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.common
import rasa.utils
import rasa.utils.common
import rasa.utils.io
from rasa import server, telemetry
from rasa.constants import ENV_SANIC_BACKLOG
from rasa.core import agent, channels, constants
from rasa.core.agent import Agent
from rasa.core.channels import console
from rasa.core.channels.channel import InputChannel
from rasa.core.utils import AvailableEndpoints
import rasa.shared.utils.io
from sanic import Sanic
from asyncio import AbstractEventLoop

if TYPE_CHECKING:
    from aiohttp import ClientSession

logger = logging.getLogger()  # get the root logger


def create_input_channels(
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
    app = Sanic("rasa_core_no_api", configure_logging=False)
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
    jwt_private_key: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    route: Optional[Text] = "/webhooks/",
    port: int = constants.DEFAULT_SERVER_PORT,
    endpoints: Optional[AvailableEndpoints] = None,
    log_file: Optional[Text] = None,
    conversation_id: Optional[Text] = uuid.uuid4().hex,
    use_syslog: bool = False,
    syslog_address: Optional[Text] = None,
    syslog_port: Optional[int] = None,
    syslog_protocol: Optional[Text] = None,
    request_timeout: Optional[int] = None,
) -> Sanic:
    """Run the agent."""
    rasa.core.utils.configure_file_logging(
        logger, log_file, use_syslog, syslog_address, syslog_port, syslog_protocol
    )

    if enable_api:
        app = server.create_app(
            cors_origins=cors,
            auth_token=auth_token,
            response_timeout=response_timeout,
            jwt_secret=jwt_secret,
            jwt_private_key=jwt_private_key,
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
                request_timeout=request_timeout,
            )

            logger.info("Killing Sanic server now.")
            running_app.stop()  # kill the sanic server
            plugin_manager().hook.after_server_stop()

        app.add_task(run_cmdline_io)

    return app


def serve_application(
    model_path: Optional[Text] = None,
    channel: Optional[Text] = None,
    interface: Optional[Text] = constants.DEFAULT_SERVER_INTERFACE,
    port: int = constants.DEFAULT_SERVER_PORT,
    credentials: Optional[Text] = None,
    cors: Optional[Union[Text, List[Text]]] = None,
    auth_token: Optional[Text] = None,
    enable_api: bool = True,
    response_timeout: int = constants.DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_private_key: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
    remote_storage: Optional[Text] = None,
    log_file: Optional[Text] = None,
    ssl_certificate: Optional[Text] = None,
    ssl_keyfile: Optional[Text] = None,
    ssl_ca_file: Optional[Text] = None,
    ssl_password: Optional[Text] = None,
    conversation_id: Optional[Text] = uuid.uuid4().hex,
    use_syslog: Optional[bool] = False,
    syslog_address: Optional[Text] = None,
    syslog_port: Optional[int] = None,
    syslog_protocol: Optional[Text] = None,
    request_timeout: Optional[int] = None,
) -> None:
    """Run the API entrypoint."""
    if not channel and not credentials:
        channel = "cmdline"

    input_channels = create_input_channels(channel, credentials)

    app = configure_app(
        input_channels,
        cors,
        auth_token,
        enable_api,
        response_timeout,
        jwt_secret,
        jwt_private_key,
        jwt_method,
        port=port,
        endpoints=endpoints,
        log_file=log_file,
        conversation_id=conversation_id,
        use_syslog=use_syslog,
        syslog_address=syslog_address,
        syslog_port=syslog_port,
        syslog_protocol=syslog_protocol,
        request_timeout=request_timeout,
    )

    ssl_context = server.create_ssl_context(
        ssl_certificate, ssl_keyfile, ssl_ca_file, ssl_password
    )
    protocol = "https" if ssl_context else "http"

    logger.info(f"Starting Rasa server on {protocol}://{interface}:{port}")

    app.register_listener(
        partial(load_agent_on_start, model_path, endpoints, remote_storage),
        "before_server_start",
    )
    app.register_listener(create_connection_pools, "after_server_start")
    app.register_listener(close_resources, "after_server_stop")

    number_of_workers = rasa.core.utils.number_of_sanic_workers(
        endpoints.lock_store if endpoints else None
    )

    telemetry.track_server_start(
        input_channels, endpoints, model_path, number_of_workers, enable_api
    )

    rasa.utils.common.update_sanic_log_level(
        log_file, use_syslog, syslog_address, syslog_port, syslog_protocol
    )

    app.run(
        host=interface,
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
    app.ctx.agent = await agent.load_agent(
        model_path=model_path,
        remote_storage=remote_storage,
        endpoints=endpoints,
        loop=loop,
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
        await event_broker.close()

    action_endpoint = current_agent.action_endpoint
    if action_endpoint:
        await action_endpoint.session.close()

    model_server = current_agent.model_server
    if model_server:
        await model_server.session.close()


async def create_connection_pools(app: Sanic, _: AbstractEventLoop) -> None:
    """Create connection pools for the agent's action server and model server."""
    current_agent = getattr(app.ctx, "agent", None)
    if not current_agent:
        logger.debug("No agent found after server start.")
        return None

    create_action_endpoint_connection_pool(current_agent)
    create_model_server_connection_pool(current_agent)

    return None


def create_action_endpoint_connection_pool(agent: Agent) -> Optional["ClientSession"]:
    """Create a connection pool for the action endpoint."""
    action_endpoint = agent.action_endpoint
    if not action_endpoint:
        logger.debug("No action endpoint found after server start.")
        return None

    return action_endpoint.session


def create_model_server_connection_pool(agent: Agent) -> Optional["ClientSession"]:
    """Create a connection pool for the model server."""
    model_server = agent.model_server
    if not model_server:
        logger.debug("No model server endpoint found after server start.")
        return None

    return model_server.session
