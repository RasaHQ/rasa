import asyncio
import logging
import os
import platform
import uuid
import warnings
from asyncio import AbstractEventLoop
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Text,
    Tuple,
    Union,
)

from sanic import Sanic
from sanic.worker.loader import AppLoader

import rasa.core.utils
import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.utils
import rasa.utils.common
import rasa.utils.io
from rasa import server, telemetry
from rasa.constants import ENV_SANIC_BACKLOG
from rasa.core import agent, channels, constants
from rasa.core.agent import Agent
from rasa.core.available_agents import AvailableAgents
from rasa.core.channels import console
from rasa.core.channels.channel import InputChannel
from rasa.core.channels.development_inspector import DevelopmentInspectProxy
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.credentials import CredentialsConfig
from rasa.core.persistor import StorageType
from rasa.shared.exceptions import RasaException
from rasa.utils import licensing

logger = logging.getLogger()  # get the root logger


def create_input_channels(
    channel: Optional[Text], credentials_config: Optional[CredentialsConfig]
) -> List[InputChannel]:
    """Instantiate the chosen input channel.

    Args:
        channel (optional): The name of the specific input channel to create.
        credentials_config: CredentialsConfig object containing channel credentials.

    Returns:
        A list of instantiated input channels. If a specific channel is provided,
        it returns a list with that single channel. If no channel is specified,
        it returns a list of all channels defined in the credentials file.
    """
    if credentials_config:
        all_credentials = credentials_config.channels
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
        return [
            _create_single_channel(
                channel,
                all_credentials.get(channel),
            )
        ]
    else:
        return [_create_single_channel(c, k) for c, k in all_credentials.items()]


def _create_single_channel(
    channel: Text,
    credentials: Optional[Dict[Text, Any]],
) -> Any:
    """Create a single input channel based on the channel name and credentials.

    Args:
        channel: The name of the input channel to create.
        credentials: The credentials for the input channel.

    Returns:
        An instance of the input channel class.

    Raises:
        RasaException: If the channel class cannot be found or instantiated.
    """
    from rasa.core.channels import BUILTIN_CHANNELS

    # Channels that have optional dependencies
    channels_with_optional_deps = {
        "facebook": "fbmessenger",
        "slack": "slack-sdk",
        "telegram": "aiogram",
        "twilio": "twilio",
        "twilio_voice": "twilio",
        "twilio_media_streams": "twilio",
        "webexteams": "webexteamssdk",
        "vier_cvg": "cvg_sdk",
    }

    if channel in BUILTIN_CHANNELS:
        channel_class = BUILTIN_CHANNELS[channel]

        return channel_class.from_credentials(credentials)
    elif channel in channels_with_optional_deps:
        # Channel is known but not available due to missing dependency
        dependency = channels_with_optional_deps[channel]
        raise RasaException(
            f"Channel '{channel}' is not available "
            f"due to missing '{dependency}' dependency. "
            f"Please install the required extra by running: "
            f"pip install 'rasa-pro[channels]' OR "
            f"poetry add 'rasa-pro[channels]'"
        )
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


def _create_app_without_api(
    cors: Optional[Union[Text, List[Text]]] = None, is_inspector_enabled: bool = False
) -> Sanic:
    app = Sanic("rasa_core_no_api", configure_logging=False)

    # Reset Sanic warnings filter that allows the triggering of Sanic warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sanic.*")

    server.add_root_route(app, is_inspector_enabled)
    server.configure_cors(app, cors)
    return app


def _is_apple_silicon_system() -> bool:
    # check if the system is MacOS
    if platform.system().lower() != "darwin":
        return False
    # check for arm architecture, indicating apple silicon
    return platform.machine().startswith("arm") or os.uname().machine.startswith("arm")


def configure_app(
    input_channels: Optional[List[InputChannel]] = None,
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
    sub_agents: Optional[AvailableAgents] = None,
    log_file: Optional[Text] = None,
    conversation_id: Optional[Text] = uuid.uuid4().hex,
    use_syslog: bool = False,
    syslog_address: Optional[Text] = None,
    syslog_port: Optional[int] = None,
    syslog_protocol: Optional[Text] = None,
    request_timeout: Optional[int] = None,
    server_listeners: Optional[List[Tuple[Callable, Text]]] = None,
    use_uvloop: Optional[bool] = True,
    keep_alive_timeout: int = constants.DEFAULT_KEEP_ALIVE_TIMEOUT,
    is_inspector_enabled: bool = False,
) -> Sanic:
    """Run the agent."""
    rasa.core.utils.configure_file_logging(
        logger, log_file, use_syslog, syslog_address, syslog_port, syslog_protocol
    )

    if enable_api:
        loader = AppLoader(
            factory=partial(
                server.create_app,
                cors_origins=cors,
                auth_token=auth_token,
                response_timeout=response_timeout,
                jwt_secret=jwt_secret,
                jwt_private_key=jwt_private_key,
                jwt_method=jwt_method,
                endpoints=endpoints,
                sub_agents=sub_agents,
                is_inspector_enabled=is_inspector_enabled,
            )
        )
    else:
        loader = AppLoader(factory=partial(_create_app_without_api, cors))

    app = loader.load()
    app.config.KEEP_ALIVE_TIMEOUT = keep_alive_timeout

    if _is_apple_silicon_system() or not use_uvloop:
        app.config.USE_UVLOOP = False
        # some library still sets the loop to uvloop, even if disabled for sanic
        # using uvloop leads to breakingio errors, see
        # https://rasahq.atlassian.net/browse/ENG-667
        asyncio.set_event_loop_policy(None)

    if input_channels:
        channels.channel.register(input_channels, app, route=route)
    else:
        input_channels = []

    if logger.isEnabledFor(logging.DEBUG):
        rasa.core.utils.list_routes(app)

    @app.main_process_start
    async def configure_async_logging(running_app: Sanic) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            rasa.utils.io.enable_async_loop_debugging(asyncio.get_event_loop())

    if "cmdline" in {c.name() for c in input_channels}:

        @app.after_server_start
        async def run_cmdline_io(running_app: Sanic) -> None:
            """Small wrapper to shut down the server once cmd io is done."""
            await console.record_messages(
                server_url=constants.DEFAULT_SERVER_FORMAT.format("http", port),
                sender_id=conversation_id,
                request_timeout=request_timeout,
            )

            logger.info("Killing Sanic server now.")
            running_app.stop()  # kill the sanic server

    if server_listeners:
        for listener, event in server_listeners:
            app.register_listener(listener, event)

    return app


def serve_application(
    model_path: Optional[Text] = None,
    channel: Optional[Text] = None,
    interface: Optional[Text] = constants.DEFAULT_SERVER_INTERFACE,
    port: int = constants.DEFAULT_SERVER_PORT,
    credentials: Optional[CredentialsConfig] = None,
    cors: Optional[Union[Text, List[Text]]] = None,
    auth_token: Optional[Text] = None,
    enable_api: bool = True,
    response_timeout: int = constants.DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_private_key: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
    sub_agents: Optional[AvailableAgents] = None,
    remote_storage: Optional[StorageType] = None,
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
    server_listeners: Optional[List[Tuple[Callable, Text]]] = None,
    inspect: Optional[bool] = False,
    voice: Optional[bool] = False,
) -> None:
    """Run the API entrypoint."""
    if not channel and not credentials:
        channel = "cmdline"

    input_channels = create_input_channels(channel, credentials)

    if inspect:
        logger.info("Starting development inspector.")
        input_channels = [DevelopmentInspectProxy(ic, voice) for ic in input_channels]

        # the inspector needs the api to retrieve slots and flows
        enable_api = True

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
        sub_agents=sub_agents,
        log_file=log_file,
        conversation_id=conversation_id,
        use_syslog=use_syslog,
        syslog_address=syslog_address,
        syslog_port=syslog_port,
        syslog_protocol=syslog_protocol,
        request_timeout=request_timeout,
        server_listeners=server_listeners,
        is_inspector_enabled=inspect,
    )

    ssl_context = server.create_ssl_context(
        ssl_certificate, ssl_keyfile, ssl_ca_file, ssl_password
    )
    protocol = "https" if ssl_context else "http"

    logger.info(f"Starting Rasa server on {protocol}://{interface}:{port}")

    async def load_agent_and_check_failure(app: Sanic, loop: AbstractEventLoop) -> None:
        """Load agent and exit if it fails in non-debug mode."""
        try:
            await load_agent_on_start(
                model_path, endpoints, remote_storage, sub_agents, app, loop
            )
        except Exception as e:
            is_debug = logger.isEnabledFor(logging.DEBUG)
            if is_debug:
                raise e  # show traceback in debug
            # non-debug: log and exit without starting server
            logger.error(f"Failed to load agent: {e}")
            os._exit(1)  # Any other exit method would show a traceback.

    app.register_listener(load_agent_and_check_failure, "before_server_start")

    app.register_listener(
        licensing.validate_limited_server_license, "after_server_start"
    )

    app.register_listener(close_resources, "after_server_stop")

    number_of_workers = rasa.core.utils.number_of_sanic_workers(
        endpoints.lock_store if endpoints else None
    )

    if not inspect:
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
        legacy=True,
    )


# noinspection PyUnusedLocal
async def load_agent_on_start(
    model_path: Text,
    endpoints: AvailableEndpoints,
    remote_storage: Optional[StorageType],
    sub_agents: Optional[AvailableAgents],
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
        sub_agents=sub_agents,
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

    privacy_manager = current_agent.privacy_manager
    if privacy_manager:
        privacy_manager.stop()
