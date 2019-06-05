import asyncio
import logging
from functools import partial
from typing import List, Optional, Text, Union

from sanic import Sanic
from sanic_cors import CORS

import rasa.core
import rasa.utils
import rasa.utils.io
from rasa.core import constants, utils
from rasa.core.agent import load_agent, Agent
from rasa.core.channels import BUILTIN_CHANNELS, InputChannel, console
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.tracker_store import TrackerStore
from rasa.core.utils import AvailableEndpoints
from rasa.model import get_model_subdirectories, get_model
from rasa.utils.common import update_sanic_log_level, class_from_module_path

logger = logging.getLogger()  # get the root logger


def create_http_input_channels(
    channel: Optional[Text], credentials_file: Optional[Text]
) -> List["InputChannel"]:
    """Instantiate the chosen input channel."""

    if credentials_file:
        all_credentials = rasa.utils.io.read_yaml_file(credentials_file)
    else:
        all_credentials = {}

    if channel:
        return [_create_single_channel(channel, all_credentials.get(channel))]
    else:
        return [_create_single_channel(c, k) for c, k in all_credentials.items()]


def _create_single_channel(channel, credentials):
    from rasa.core.channels import BUILTIN_CHANNELS

    if channel in BUILTIN_CHANNELS:
        return BUILTIN_CHANNELS[channel].from_credentials(credentials)
    else:
        # try to load channel based on class name
        try:
            input_channel_class = class_from_module_path(channel)
            return input_channel_class.from_credentials(credentials)
        except (AttributeError, ImportError):
            raise Exception(
                "Failed to find input channel class for '{}'. Unknown "
                "input channel. Check your credentials configuration to "
                "make sure the mentioned channel is not misspelled. "
                "If you are creating your own channel, make sure it "
                "is a proper name of a class in a module.".format(channel)
            )


def _configure_logging(log_file: Optional[Text]):
    if log_file is not None:
        formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def configure_app(
    input_channels: Optional[List["InputChannel"]] = None,
    cors: Optional[Union[Text, List[Text]]] = None,
    auth_token: Optional[Text] = None,
    enable_api: bool = True,
    jwt_secret: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    route: Optional[Text] = "/webhooks/",
    port: int = constants.DEFAULT_SERVER_PORT,
    log_file: Optional[Text] = None,
):
    """Run the agent."""
    from rasa import server

    if enable_api:
        app = server.create_app(
            cors_origins=cors,
            auth_token=auth_token,
            jwt_secret=jwt_secret,
            jwt_method=jwt_method,
        )
    else:
        app = Sanic(__name__, configure_logging=False)
        CORS(app, resources={r"/*": {"origins": cors or ""}}, automatic_options=True)

    _configure_logging(log_file)

    if input_channels:
        rasa.core.channels.channel.register(input_channels, app, route=route)
    else:
        input_channels = []

    if logger.isEnabledFor(logging.DEBUG):
        utils.list_routes(app)

    # configure async loop logging
    async def configure_async_logging():
        if logger.isEnabledFor(logging.DEBUG):
            rasa.utils.io.enable_async_loop_debugging(asyncio.get_event_loop())

    app.add_task(configure_async_logging)

    if "cmdline" in {c.name() for c in input_channels}:

        async def run_cmdline_io(running_app: Sanic):
            """Small wrapper to shut down the server once cmd io is done."""
            await asyncio.sleep(1)  # allow server to start
            await console.record_messages(
                server_url=constants.DEFAULT_SERVER_FORMAT.format(port)
            )

            logger.info("Killing Sanic server now.")
            running_app.stop()  # kill the sanic serverx

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
    jwt_secret: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
    remote_storage: Optional[Text] = None,
    log_file: Optional[Text] = None,
):
    if not channel and not credentials:
        channel = "cmdline"

    input_channels = create_http_input_channels(channel, credentials)

    app = configure_app(
        input_channels,
        cors,
        auth_token,
        enable_api,
        jwt_secret,
        jwt_method,
        port=port,
        log_file=log_file,
    )

    logger.info(
        "Starting Rasa Core server on "
        "{}".format(constants.DEFAULT_SERVER_FORMAT.format(port))
    )

    app.register_listener(
        partial(load_agent_on_start, model_path, endpoints, remote_storage),
        "before_server_start",
    )

    update_sanic_log_level(log_file)

    app.run(host="0.0.0.0", port=port)


# noinspection PyUnusedLocal
async def load_agent_on_start(
    model_path: Text,
    endpoints: Optional[AvailableEndpoints],
    remote_storage: Optional[Text],
    app: Sanic,
    loop: Text,
):
    """Load an agent.

    Used to be scheduled on server start
    (hence the `app` and `loop` arguments)."""
    from rasa.core import broker

    try:
        _, nlu_model = get_model_subdirectories(get_model(model_path))
        _interpreter = NaturalLanguageInterpreter.create(nlu_model, endpoints.nlu)
    except Exception:
        logger.debug("Could not load interpreter from '{}'".format(model_path))
        _interpreter = None

    _broker = broker.from_endpoint_config(endpoints.event_broker)
    _tracker_store = TrackerStore.find_tracker_store(
        None, endpoints.tracker_store, _broker
    )

    model_server = endpoints.model if endpoints and endpoints.model else None

    app.agent = await load_agent(
        model_path,
        model_server=model_server,
        remote_storage=remote_storage,
        interpreter=_interpreter,
        generator=endpoints.nlg,
        tracker_store=_tracker_store,
        action_endpoint=endpoints.action,
    )

    if not app.agent:
        logger.error(
            "Agent could not be loaded with the provided configuration."
            "Load default agent without any model."
        )
        app.agent = Agent(
            interpreter=_interpreter,
            generator=endpoints.nlg,
            tracker_store=_tracker_store,
            action_endpoint=endpoints.action,
            model_server=model_server,
            remote_storage=remote_storage,
        )

    return app.agent


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.core.run` directly is no longer supported. "
        "Please use `rasa run` to start a Rasa server or `rasa shell` to chat with "
        "your bot on the command line."
    )
