import argparse
import asyncio
import logging
from functools import partial
from typing import List, Optional, Text

from sanic import Sanic
from sanic_cors import CORS

import rasa.core
import rasa.core.cli.arguments
import rasa.utils
from rasa.core import constants, utils, cli
from rasa.core.channels import BUILTIN_CHANNELS, InputChannel, console
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.tracker_store import TrackerStore

logger = logging.getLogger()  # get the root logger


def create_argument_parser():
    """Parse all the command line arguments for the run script."""

    parser = argparse.ArgumentParser(description="starts the bot")
    parser.add_argument(
        "-d", "--core", required=True, type=str, help="core model to run"
    )
    parser.add_argument("-u", "--nlu", type=str, help="nlu model to run")

    cli.arguments.add_logging_option_arguments(parser)
    cli.run.add_run_arguments(parser)
    return parser


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
            input_channel_class = utils.class_from_module_path(channel)
            return input_channel_class.from_credentials(credentials)
        except (AttributeError, ImportError):
            raise Exception(
                "Failed to find input channel class for '{}'. Unknown "
                "input channel. Check your credentials configuration to "
                "make sure the mentioned channel is not misspelled. "
                "If you are creating your own channel, make sure it "
                "is a proper name of a class in a module.".format(channel)
            )


def configure_app(
    input_channels=None,
    cors=None,
    auth_token=None,
    enable_api=True,
    jwt_secret=None,
    jwt_method=None,
    route="/webhooks/",
    port=None,
):
    """Run the agent."""
    from rasa.core import server

    if enable_api:
        app = server.create_app(
            cors_origins=cors,
            auth_token=auth_token,
            jwt_secret=jwt_secret,
            jwt_method=jwt_method,
        )
    else:
        app = Sanic(__name__)
        CORS(app, resources={r"/*": {"origins": cors or ""}}, automatic_options=True)

    if input_channels:
        rasa.core.channels.channel.register(input_channels, app, route=route)
    else:
        input_channels = []

    if logger.isEnabledFor(logging.DEBUG):
        utils.list_routes(app)

    # configure async loop logging
    async def configure_logging():
        if logger.isEnabledFor(logging.DEBUG):
            rasa.utils.io.enable_async_loop_debugging(asyncio.get_event_loop())

    app.add_task(configure_logging)

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
    core_model=None,
    nlu_model=None,
    channel=None,
    port=constants.DEFAULT_SERVER_PORT,
    credentials=None,
    cors=None,
    auth_token=None,
    enable_api=True,
    jwt_secret=None,
    jwt_method=None,
    endpoints=None,
):
    if not channel and not credentials:
        channel = "cmdline"

    input_channels = create_http_input_channels(channel, credentials)

    app = configure_app(
        input_channels, cors, auth_token, enable_api, jwt_secret, jwt_method, port=port
    )

    logger.info(
        "Starting Rasa Core server on "
        "{}".format(constants.DEFAULT_SERVER_FORMAT.format(port))
    )

    app.register_listener(
        partial(load_agent_on_start, core_model, endpoints, nlu_model),
        "before_server_start",
    )
    app.run(host="0.0.0.0", port=port, access_log=logger.isEnabledFor(logging.DEBUG))


# noinspection PyUnusedLocal
async def load_agent_on_start(core_model, endpoints, nlu_model, app, loop):
    """Load an agent.

    Used to be scheduled on server start
    (hence the `app` and `loop` arguments)."""
    from rasa.core import broker
    from rasa.core.agent import Agent

    _interpreter = NaturalLanguageInterpreter.create(nlu_model, endpoints.nlu)
    _broker = broker.from_endpoint_config(endpoints.event_broker)

    _tracker_store = TrackerStore.find_tracker_store(
        None, endpoints.tracker_store, _broker
    )

    if endpoints and endpoints.model:
        from rasa.core import agent

        app.agent = Agent(
            interpreter=_interpreter,
            generator=endpoints.nlg,
            tracker_store=_tracker_store,
            action_endpoint=endpoints.action,
        )

        await agent.load_from_server(app.agent, model_server=endpoints.model)
    else:
        app.agent = Agent.load(
            core_model,
            interpreter=_interpreter,
            generator=endpoints.nlg,
            tracker_store=_tracker_store,
            action_endpoint=endpoints.action,
        )

    return app.agent


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.core.run` directly is "
        "no longer supported. "
        "Please use `rasa shell` instead."
    )
