import argparse
import os
import webbrowser
from asyncio import AbstractEventLoop
from typing import List, Optional, Text
from urllib.parse import urlencode

import structlog
from sanic import Sanic

from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import shell as arguments
from rasa.cli.arguments.default_arguments import add_sub_agents_param
from rasa.core import constants
from rasa.core.config.configuration import (
    Configuration,
    CredentialsConfigPath,
    EndpointsConfigPath,
)
from rasa.core.config.credentials import CredentialsConfig
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.exceptions import ModelNotFound
from rasa.model import get_local_model
from rasa.shared.utils.cli import print_error
from rasa.utils.cli import remove_argument_from_parser

structlogger = structlog.get_logger()


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all inspect parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    inspect_parser = subparsers.add_parser(
        "inspect",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=(
            "Loads your trained model and lets you talk to your "
            "assistant in the browser."
        ),
    )
    inspect_parser.set_defaults(func=inspect)
    arguments.set_shell_arguments(inspect_parser)

    # additional argument for voice (only applicable with --legacy)
    inspect_parser.add_argument(
        "--voice",
        help="Enable voice (only applicable with --legacy).",
        action="store_true",
        default=False,
    )
    inspect_parser.add_argument(
        "--legacy",
        help="Use the legacy inspector UI (socketio-based).",
        action="store_true",
        default=False,
    )

    add_sub_agents_param(inspect_parser)

    # it'd be confusing to expose those arguments to the user,
    # so we remove them
    remove_argument_from_parser(inspect_parser, "--credentials")
    remove_argument_from_parser(inspect_parser, "--connector")
    remove_argument_from_parser(inspect_parser, "--enable-api")


async def open_inspector_in_browser(
    server_url: Text,
    legacy: bool = False,
    voice: bool = False,
    token: Optional[Text] = None,
) -> None:
    """Opens the rasa inspector in the default browser."""
    dev_port = os.environ.get("RASA_INSPECTOR_DEV_PORT")
    query_params: dict = {"projectUrl": server_url}
    if token:
        query_params["token"] = token
    query_string = f"?{urlencode(query_params)}"

    if dev_port:
        webbrowser.open(f"http://localhost:{dev_port}{query_string}")
    elif legacy and voice:
        webbrowser.open(
            f"{server_url}/webhooks/browser_audio/inspect.html{query_string}"
        )
    elif legacy:
        webbrowser.open(f"{server_url}/webhooks/socketio/inspect.html{query_string}")
    else:
        webbrowser.open(f"{server_url}/webhooks/inspector/inspect.html{query_string}")


def inspect(args: argparse.Namespace) -> None:
    """Inspect the bot using the most recent model."""
    import rasa.cli.run
    from rasa.cli.validation.config_path_validation import get_validated_path
    from rasa.shared.constants import DEFAULT_MODELS_PATH

    _credentials_path = CredentialsConfigPath.validate()
    _credentials = (
        CredentialsConfig.load_from_file(_credentials_path)
        if _credentials_path
        else None
    )
    _inspector_server_url = (
        (_credentials.channels.get("inspector") or {}).get("server_url")
        if _credentials
        else None
    )

    async def after_start_hook_open_inspector(_: Sanic, __: AbstractEventLoop) -> None:
        """Hook to open the browser on server start."""
        server_url = _inspector_server_url
        if server_url == "0.0.0.0" or server_url is None:
            server_url = constants.DEFAULT_SERVER_FORMAT.format("http", args.port)
        await open_inspector_in_browser(
            server_url, args.legacy, args.voice, args.auth_token
        )

    # the following arguments are not exposed to the user
    if args.voice and not args.legacy:
        structlogger.warning(
            "inspect.voice_requires_legacy",
            event_info=(
                "--voice has no effect without --legacy. "
                "The default inspector already includes voice support."
            ),
        )
    if args.voice and args.legacy:
        args.connector = "browser_audio"
    elif args.legacy:
        args.connector = "socketio"
    else:
        args.connector = "inspector"
    args.enable_api = True
    args.inspect = True
    args.credentials = None
    args.inspector_server_url = _inspector_server_url
    args.server_listeners = [(after_start_hook_open_inspector, "after_server_start")]
    dev_port = os.environ.get("RASA_INSPECTOR_DEV_PORT")
    if dev_port:
        inspector_frontend_url = f"http://localhost:{dev_port}"
        if args.cors:
            args.cors.append(inspector_frontend_url)
        else:
            args.cors = [inspector_frontend_url]

    model = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

    # Load endpoints with proper endpoint file location
    # This will initialise the endpoints singleton properly so that
    # it can be used safely throughout the codebase with
    # `Configuration.get_instance().endpoints`
    Configuration.initialise_endpoints(
        endpoints_path=EndpointsConfigPath.validate(args.endpoints)
    )
    Configuration.initialise_sub_agents(args.sub_agents)

    try:
        model = get_local_model(model)
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `rasa train`."
        )
        return

    metadata = LocalModelStorage.metadata_from_archive(model)

    telemetry.track_inspect_started(args.connector, metadata.assistant_id)
    rasa.cli.run.run(args)
