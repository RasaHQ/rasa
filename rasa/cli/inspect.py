import argparse
import os
import webbrowser
from asyncio import AbstractEventLoop
from typing import List, Optional, Text
from urllib.parse import urlencode

from sanic import Sanic

from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import shell as arguments
from rasa.cli.arguments.default_arguments import add_sub_agents_param
from rasa.core import constants
from rasa.core.config.configuration import Configuration, EndpointsConfigPath
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.exceptions import ModelNotFound
from rasa.model import get_local_model
from rasa.shared.utils.cli import print_error
from rasa.utils.cli import remove_argument_from_parser


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

    # additional argument for voice
    inspect_parser.add_argument(
        "--voice", help="Enable voice", action="store_true", default=False
    )
    inspect_parser.add_argument(
        "--nextgen",
        help="Serve the next-gen inspector UI.",
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
    voice: bool = False,
    nextgen: bool = False,
    token: Optional[Text] = None,
) -> None:
    """Opens the rasa inspector in the default browser."""
    dev_port = os.environ.get("RASA_INSPECTOR_DEV_PORT")
    query_params = {"projectUrl": server_url}
    if dev_port:
        query_string = f"?{urlencode(query_params)}"
        webbrowser.open(f"http://localhost:{dev_port}{query_string}")
    else:
        if voice:
            channel = "browser_audio"
        elif nextgen:
            channel = "inspector"
        else:
            channel = "socketio"
        if token:
            query_params["token"] = token
        query_string = f"?{urlencode(query_params)}"
        webbrowser.open(f"{server_url}/webhooks/{channel}/inspect.html{query_string}")


def inspect(args: argparse.Namespace) -> None:
    """Inspect the bot using the most recent model."""
    import rasa.cli.run
    from rasa.cli.validation.config_path_validation import get_validated_path
    from rasa.shared.constants import DEFAULT_MODELS_PATH

    async def after_start_hook_open_inspector(_: Sanic, __: AbstractEventLoop) -> None:
        """Hook to open the browser on server start."""
        server_url = constants.DEFAULT_SERVER_FORMAT.format("http", args.port)
        await open_inspector_in_browser(
            server_url, args.voice, args.nextgen, args.auth_token
        )

    # the following arguments are not exposed to the user
    if args.voice:
        args.connector = "browser_audio"
    elif args.nextgen:
        args.connector = "inspector"
    else:
        args.connector = "socketio"
    args.enable_api = True
    args.inspect = True
    args.credentials = None
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
