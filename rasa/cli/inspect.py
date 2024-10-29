import argparse
import webbrowser
from asyncio import AbstractEventLoop
from typing import List, Text

from sanic import Sanic

from rasa.cli import SubParsersAction
from rasa.cli.arguments import shell as arguments
from rasa.cli.arguments.default_arguments import add_skip_validation_flag
from rasa.core import constants
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
    add_skip_validation_flag(inspect_parser)

    inspect_parser.add_argument("--voice", help="Enable voice", action="store_true")

    # it'd be confusing to expose those arguments to the user,
    # so we remove them
    remove_argument_from_parser(inspect_parser, "--credentials")
    remove_argument_from_parser(inspect_parser, "--connector")
    remove_argument_from_parser(inspect_parser, "--enable-api")


async def open_inspector_in_browser(server_url: Text, voice: bool = False) -> None:
    """Opens the rasa inspector in the default browser."""
    channel = "socketio" if not voice else "browser_audio"
    webbrowser.open(f"{server_url}/webhooks/{channel}/inspect.html")


def inspect(args: argparse.Namespace) -> None:
    """Inspect the bot using the most recent model."""
    import rasa.cli.run

    async def after_start_hook_open_inspector(_: Sanic, __: AbstractEventLoop) -> None:
        """Hook to open the browser on server start."""
        server_url = constants.DEFAULT_SERVER_FORMAT.format("http", args.port)
        await open_inspector_in_browser(server_url, args.voice)

    # the following arguments are not exposed to the user
    if args.voice:
        args.connector = "browser_audio"
    else:
        args.connector = "rasa.core.channels.socketio.SocketIOInput"
    args.enable_api = True
    args.inspect = True
    args.credentials = None
    args.server_listeners = [(after_start_hook_open_inspector, "after_server_start")]

    rasa.cli.run.run(args)
