import argparse
from asyncio import AbstractEventLoop
import logging
from typing import List, Text
import uuid
import webbrowser

from sanic import Sanic

from rasa.cli import SubParsersAction
from rasa.cli.arguments import shell as arguments
from rasa.core import constants

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all chat parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    chat_parser = subparsers.add_parser(
        "chat",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=(
            "Loads your trained model and lets you talk to your "
            "assistant in the browser."
        ),
    )
    chat_parser.set_defaults(func=chat)

    chat_parser.add_argument(
        "--conversation-id",
        default=uuid.uuid4().hex,
        required=False,
        help="Set the conversation ID.",
    )

    arguments.set_shell_arguments(chat_parser)


async def open_chat_in_browser(server_url: Text) -> None:
    """Opens the rasa chat in the default browser."""
    webbrowser.open(f"{server_url}/webhooks/socketio/chat.html")


def chat(args: argparse.Namespace) -> None:
    """Chat to the bot using the most recent model."""
    import rasa.cli.run

    args.connector = "socketio"

    async def after_start_hook_open_chat(_: Sanic, __: AbstractEventLoop) -> None:
        """Hook to open the browser on server start."""
        server_url = constants.DEFAULT_SERVER_FORMAT.format("http", args.port)
        await open_chat_in_browser(server_url)

    args.server_listeners = [(after_start_hook_open_chat, "after_server_start")]

    rasa.cli.run.run(args)
