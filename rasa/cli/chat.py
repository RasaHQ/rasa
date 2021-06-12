import argparse
import functools
import logging
from typing import List
import uuid
import webbrowser
from sanic import Sanic
from rasa.cli import SubParsersAction
from rasa.cli.arguments import shell as arguments

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


async def open_browser(port, app: Sanic, loop):
    webbrowser.open(f"http://localhost:{port}/webhooks/socketio/chat.html")


def chat(args: argparse.Namespace) -> None:
    import rasa.cli.run

    args.connector = "socketio"

    args.server_listeners = [
        (functools.partial(open_browser, args.port), "after_server_start")
    ]

    rasa.cli.run.run(args)
