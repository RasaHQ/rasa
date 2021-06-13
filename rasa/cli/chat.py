import argparse
from asyncio import AbstractEventLoop
import functools
import logging
from typing import List
import uuid
import webbrowser

from sanic import Sanic


from rasa.cli import SubParsersAction
from rasa.cli.arguments import chat as chat_arguments, shell as shell_arguments
import rasa.cli.utils
from rasa.constants import DEFAULT_SERVER_HOST
import rasa.core.channels.bridge
from rasa.core.utils import AvailableEndpoints
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH
from rasa.shared.exceptions import FileNotFoundException
import rasa.shared.utils.io
from rasa.core.channels.bridge import Webhook, base_url
import rasa.shared.utils.cli
from rasa.utils.endpoints import EndpointConfig

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

    shell_arguments.set_shell_arguments(chat_parser)
    chat_arguments.add_share_param(chat_parser)


def setup_rasa_bridge():
    from rasa.core.channels import bridge
    from nacl.encoding import Base64Encoder
    import questionary

    confirm = questionary.confirm(
        "Couldn't find a Rasa Bridge configuration yet, do you want to set it up?"
    ).ask()
    if not confirm:
        rasa.shared.utils.cli.print_error_and_exit(
            "Can't use Rasa Bridge without a configuration. Please either run "
            "without the bridge or configure it before running the server."
        )

    webhook = bridge.create_webhook()
    if not webhook:
        logger.error("Couldn't create webhook :(")
        return

    secret = webhook.encryption_key.encode(Base64Encoder).decode("utf-8")

    rasa.shared.utils.cli.print_success("Successfully created a Rasa Bridge.")
    should_add = questionary.confirm(
        "The Rasa Bridge configuration needs to be "
        "added to your endpoints.yml - Continue?"
    ).ask()

    if not should_add:
        rasa.shared.utils.cli.print_success(
            f"Use the following configuration to enable it in "
            f"your endpoints.yml:\n"
            f"bridge:\n"
            f"  token: {webhook.token}\n"
            f"  secret: {secret}"
        )
        rasa.shared.utils.cli.print_error_and_exit(
            "Please configure the bridge manually and restart the server."
        )

    rasa.shared.utils.cli.print_success(
        f"You can use {bridge.url_for_webhook(webhook)} to configure "
        f"your channels webhook."
    )

    # TODO: this is just a hack - won't necessarily end up in the endpoints.yml
    endpoints_path = "endpoints.yml"
    try:
        endpoints = rasa.shared.utils.io.read_yaml_file(endpoints_path)
    except FileNotFoundException:
        endpoints = {}
    endpoints["bridge"] = {"token": webhook.token, "secret": secret}
    rasa.shared.utils.io.write_yaml(endpoints, endpoints_path)
    return EndpointConfig(token=webhook.token, secret=secret)


def chat_for_webhook(webhook: Webhook):
    return f"{base_url}/chat/{webhook.token}"


async def open_browser(port, _: Sanic, __: AbstractEventLoop):
    webbrowser.open(f"http://localhost:{port}/webhooks/socketio/chat.html")


def chat(args: argparse.Namespace) -> None:
    import rasa.cli.run

    if args.share:
        args.endpoints = rasa.cli.utils.get_validated_path(
            args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
        )
        endpoints = AvailableEndpoints.read_endpoints(args.endpoints)

        if not endpoints or not endpoints.bridge:
            bridge_endpoint = setup_rasa_bridge()
        else:
            bridge_endpoint = endpoints.bridge

        if not bridge_endpoint:
            raise Exception("Failed to create bridge.")

        async def run_bridge_io(running_app: Sanic, loop: AbstractEventLoop):
            """Small wrapper to shut down the server once bridge io is done."""

            access_token = rasa.core.channels.bridge.retrieve_access_token()
            webhook = rasa.core.channels.bridge.webhook_from_endpoint(bridge_endpoint)
            await rasa.core.channels.bridge.retrieve_webhook_calls(
                access_token, webhook, running_app, f"{DEFAULT_SERVER_HOST}:{args.port}"
            )
            rasa.shared.utils.cli.print_success(
                f"You can use {chat_for_webhook(webhook)} to chat to your bot."
            )

        args.server_listeners = [(run_bridge_io, "after_server_start")]
    else:
        args.connector = "socketio"

        args.server_listeners = [
            (functools.partial(open_browser, args.port), "after_server_start")
        ]

    rasa.cli.run.run(args)
