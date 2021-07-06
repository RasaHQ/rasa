# this builtin is needed so we can overwrite in test
import asyncio
import json
import logging
import os

import aiohttp
import questionary
from aiohttp import ClientTimeout
from prompt_toolkit.styles import Style
from typing import Any, Generator
from typing import Text, Optional, Dict, List

import rasa.shared.utils.cli
import rasa.shared.utils.io
from rasa.cli import utils as cli_utils
from rasa.core import utils
from rasa.core.channels.rest import RestInput
from rasa.core.constants import DEFAULT_SERVER_URL
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.utils.io import DEFAULT_ENCODING

logger = logging.getLogger(__name__)

STREAM_READING_TIMEOUT_ENV = "RASA_SHELL_STREAM_READING_TIMEOUT_IN_SECONDS"
DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS = 10


def print_buttons(
    message: Dict[Text, Any],
    is_latest_message: bool = False,
    color: Text = rasa.shared.utils.io.bcolors.OKBLUE,
) -> Optional[questionary.Question]:
    if is_latest_message:
        choices = cli_utils.button_choices_from_message_data(
            message, allow_free_text_input=True
        )
        question = questionary.select(
            message.get("text"),
            choices,
            style=Style([("qmark", "#6d91d3"), ("", "#6d91d3"), ("answer", "#b373d6")]),
        )
        return question
    else:
        rasa.shared.utils.cli.print_color("Buttons:", color=color)
        for idx, button in enumerate(message.get("buttons")):
            rasa.shared.utils.cli.print_color(
                cli_utils.button_to_string(button, idx), color=color
            )


def print_bot_output(
    message: Dict[Text, Any],
    is_latest_message: bool = False,
    color: Text = rasa.shared.utils.io.bcolors.OKBLUE,
) -> Optional[questionary.Question]:
    if "buttons" in message:
        question = print_buttons(message, is_latest_message, color)
        if question:
            return question

    if "text" in message:
        rasa.shared.utils.cli.print_color(message.get("text"), color=color)

    if "image" in message:
        rasa.shared.utils.cli.print_color("Image: " + message.get("image"), color=color)

    if "attachment" in message:
        rasa.shared.utils.cli.print_color(
            "Attachment: " + message.get("attachment"), color=color
        )

    if "elements" in message:
        rasa.shared.utils.cli.print_color("Elements:", color=color)
        for idx, element in enumerate(message.get("elements")):
            rasa.shared.utils.cli.print_color(
                cli_utils.element_to_string(element, idx), color=color
            )

    if "quick_replies" in message:
        rasa.shared.utils.cli.print_color("Quick Replies:", color=color)
        for idx, element in enumerate(message.get("quick_replies")):
            rasa.shared.utils.cli.print_color(
                cli_utils.button_to_string(element, idx), color=color
            )

    if "custom" in message:
        rasa.shared.utils.cli.print_color("Custom json:", color=color)
        rasa.shared.utils.cli.print_color(
            json.dumps(message.get("custom"), indent=2), color=color
        )


def get_user_input(previous_response: Optional[Dict[str, Any]]) -> Optional[Text]:
    button_response = None
    if previous_response is not None:
        button_response = print_bot_output(previous_response, is_latest_message=True)

    if button_response is not None:
        response = cli_utils.payload_from_button_question(button_response)
        if response == cli_utils.FREE_TEXT_INPUT_PROMPT:
            # Re-prompt user with a free text input
            response = get_user_input({})
    else:
        response = questionary.text(
            "",
            qmark="Your input ->",
            style=Style([("qmark", "#b373d6"), ("", "#b373d6")]),
        ).ask()
    return response.strip() if response is not None else None


async def send_message_receive_block(
    server_url: Text, auth_token: Text, sender_id: Text, message: Text
) -> List[Dict[Text, Any]]:
    payload = {"sender": sender_id, "message": message}

    url = f"{server_url}/webhooks/rest/webhook?token={auth_token}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, raise_for_status=True) as resp:
            return await resp.json()


async def _send_message_receive_stream(
    server_url: Text, auth_token: Text, sender_id: Text, message: Text
) -> Generator[Dict[Text, Any], None, None]:
    payload = {"sender": sender_id, "message": message}

    url = f"{server_url}/webhooks/rest/webhook?stream=true&token={auth_token}"

    # Define timeout to not keep reading in case the server crashed in between
    timeout = _get_stream_reading_timeout()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, raise_for_status=True) as resp:

            async for line in resp.content:
                if line:
                    yield json.loads(line.decode(DEFAULT_ENCODING))


def _get_stream_reading_timeout() -> ClientTimeout:
    timeout_in_seconds = int(
        os.environ.get(
            STREAM_READING_TIMEOUT_ENV, DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS
        )
    )

    return ClientTimeout(timeout_in_seconds)


async def record_messages(
    sender_id: Text,
    server_url: Text = DEFAULT_SERVER_URL,
    auth_token: Text = "",
    max_message_limit: Optional[int] = None,
    use_response_stream: bool = True,
) -> int:
    """Read messages from the command line and print bot responses."""

    exit_text = INTENT_MESSAGE_PREFIX + "stop"

    rasa.shared.utils.cli.print_success(
        "Bot loaded. Type a message and press enter "
        "(use '{}' to exit): ".format(exit_text)
    )

    num_messages = 0
    previous_response = None
    await asyncio.sleep(0.5)  # Wait for server to start
    while not utils.is_limit_reached(num_messages, max_message_limit):
        text = get_user_input(previous_response)

        if text == exit_text or text is None:
            break

        if use_response_stream:
            bot_responses = _send_message_receive_stream(
                server_url, auth_token, sender_id, text
            )
            previous_response = None
            async for response in bot_responses:
                if previous_response is not None:
                    print_bot_output(previous_response)
                previous_response = response
        else:
            bot_responses = await send_message_receive_block(
                server_url, auth_token, sender_id, text
            )
            previous_response = None
            for response in bot_responses:
                if previous_response is not None:
                    print_bot_output(previous_response)
                previous_response = response

        num_messages += 1
        await asyncio.sleep(0)  # Yield event loop for others coroutines
    return num_messages


class CmdlineInput(RestInput):
    @classmethod
    def name(cls) -> Text:
        return "cmdline"

    def url_prefix(self) -> Text:
        return RestInput.name()
