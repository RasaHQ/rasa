# this builtin is needed so we can overwrite in test
import aiohttp
import json
import logging
import questionary
from typing import Text, Optional

from aiohttp import ClientTimeout
from async_generator import async_generator, yield_
from prompt_toolkit.styles import Style

from rasa.cli import utils as cli_utils

from rasa.core import utils
from rasa.core.channels.channel import UserMessage
from rasa.core.channels.channel import RestInput
from rasa.core.constants import DEFAULT_SERVER_URL
from rasa.core.interpreter import INTENT_MESSAGE_PREFIX

logger = logging.getLogger(__name__)


DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS = 10


def print_bot_output(
    message, color=cli_utils.bcolors.OKBLUE
) -> Optional[questionary.Question]:
    if ("text" in message) and not ("buttons" in message):
        cli_utils.print_color(message.get("text"), color=color)

    if "image" in message:
        cli_utils.print_color("Image: " + message.get("image"), color=color)

    if "attachment" in message:
        cli_utils.print_color("Attachment: " + message.get("attachment"), color=color)

    if "buttons" in message:
        choices = cli_utils.button_choices_from_message_data(
            message, allow_free_text_input=True
        )

        question = questionary.select(
            message.get("text"),
            choices,
            style=Style([("qmark", "#6d91d3"), ("", "#6d91d3"), ("answer", "#b373d6")]),
        )
        return question

    if "elements" in message:
        cli_utils.print_color("Elements:", color=color)
        for idx, element in enumerate(message.get("elements")):
            cli_utils.print_color(
                cli_utils.element_to_string(element, idx), color=color
            )

    if "quick_replies" in message:
        cli_utils.print_color("Quick Replies:", color=color)
        for idx, element in enumerate(message.get("quick_replies")):
            cli_utils.print_color(cli_utils.button_to_string(element, idx), color=color)

    if "custom" in message:
        cli_utils.print_color("Custom json:", color=color)
        cli_utils.print_color(json.dumps(message.get("custom"), indent=2), color=color)


def get_user_input(button_question: questionary.Question) -> Optional[Text]:
    if button_question is not None:
        response = cli_utils.payload_from_button_question(button_question)
        if response == cli_utils.FREE_TEXT_INPUT_PROMPT:
            # Re-prompt user with a free text input
            response = get_user_input(None)
    else:
        response = questionary.text(
            "",
            qmark="Your input ->",
            style=Style([("qmark", "#b373d6"), ("", "#b373d6")]),
        ).ask()
    return response.strip() if response is not None else None


async def send_message_receive_block(server_url, auth_token, sender_id, message):
    payload = {"sender": sender_id, "message": message}

    url = "{}/webhooks/rest/webhook?token={}".format(server_url, auth_token)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, raise_for_status=True) as resp:
            return await resp.json()


@async_generator  # needed for python 3.5 compatibility
async def send_message_receive_stream(server_url, auth_token, sender_id, message):
    payload = {"sender": sender_id, "message": message}

    url = "{}/webhooks/rest/webhook?stream=true&token={}".format(server_url, auth_token)

    # Define timeout to not keep reading in case the server crashed in between
    timeout = ClientTimeout(DEFAULT_STREAM_READING_TIMEOUT_IN_SECONDS)
    # TODO: check if this properly receives UTF-8 data
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, raise_for_status=True) as resp:

            async for line in resp.content:
                if line:
                    await yield_(json.loads(line.decode("utf-8")))


async def record_messages(
    server_url=DEFAULT_SERVER_URL,
    auth_token=None,
    sender_id=UserMessage.DEFAULT_SENDER_ID,
    max_message_limit=None,
    use_response_stream=True,
):
    """Read messages from the command line and print bot responses."""

    auth_token = auth_token if auth_token else ""

    exit_text = INTENT_MESSAGE_PREFIX + "stop"

    cli_utils.print_success(
        "Bot loaded. Type a message and press enter "
        "(use '{}' to exit): ".format(exit_text)
    )

    num_messages = 0
    button_question = None
    while not utils.is_limit_reached(num_messages, max_message_limit):
        text = get_user_input(button_question)

        if text == exit_text or text is None:
            break

        if use_response_stream:
            bot_responses = send_message_receive_stream(
                server_url, auth_token, sender_id, text
            )
            async for response in bot_responses:
                button_question = print_bot_output(response)
        else:
            bot_responses = await send_message_receive_block(
                server_url, auth_token, sender_id, text
            )
            for response in bot_responses:
                button_question = print_bot_output(response)

        num_messages += 1
    return num_messages


class CmdlineInput(RestInput):
    @classmethod
    def name(cls):
        return "cmdline"

    def url_prefix(self):
        return RestInput.name()
