# this builtin is needed so we can overwrite in test
import aiohttp

import questionary

import json
from prompt_toolkit.styles import Style

from rasa_core import utils
from rasa_core.channels import UserMessage
from rasa_core.channels.channel import (
    button_to_string, element_to_string,
    RestInput)
from rasa_core.constants import DEFAULT_SERVER_URL
from rasa_core.interpreter import INTENT_MESSAGE_PREFIX


def print_bot_output(message, color=utils.bcolors.OKBLUE):
    if "text" in message:
        utils.print_color(message.get("text"), color)

    if "image" in message:
        utils.print_color("Image: " + message.get("image"), color)

    if "attachment" in message:
        utils.print_color("Attachment: " + message.get("attachment"), color)

    if "buttons" in message:
        utils.print_color("Buttons:", color)
        for idx, button in enumerate(message.get("buttons")):
            utils.print_color(button_to_string(button, idx), color)

    if "elements" in message:
        for idx, element in enumerate(message.get("elements")):
            element_str = "Elements:\n" + element_to_string(element, idx)
            utils.print_color(element_str, color)


def get_cmd_input():
    return questionary.text("",
                            qmark="Your input ->",
                            style=Style([('qmark', '#b373d6'),
                                         ('', '#b373d6')])).ask().strip()


async def send_message_receive_block(server_url,
                                     auth_token,
                                     sender_id,
                                     message):
    payload = {
        "sender": sender_id,
        "message": message
    }

    url = "{}/webhooks/rest/webhook?token={}".format(server_url, auth_token)
    async with aiohttp.ClientSession() as session:
        async with session.post(url,
                                json=payload,
                                raise_for_status=True) as resp:
            return await resp.json()


async def send_message_receive_stream(server_url,
                                      auth_token,
                                      sender_id,
                                      message):
    payload = {
        "sender": sender_id,
        "message": message
    }

    url = "{}/webhooks/rest/webhook?stream=true&token={}".format(
        server_url, auth_token)

    # TODO: check if this properly receives UTF-8 data
    async with aiohttp.ClientSession() as session:
        async with session.post(url,
                                json=payload,
                                raise_for_status=True) as resp:

            async for line in resp.content:
                if line:
                    yield json.loads(line)


async def record_messages(server_url=DEFAULT_SERVER_URL,
                          auth_token=None,
                          sender_id=UserMessage.DEFAULT_SENDER_ID,
                          max_message_limit=None,
                          use_response_stream=True,
                          on_finish=None):
    """Read messages from the command line and print bot responses."""

    auth_token = auth_token if auth_token else ""

    exit_text = INTENT_MESSAGE_PREFIX + 'stop'

    utils.print_color("Bot loaded. Type a message and press enter "
                      "(use '{}' to exit): ".format(exit_text),
                      utils.bcolors.OKGREEN)

    num_messages = 0
    while not utils.is_limit_reached(num_messages, max_message_limit):
        text = get_cmd_input()
        if text == exit_text:
            break

        if use_response_stream:
            bot_responses = send_message_receive_stream(server_url,
                                                        auth_token,
                                                        sender_id, text)
            async for response in bot_responses:
                print_bot_output(response)
        else:
            bot_responses = await send_message_receive_block(server_url,
                                                             auth_token,
                                                             sender_id, text)
            for response in bot_responses:
                print_bot_output(response)

        num_messages += 1

    if on_finish:
        on_finish()


class CmdlineInput(RestInput):

    @classmethod
    def name(cls):
        return "cmdline"

    def url_prefix(self):
        return RestInput.name()
