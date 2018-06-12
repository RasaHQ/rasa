from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import re

from rasa_core.channels import UserMessage, console
from rasa_core.constants import DEFAULT_SERVER_URL


def replay_messages(filename,
                    server_url=DEFAULT_SERVER_URL,
                    auth_token=None,
                    sender_id=UserMessage.DEFAULT_SENDER_ID,
                    max_message_limit=None,
                    message_line_pattern=".*",
                    output_channel=None):
    """Read messages from the command line and print bot responses."""

    auth_token = auth_token if auth_token else ""

    message_filter = re.compile(message_line_pattern)

    with io.open(filename, 'r') as f:
        for num_messages, line in enumerate(f):
            m = message_filter.match(line)
            if m is not None:
                message = m.group(1 if m.lastindex else 0)
                bot_responses = console.send_message_receive_stream(
                        server_url,
                        auth_token,
                        sender_id,
                        message)
                for response in bot_responses:
                    output_channel.send_response(response)

            if console.is_msg_limit_reached(num_messages, max_message_limit):
                break
