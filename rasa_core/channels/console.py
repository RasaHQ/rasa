from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
from builtins import input
from typing import Text

from rasa_core.channels.channel import UserMessage
from rasa_core.channels.channel import InputChannel, OutputChannel
from rasa_core import utils


class ConsoleOutputChannel(OutputChannel):
    """Simple bot that outputs the bots messages to the command line."""

    default_output_color = utils.bcolors.OKBLUE

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        utils.print_color(message, self.default_output_color)


class ConsoleInputChannel(InputChannel):
    """Input channel that reads the user messages from the command line."""

    def _record_messages(self, on_message, max_message_limit=None):
        utils.print_color("Bot loaded. Type a message and press enter : ",
                          utils.bcolors.OKGREEN)
        num_messages = 0
        while max_message_limit is None or num_messages < max_message_limit:
            text = input().strip()
            if six.PY2:
                # in python 2 input doesn't return unicode values
                text = text.decode("utf-8")
            if text == '_stop':
                import os
                # sys.exit(1)
                os._exit(1)
            on_message(UserMessage(text, ConsoleOutputChannel()))
            num_messages += 1

    def start_async_listening(self, message_queue):
        self._record_messages(message_queue.enqueue)

    def start_sync_listening(self, message_handler):
        self._record_messages(message_handler)
