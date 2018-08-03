from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import re

from typing import Optional, Text

from rasa_core.channels import OutputChannel
from rasa_core.channels.channel import InputChannel, UserMessage


class FileInputChannel(InputChannel):
    """Input channel that reads the user messages from a specified file.

    If there are lines in the file that should not be used as user messages,
    or if the user messages are surrounded by other symbols that should not be
    part of the user message, you can use a regular expression to only match
    the user message. The `message_line_pattern` needs to be passed in as a
    string. The default is `.*` hence, considering the whole line as the user
    message. Either the whole message (if no capturing group is present) or the
    first capturing group will be used as the user message."""

    @classmethod
    def name(cls):
        return "file"

    def __init__(self,
                 filename,
                 output_channel=None,
                 message_line_pattern=".*",
                 max_messages=None):
        # type: (Text, OutputChannel, Text, Optional[int]) -> None
        from rasa_core.channels.console import ConsoleOutputChannel

        self.message_filter = re.compile(message_line_pattern)
        self.filename = filename
        self.max_messages = max_messages
        if output_channel:
            self.output_channel = output_channel
        else:
            self.output_channel = ConsoleOutputChannel()

    def _record_messages(self, on_message):
        with io.open(self.filename, 'r') as f:
            for i, line in enumerate(f):
                m = self.message_filter.match(line)
                if m is not None:
                    message = m.group(1 if m.lastindex else 0)
                    on_message(UserMessage(message, self.output_channel))
                if self.max_messages is not None and i >= self.max_messages:
                    break

    def start_async_listening(self, message_queue):
        self._record_messages(message_queue.enqueue)

    def start_sync_listening(self, message_handler):
        self._record_messages(message_handler)
