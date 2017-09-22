from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels.console import ConsoleInputChannel


def test_console_input():
    import rasa_core.channels.console
    # Overwrites the input() function and when someone else tries to read
    # something from the command line this function gets called. But instead of
    # waiting input for the user, this simulates the input of
    # "2", therefore it looks like the user is always typing "2" if someone
    # requests a cmd input.

    rasa_core.channels.console.input = lambda _=None: "Test Input"

    recorded = []

    def on_message(message):
        recorded.append(message)

    channel = ConsoleInputChannel()
    channel._record_messages(on_message, max_message_limit=3)
    assert [r.text for r in recorded] == ["Test Input",
                                          "Test Input",
                                          "Test Input"]
