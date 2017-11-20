from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels import UserMessage
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Button, BotMessage
from rasa_core.processor import MessageProcessor


def test_message_processor(default_processor):
    out = CollectingOutputChannel()
    default_processor.handle_message(UserMessage("_greet[name=Core]", out))
    assert ("default", "hey there Core!") == out.latest_output()


def test_logging_of_bot_utterances_on_tracker(default_processor, default_dispatcher, default_agent):
    sender_id = "test_logging_of_bot_utterances_on_tracker"
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)
    buttons = [
        Button(title="Btn1", payload="_btn1"),
        Button(title="Btn2", payload="_btn2")
    ]

    default_dispatcher.utter_template("utter_goodbye")
    default_dispatcher.utter_attachment("http://my-attachment")
    default_dispatcher.utter_message("my test message")
    default_dispatcher.utter_button_message("my message", buttons)

    assert len(default_dispatcher.latest_bot_messages) == 4

    default_processor._log_bot_utterances_on_tracker(tracker, default_dispatcher)
    assert not default_dispatcher.latest_bot_messages