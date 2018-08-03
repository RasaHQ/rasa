from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels import UserMessage
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Button


def test_message_processor(default_processor):
    out = CollectingOutputChannel()
    default_processor.handle_message(UserMessage('/greet{"name":"Core"}', out))
    assert {'recipient_id': 'default',
            'text': 'hey there Core!'} == out.latest_output()


def test_logging_of_bot_utterances_on_tracker(default_processor,
                                              default_dispatcher_collecting,
                                              default_agent):
    sender_id = "test_logging_of_bot_utterances_on_tracker"
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)
    buttons = [
        Button(title="Btn1", payload="_btn1"),
        Button(title="Btn2", payload="_btn2")
    ]

    default_dispatcher_collecting.utter_template("utter_goodbye", tracker)
    default_dispatcher_collecting.utter_attachment("http://my-attachment")
    default_dispatcher_collecting.utter_message("my test message")
    default_dispatcher_collecting.utter_button_message("my message", buttons)

    assert len(default_dispatcher_collecting.latest_bot_messages) == 4

    default_processor.log_bot_utterances_on_tracker(
            tracker, default_dispatcher_collecting)
    assert not default_dispatcher_collecting.latest_bot_messages
