import datetime
import pytest
import uuid

from rasa_core.channels import CollectingOutputChannel
from rasa_core.channels import UserMessage
from rasa_core.dispatcher import Button, Dispatcher
from rasa_core.events import (
    ReminderScheduled, UserUttered, ActionExecuted,
    BotUttered, Restarted)
from rasa_nlu.training_data import Message


@pytest.fixture(scope="module")
def loop():
    from pytest_sanic.plugin import loop as sanic_loop
    return next(sanic_loop())


def test_message_processor(loop, default_processor):
    out = CollectingOutputChannel()
    loop.run_until_complete(default_processor.handle_message(
        UserMessage('/greet{"name":"Core"}', out)))
    assert {'recipient_id': 'default',
            'text': 'hey there Core!'} == out.latest_output()


def test_parsing(loop, default_processor):
    message = Message('/greet{"name": "boy"}')
    parsed = loop.run_until_complete(default_processor._parse_message(message))
    assert parsed["intent"]["name"] == 'greet'
    assert parsed["entities"][0]["entity"] == 'name'


def test_reminder_scheduled(loop, default_processor):
    out = CollectingOutputChannel()
    sender_id = uuid.uuid4().hex

    d = Dispatcher(sender_id, out, default_processor.nlg)
    r = ReminderScheduled("utter_greet", datetime.datetime.now())
    t = default_processor.tracker_store.get_or_create_tracker(sender_id)

    t.update(UserUttered("test"))
    t.update(ActionExecuted("action_reminder_reminder"))
    t.update(r)

    default_processor.tracker_store.save(t)
    loop.run_until_complete(default_processor.handle_reminder(r, d))

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert t.events[-4] == UserUttered(None)
    assert t.events[-3] == ActionExecuted("utter_greet")
    assert t.events[-2] == BotUttered("hey there None!", {'elements': None,
                                                          'buttons': None,
                                                          'attachment': None})
    assert t.events[-1] == ActionExecuted("action_listen")


def test_reminder_aborted(loop, default_processor):
    out = CollectingOutputChannel()
    sender_id = uuid.uuid4().hex

    d = Dispatcher(sender_id, out, default_processor.nlg)
    r = ReminderScheduled("utter_greet", datetime.datetime.now(),
                          kill_on_user_message=True)
    t = default_processor.tracker_store.get_or_create_tracker(sender_id)

    t.update(r)
    t.update(UserUttered("test"))  # cancels the reminder

    default_processor.tracker_store.save(t)
    loop.run_until_complete(default_processor.handle_reminder(r, d))

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 3  # nothing should have been executed


def test_reminder_restart(loop, default_processor):
    out = CollectingOutputChannel()
    sender_id = uuid.uuid4().hex

    d = Dispatcher(sender_id, out, default_processor.nlg)
    r = ReminderScheduled("utter_greet", datetime.datetime.now(),
                          kill_on_user_message=False)
    t = default_processor.tracker_store.get_or_create_tracker(sender_id)

    t.update(r)
    t.update(Restarted())  # cancels the reminder
    t.update(UserUttered("test"))

    default_processor.tracker_store.save(t)
    loop.run_until_complete(default_processor.handle_reminder(r, d))

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 4  # nothing should have been executed


def test_logging_of_bot_utterances_on_tracker(loop, default_processor,
                                              default_dispatcher_collecting,
                                              default_agent):
    sender_id = "test_logging_of_bot_utterances_on_tracker"
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)
    buttons = [
        Button(title="Btn1", payload="_btn1"),
        Button(title="Btn2", payload="_btn2")
    ]

    loop.run_until_complete(
        default_dispatcher_collecting.utter_template("utter_goodbye", tracker))
    loop.run_until_complete(
        default_dispatcher_collecting.utter_attachment("http://my-attachment"))
    loop.run_until_complete(
        default_dispatcher_collecting.utter_message("my test message"))
    loop.run_until_complete(
        default_dispatcher_collecting.utter_button_message("my message",
                                                           buttons))

    assert len(default_dispatcher_collecting.latest_bot_messages) == 4

    default_processor.log_bot_utterances_on_tracker(
        tracker, default_dispatcher_collecting)
    assert not default_dispatcher_collecting.latest_bot_messages
