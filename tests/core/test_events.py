import copy

import pytest
import pytz
import time
from datetime import datetime
from dateutil import parser
from typing import Type

from rasa.core import utils
from rasa.core.events import (
    Event,
    UserUttered,
    SlotSet,
    Restarted,
    ActionExecuted,
    AllSlotsReset,
    ReminderScheduled,
    ReminderCancelled,
    ConversationResumed,
    ConversationPaused,
    StoryExported,
    ActionReverted,
    BotUttered,
    FollowupAction,
    UserUtteranceReverted,
    AgentUttered,
    SessionStarted,
    md_format_message,
)


@pytest.mark.parametrize(
    "one_event,another_event",
    [
        (
            UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),
            UserUttered("/goodbye", {"name": "goodbye", "confidence": 1.0}, []),
        ),
        (SlotSet("my_slot", "value"), SlotSet("my__other_slot", "value")),
        (Restarted(), None),
        (AllSlotsReset(), None),
        (ConversationPaused(), None),
        (ConversationResumed(), None),
        (StoryExported(), None),
        (ActionReverted(), None),
        (UserUtteranceReverted(), None),
        (SessionStarted(), None),
        (ActionExecuted("my_action"), ActionExecuted("my_other_action")),
        (FollowupAction("my_action"), FollowupAction("my_other_action")),
        (
            BotUttered("my_text", {"my_data": 1}),
            BotUttered("my_other_test", {"my_other_data": 1}),
        ),
        (
            AgentUttered("my_text", "my_data"),
            AgentUttered("my_other_test", "my_other_data"),
        ),
        (
            ReminderScheduled("my_intent", datetime.now()),
            ReminderScheduled("my_other_intent", datetime.now()),
        ),
    ],
)
def test_event_has_proper_implementation(one_event, another_event):
    # equals tests
    assert (
        one_event != another_event
    ), "Same events with different values need to be different"
    assert one_event == copy.deepcopy(one_event), "Event copies need to be the same"
    assert one_event != 42, "Events aren't equal to 42!"

    # hash test
    assert hash(one_event) == hash(
        copy.deepcopy(one_event)
    ), "Same events should have the same hash"
    assert hash(one_event) != hash(
        another_event
    ), "Different events should have different hashes"

    # str test
    assert "object at 0x" not in str(one_event), "Event has a proper str method"


@pytest.mark.parametrize(
    "one_event",
    [
        UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),
        UserUttered(metadata={"type": "text"}),
        UserUttered(metadata=None),
        UserUttered(text="hi", message_id="1", metadata={"type": "text"}),
        SlotSet("name", "rasa"),
        Restarted(),
        AllSlotsReset(),
        ConversationPaused(),
        ConversationResumed(),
        StoryExported(),
        ActionReverted(),
        UserUtteranceReverted(),
        SessionStarted(),
        ActionExecuted("my_action"),
        ActionExecuted("my_action", "policy_1_TEDPolicy", 0.8),
        FollowupAction("my_action"),
        BotUttered("my_text", {"my_data": 1}),
        AgentUttered("my_text", "my_data"),
        ReminderScheduled("my_intent", datetime.now()),
        ReminderScheduled("my_intent", datetime.now(pytz.timezone("US/Central"))),
    ],
)
def test_dict_serialisation(one_event):
    evt_dict = one_event.as_dict()
    recovered_event = Event.from_parameters(evt_dict)
    assert hash(one_event) == hash(recovered_event)


def test_json_parse_setslot():
    evt = {"event": "slot", "name": "departure_airport", "value": "BER"}
    assert Event.from_parameters(evt) == SlotSet("departure_airport", "BER")


def test_json_parse_restarted():
    evt = {"event": "restart"}
    assert Event.from_parameters(evt) == Restarted()


def test_json_parse_session_started():
    evt = {"event": "session_started"}
    assert Event.from_parameters(evt) == SessionStarted()


def test_json_parse_reset():
    evt = {"event": "reset_slots"}
    assert Event.from_parameters(evt) == AllSlotsReset()


def test_json_parse_user():
    evt = {
        "event": "user",
        "text": "Hey",
        "parse_data": {"intent": {"name": "greet", "confidence": 0.9}, "entities": []},
        "metadata": {},
    }
    assert Event.from_parameters(evt) == UserUttered(
        "Hey",
        intent={"name": "greet", "confidence": 0.9},
        entities=[],
        parse_data={"intent": {"name": "greet", "confidence": 0.9}, "entities": []},
        metadata={},
    )


def test_json_parse_bot():
    evt = {"event": "bot", "text": "Hey there!", "data": {}}
    assert Event.from_parameters(evt) == BotUttered("Hey there!", {})


def test_json_parse_rewind():
    evt = {"event": "rewind"}
    assert Event.from_parameters(evt) == UserUtteranceReverted()


def test_json_parse_reminder():
    evt = {
        "event": "reminder",
        "intent": "my_intent",
        "entities": {"entity1": "value1", "entity2": "value2"},
        "date_time": "2018-09-03T11:41:10.128172",
        "name": "my_reminder",
        "kill_on_user_msg": True,
    }
    assert Event.from_parameters(evt) == ReminderScheduled(
        "my_intent",
        parser.parse("2018-09-03T11:41:10.128172"),
        name="my_reminder",
        kill_on_user_message=True,
    )


def test_json_parse_reminder_cancelled():
    evt = {
        "event": "cancel_reminder",
        "name": "my_reminder",
        "intent": "my_intent",
        "entities": [
            {"entity": "entity1", "value": "value1"},
            {"entity": "entity2", "value": "value2"},
        ],
        "date_time": "2018-09-03T11:41:10.128172",
    }
    assert Event.from_parameters(evt) == ReminderCancelled(
        name="my_reminder",
        intent="my_intent",
        entities=[
            {"entity": "entity1", "value": "value1"},
            {"entity": "entity2", "value": "value2"},
        ],
        timestamp=parser.parse("2018-09-03T11:41:10.128172"),
    )


def test_json_parse_undo():
    evt = {"event": "undo"}
    assert Event.from_parameters(evt) == ActionReverted()


def test_json_parse_export():
    evt = {"event": "export"}
    assert Event.from_parameters(evt) == StoryExported()


def test_json_parse_followup():
    evt = {"event": "followup", "name": "my_action"}
    assert Event.from_parameters(evt) == FollowupAction("my_action")


def test_json_parse_pause():
    evt = {"event": "pause"}
    assert Event.from_parameters(evt) == ConversationPaused()


def test_json_parse_resume():
    evt = {"event": "resume"}
    assert Event.from_parameters(evt) == ConversationResumed()


def test_json_parse_action():
    evt = {"event": "action", "name": "my_action"}
    assert Event.from_parameters(evt) == ActionExecuted("my_action")


def test_json_parse_agent():
    evt = {"event": "agent", "text": "Hey, how are you?"}
    assert Event.from_parameters(evt) == AgentUttered("Hey, how are you?")


@pytest.mark.parametrize(
    "event_class",
    [
        UserUttered,
        BotUttered,
        ActionReverted,
        Restarted,
        AllSlotsReset,
        ConversationResumed,
        ConversationPaused,
        StoryExported,
        UserUtteranceReverted,
        AgentUttered,
    ],
)
def test_correct_timestamp_setting_for_events_with_no_required_params(event_class):
    event = event_class()
    time.sleep(0.01)
    event2 = event_class()

    assert event.timestamp < event2.timestamp


@pytest.mark.parametrize("event_class", [SlotSet, ActionExecuted, FollowupAction])
def test_correct_timestamp_setting(event_class):
    event = event_class("test")
    time.sleep(0.01)
    event2 = event_class("test")

    assert event.timestamp < event2.timestamp


@pytest.mark.parametrize("event_class", utils.all_subclasses(Event))
def test_event_metadata_dict(event_class: Type[Event]):
    metadata = {"foo": "bar", "quux": 42}

    # Create the event from a `dict` that will be accepted by the
    # `_from_parameters` method of any `Event` subclass (the values themselves
    # are not important).
    event = Event.from_parameters(
        {
            "metadata": metadata,
            "event": event_class.type_name,
            "parse_data": {},
            "date_time": "2019-11-20T16:09:16Z",
        }
    )
    assert event.as_dict()["metadata"] == metadata


@pytest.mark.parametrize("event_class", utils.all_subclasses(Event))
def test_event_default_metadata(event_class: Type[Event]):
    # Create an event without metadata. When converting the `Event` to a
    # `dict`, it should not include a `metadata` property - unless it's a
    # `UserUttered` or a `BotUttered` event (or subclasses of them), in which
    # case the metadata should be included with a default value of {}.
    event = Event.from_parameters(
        {
            "event": event_class.type_name,
            "parse_data": {},
            "date_time": "2019-11-20T16:09:16Z",
        }
    )

    if isinstance(event, BotUttered) or isinstance(event, UserUttered):
        assert event.as_dict()["metadata"] == {}
    else:
        assert "metadata" not in event.as_dict()


def test_md_format_message():
    assert (
        md_format_message("Hello there!", intent="greet", entities=[]) == "Hello there!"
    )


def test_md_format_message_empty():
    assert md_format_message("", intent=None, entities=[]) == ""


def test_md_format_message_using_short_entity_syntax():
    formatted = md_format_message(
        "I am from Berlin.",
        intent="location",
        entities=[{"start": 10, "end": 16, "entity": "city", "value": "Berlin"}],
    )
    assert formatted == """I am from [Berlin](city)."""


def test_md_format_message_using_long_entity_syntax():
    formatted = md_format_message(
        "I am from Berlin in Germany.",
        intent="location",
        entities=[
            {"start": 10, "end": 16, "entity": "city", "value": "Berlin"},
            {
                "start": 20,
                "end": 27,
                "entity": "country",
                "value": "Germany",
                "role": "destination",
            },
        ],
    )
    assert (
        formatted
        == """I am from [Berlin](city) in [Germany]{"entity": "country", "role": "destination"}."""
    )
