import copy

import pytest
import pytz
import time
from datetime import datetime
from dateutil import parser
from typing import Type, Optional, Text, List, Any, Dict

import rasa.shared.utils.common
import rasa.shared.core.events
from rasa.core.test import (
    WronglyClassifiedUserUtterance,
    WarningPredictedAction,
    WronglyPredictedAction,
    EndToEndUserUtterance,
    EvaluationStore,
)
from rasa.shared.exceptions import UnsupportedFeatureException
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
)
from rasa.shared.core.events import (
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
    EntitiesAdded,
    DefinePrevUserUtteredFeaturization,
    ActiveLoop,
    LegacyForm,
    LoopInterrupted,
    ActionExecutionRejected,
    LegacyFormValidation,
    format_message,
)
from rasa.shared.nlu.constants import INTENT_NAME_KEY, METADATA_MODEL_ID
from tests.core.policies.test_rule_policy import GREET_INTENT_NAME, UTTER_GREET_ACTION


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


def test_json_parse_action_executed_with_no_hide_rule():
    evt = {
        "event": "action",
        "name": "action_listen",
        "policy": None,
        "confidence": None,
        "timestamp": None,
    }
    deserialised: ActionExecuted = Event.from_parameters(evt)
    expected = ActionExecuted("action_listen")
    assert deserialised == expected
    assert deserialised.hide_rule_turn == expected.hide_rule_turn


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


@pytest.mark.parametrize("event_class", rasa.shared.utils.common.all_subclasses(Event))
def test_event_metadata_dict(event_class: Type[Event]):
    metadata = {"foo": "bar", "quux": 42}
    parameters = {
        "metadata": metadata,
        "event": event_class.type_name,
        "parse_data": {},
        "date_time": "2019-11-20T16:09:16Z",
    }
    # `ActionExecuted` class and its subclasses require either that action_name
    # is not None if it is not an end-to-end predicted action
    if event_class.type_name in ["action", "wrong_action", "warning_predicted"]:
        parameters["name"] = "test"

    # Create the event from a `dict` that will be accepted by the
    # `_from_parameters` method of any `Event` subclass (the values themselves
    # are not important).
    event = Event.from_parameters(parameters)
    assert event.as_dict()["metadata"] == metadata


@pytest.mark.parametrize("event_class", rasa.shared.utils.common.all_subclasses(Event))
def test_event_default_metadata(event_class: Type[Event]):
    parameters = {
        "event": event_class.type_name,
        "parse_data": {},
        "date_time": "2019-11-20T16:09:16Z",
    }
    # `ActionExecuted` class and its subclasses require either that action_name
    # is not None if it is not an end-to-end predicted action
    if event_class.type_name in ["action", "wrong_action", "warning_predicted"]:
        parameters["name"] = "test"

    # Create an event without metadata. When converting the `Event` to a
    # `dict`, it should not include a `metadata` property - unless it's a
    # `UserUttered` or a `BotUttered` event (or subclasses of them), in which
    # case the metadata should be included with a default value of {}.
    event = Event.from_parameters(parameters)

    if isinstance(event, BotUttered) or isinstance(event, UserUttered):
        assert event.as_dict()["metadata"] == {}
    else:
        assert "metadata" not in event.as_dict()


@pytest.mark.parametrize(
    "event, intent_name",
    [
        (UserUttered("text", {}), None),
        (UserUttered("dasd", {"name": None}), None),
        (UserUttered("adasd", {"name": "intent"}), "intent"),
    ],
)
def test_user_uttered_intent_name(event: UserUttered, intent_name: Optional[Text]):
    assert event.intent_name == intent_name


def test_md_format_message():
    assert format_message("Hello there!", intent="greet", entities=[]) == "Hello there!"


def test_md_format_message_empty():
    assert format_message("", intent=None, entities=[]) == ""


def test_md_format_message_using_short_entity_syntax():
    formatted = format_message(
        "I am from Berlin.",
        intent="location",
        entities=[{"start": 10, "end": 16, "entity": "city", "value": "Berlin"}],
    )
    assert formatted == """I am from [Berlin](city)."""


def test_md_format_message_using_short_entity_syntax_no_start_end():
    formatted = format_message(
        "hello", intent="location", entities=[{"entity": "city", "value": "Berlin"}]
    )
    assert formatted == "hello"


def test_md_format_message_no_text():
    formatted = format_message("", intent="location", entities=[])
    assert formatted == ""


def test_md_format_message_using_short_entity_syntax_no_start_end_or_text():
    formatted = format_message(
        "", intent="location", entities=[{"entity": "city", "value": "Berlin"}]
    )
    assert formatted == ""


def test_md_format_message_using_long_entity_syntax():
    formatted = format_message(
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
        formatted == """I am from [Berlin](city) in [Germany]"""
        """{"entity": "country", "role": "destination"}."""
    )


def test_md_format_message_using_long_entity_syntax_no_start_end():
    formatted = format_message(
        "I am from Berlin.",
        intent="location",
        entities=[
            {"start": 10, "end": 16, "entity": "city", "value": "Berlin"},
            {"entity": "country", "value": "Germany", "role": "destination"},
        ],
    )
    assert formatted == "I am from [Berlin](city)."


@pytest.mark.parametrize(
    (
        "events_to_split,event_type_to_split_on,additional_splitting_conditions,"
        "n_resulting_lists,include_splitting_event"
    ),
    [
        # splitting on an action that is not contained in the list results in
        # the same list of events
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
            ],
            BotUttered,
            {},
            1,
            False,
        ),
        # splitting on UserUttered in general results in two lists
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
            ],
            UserUttered,
            {},
            2,
            True,
        ),
        # we have the same number of lists when not including the event we're
        # splitting on
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
            ],
            UserUttered,
            {},
            2,
            False,
        ),
        # splitting on a specific UserUttered event does not result in a split
        # if it does not match
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
            ],
            UserUttered,
            {"intent": "wrong-intent"},
            1,
            True,
        ),
        # splitting on the right UserUttered event does result in the right split
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
            ],
            UserUttered,
            {"intent": {"name": GREET_INTENT_NAME}},
            2,
            True,
        ),
        # splitting on a specific ActionExecuted works as well
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
            ],
            ActionExecuted,
            {"action_name": UTTER_GREET_ACTION},
            2,
            True,
        ),
    ],
)
def test_split_events(
    events_to_split: List[Event],
    event_type_to_split_on: Type[Event],
    additional_splitting_conditions: Dict[Text, Any],
    n_resulting_lists: int,
    include_splitting_event: bool,
):
    split_events = rasa.shared.core.events.split_events(
        events_to_split,
        event_type_to_split_on,
        additional_splitting_conditions,
        include_splitting_event=include_splitting_event,
    )
    assert len(split_events) == n_resulting_lists

    # if we're not including the splitting event, that event type should not be
    # contained in the resulting events
    if not include_splitting_event:
        assert all(
            not any(isinstance(event, event_type_to_split_on) for event in events)
            for events in split_events
        )

    # make sure the event we're splitting on is the first one if a split happened
    if len(split_events) > 1 and include_splitting_event:
        assert all(
            isinstance(events[0], event_type_to_split_on) for events in split_events[1:]
        )


@pytest.mark.parametrize(
    "test_events,begin_with_session_start",
    [
        # a typical session start
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
            True,
        ),
        # also a session start, but with timestamps
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME, timestamp=1),
                SessionStarted(timestamp=2),
                ActionExecuted(ACTION_LISTEN_NAME, timestamp=3),
            ],
            True,
        ),
        # also a session start, but with metadata
        (
            [
                ActionExecuted(
                    ACTION_SESSION_START_NAME,
                    timestamp=1,
                    metadata={METADATA_MODEL_ID: "123"},
                ),
                SessionStarted(timestamp=2, metadata={METADATA_MODEL_ID: "123"}),
                ActionExecuted(
                    ACTION_LISTEN_NAME, timestamp=3, metadata={METADATA_MODEL_ID: "123"}
                ),
            ],
            True,
        ),
        # providing a single `action_listen` is not a session start
        ([ActionExecuted(ACTION_LISTEN_NAME, timestamp=3)], False),
        # providing a single `action_session_start` is not a session start
        ([ActionExecuted(ACTION_SESSION_START_NAME)], False),
        # providing no events is not a session start
        ([], False),
    ],
)
def test_events_begin_with_session_start(
    test_events: List[Event], begin_with_session_start: bool
):
    assert (
        rasa.shared.core.events.do_events_begin_with_session_start(test_events)
        == begin_with_session_start
    )


@pytest.mark.parametrize(
    "end_to_end_event",
    [
        ActionExecuted(action_text="I insist on using Markdown"),
        UserUttered(text="Markdown is much more readable"),
        UserUttered(
            text="but YAML ❤️",
            intent={INTENT_NAME_KEY: "use_yaml"},
            use_text_for_featurization=True,
        ),
    ],
)
def test_print_end_to_end_events(end_to_end_event: Event):
    with pytest.raises(UnsupportedFeatureException):
        end_to_end_event.as_story_string()


@pytest.mark.parametrize(
    "events,comparison_result",
    [
        (
            [
                ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ],
            True,
        ),
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ],
            False,
        ),
        (
            [
                ActionExecuted(
                    ACTION_UNLIKELY_INTENT_NAME, metadata={"test": {"data1": 1}}
                ),
                ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
            ],
            False,
        ),
        (
            [
                ActionExecuted(
                    ACTION_UNLIKELY_INTENT_NAME, metadata={"test": {"data1": 1}}
                ),
                ActionExecuted(
                    ACTION_UNLIKELY_INTENT_NAME, metadata={"test": {"data1": 1}}
                ),
            ],
            True,
        ),
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME, metadata={"test": {"data1": 1}}),
                ActionExecuted(
                    ACTION_UNLIKELY_INTENT_NAME, metadata={"test": {"data1": 1}}
                ),
            ],
            False,
        ),
    ],
)
def test_event_executed_comparison(events: List[Event], comparison_result: bool):
    result = all(event == events[0] for event in events)
    assert result == comparison_result


tested_events = [
    EntitiesAdded(
        entities=[
            {
                "entity": "city",
                "value": "London",
                "role": "destination",
                "group": "test",
            },
            {"entity": "count", "value": 1},
        ],
        timestamp=None,
    ),
    DefinePrevUserUtteredFeaturization(
        use_text_for_featurization=False, timestamp=None, metadata=None
    ),
    ReminderCancelled(timestamp=1621590172.3872123),
    ReminderScheduled(timestamp=None, trigger_date_time=datetime.now(), intent="greet"),
    ActionExecutionRejected(action_name="my_action"),
    LegacyFormValidation(validate=True, timestamp=None),
    LoopInterrupted(timestamp=None, is_interrupted=False),
    ActiveLoop(name="loop"),
    LegacyForm(name="my_form"),
    AllSlotsReset(),
    SlotSet(key="my_slot", value={}),
    SlotSet(key="my slot", value=[]),
    SlotSet(key="test", value=1),
    SlotSet(key="test", value="text"),
    ConversationResumed(),
    ConversationPaused(),
    FollowupAction(name="test"),
    StoryExported(),
    Restarted(),
    ActionReverted(),
    UserUtteranceReverted(),
    BotUttered(text="Test bot utterance"),
    UserUttered(
        parse_data={
            "entities": [],
            "response_selector": {
                "all_retrieval_intents": [],
                "chitchat/ask_weather": {"response": {}, "ranking": []},
            },
        }
    ),
    UserUttered(
        text="hello",
        parse_data={
            "intent": {"name": "greet", "confidence": 0.9604260921478271},
            "entities": [
                {"entity": "city", "value": "London"},
                {"entity": "count", "value": 1},
            ],
            "text": "hi",
            "message_id": "3f4c04602a4947098c574b107d3ccc50",
            "metadata": {},
            "intent_ranking": [
                {"name": "greet", "confidence": 0.9604260921478271},
                {"name": "goodbye", "confidence": 0.01835782080888748},
                {"name": "deny", "confidence": 0.011255578137934208},
                {"name": "bot_challenge", "confidence": 0.004019865766167641},
                {"name": "affirm", "confidence": 0.002524246694520116},
                {"name": "mood_great", "confidence": 0.002214624546468258},
                {"name": "chitchat", "confidence": 0.0009614597074687481},
                {"name": "mood_unhappy", "confidence": 0.00024030178610701114},
            ],
            "response_selector": {
                "all_retrieval_intents": [],
                "default": {
                    "response": {
                        "id": 11,
                        "responses": [{"text": "chitchat/ask_name"}],
                        "response_templates": [{"text": "chitchat/ask_name"}],
                        "confidence": 0.9618658423423767,
                        "intent_response_key": "chitchat/ask_name",
                        "utter_action": "utter_chitchat/ask_name",
                        "template_name": "utter_chitchat/ask_name",
                    },
                    "ranking": [
                        {
                            "id": 11,
                            "confidence": 0.9618658423423767,
                            "intent_response_key": "chitchat/ask_name",
                        },
                        {
                            "id": 12,
                            "confidence": 0.03813415765762329,
                            "intent_response_key": "chitchat/ask_weather",
                        },
                    ],
                },
            },
        },
    ),
    SessionStarted(),
    ActionExecuted(action_name="action_listen"),
    AgentUttered(),
    EndToEndUserUtterance(),
    WronglyClassifiedUserUtterance(
        event=UserUttered(), eval_store=EvaluationStore(intent_targets=["test"])
    ),
    WronglyPredictedAction(
        action_name_prediction="test",
        action_name_target="demo",
        action_text_target="example",
    ),
    WarningPredictedAction(action_name="action_listen", action_name_prediction="test"),
]


@pytest.mark.parametrize("event", tested_events)
def test_event_fingerprint_consistency(event: Event):
    f1 = event.fingerprint()

    event2 = copy.deepcopy(event)
    f2 = event2.fingerprint()

    assert f1 == f2


@pytest.mark.parametrize("event_class", rasa.shared.utils.common.all_subclasses(Event))
def test_event_subclasses_are_tested(event_class: Type[Event]):
    subclasses = [event.__class__ for event in tested_events]
    assert event_class in subclasses


@pytest.mark.parametrize("event", tested_events)
def test_event_fingerprint_uniqueness(event: Event):
    f1 = event.fingerprint()
    event.type_name = "test"
    f2 = event.fingerprint()

    assert f1 != f2


def test_session_started_event_is_not_serialised():
    assert SessionStarted().as_story_string() is None


@pytest.mark.parametrize(
    "event",
    [
        {
            "event": "user",
            "text": "Hey",
            "parse_data": {
                "intent": {"name": "greet", "confidence": 0.9},
                "entities": [],
            },
            "metadata": {},
        },
        {
            "event": "action",
            "name": "action_listen",
            "policy": None,
            "confidence": None,
            "timestamp": None,
        },
    ],
)
def test_remove_parse_data(event: Dict[Text, Any]):
    reduced_event = rasa.shared.core.events.remove_parse_data(event)
    assert "parse_data" not in reduced_event
