import datetime
import json
import logging
import os
import textwrap
import time
from pathlib import Path
import tempfile
from typing import List, Text, Dict, Any, Type

import fakeredis
import freezegun
import pytest

from rasa.core.actions.action import ActionExtractSlots
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.training import load_data
import rasa.shared.utils.io
import rasa.utils.io
from rasa.core import training
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
    LOOP_NAME,
    REQUESTED_SLOT,
    LOOP_INTERRUPTED,
)
from rasa.shared.constants import (
    ASSISTANT_ID_KEY,
    DEFAULT_SENDER_ID,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.core.agent import Agent
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    AgentUttered,
    AllSlotsReset,
    ConversationPaused,
    ConversationResumed,
    FollowupAction,
    ReminderScheduled,
    SlotSet,
    StoryExported,
    UserUttered,
    ActionExecuted,
    Restarted,
    ActionReverted,
    UserUtteranceReverted,
    SessionStarted,
    Event,
    ActiveLoop,
    ActionExecutionRejected,
    BotUttered,
    LegacyForm,
    LegacyFormValidation,
    LoopInterrupted,
    DefinePrevUserUtteredFeaturization,
    EntitiesAdded,
)
from rasa.shared.core.slots import (
    FloatSlot,
    BooleanSlot,
    ListSlot,
    TextSlot,
    Slot,
    AnySlot,
)
from rasa.core.tracker_store import (
    InMemoryTrackerStore,
    RedisTrackerStore,
    SQLTrackerStore,
)
from rasa.core.tracker_store import TrackerStore
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from tests.core.conftest import MockedMongoTrackerStore
from tests.dialogues import (
    TEST_DIALOGUES,
    TEST_MOODBOT_DIALOGUE,
    TEST_DOMAINS_FOR_DIALOGUES,
)
from tests.core.utilities import tracker_from_dialogue, user_uttered, get_tracker

from rasa.shared.nlu.constants import (
    ACTION_NAME,
    METADATA_MODEL_ID,
    PREDICTED_CONFIDENCE_KEY,
)

test_domain = Domain.load("data/test_moodbot/domain.yml")


class MockRedisTrackerStore(RedisTrackerStore):
    def __init__(self, _domain: Domain) -> None:
        super().__init__(_domain)

        # Patch the Redis connection in RedisTrackerStore using fakeredis
        self.red = fakeredis.FakeStrictRedis()

        # added in redis==3.3.0, but not yet in fakeredis
        self.red.connection_pool.connection_class.health_check_interval = 0


def stores_to_be_tested():
    temp = tempfile.mkdtemp()
    return [
        MockRedisTrackerStore(test_domain),
        InMemoryTrackerStore(test_domain),
        SQLTrackerStore(test_domain, db=os.path.join(temp, "rasa.db")),
        MockedMongoTrackerStore(test_domain),
    ]


def stores_to_be_tested_ids():
    return ["redis-tracker", "in-memory-tracker", "SQL-tracker", "mongo-tracker"]


def test_tracker_duplicate(moodbot_domain: Domain):
    tracker = tracker_from_dialogue(TEST_MOODBOT_DIALOGUE, moodbot_domain)
    num_actions = len(
        [
            event
            for event in TEST_MOODBOT_DIALOGUE.events
            if isinstance(event, ActionExecuted)
        ]
    )

    # There is always one duplicated tracker more than we have actions,
    # as the tracker also gets duplicated for the
    # action that would be next (but isn't part of the operations)
    assert len(list(tracker.generate_all_prior_trackers())) == num_actions + 1


@pytest.mark.parametrize("store", stores_to_be_tested(), ids=stores_to_be_tested_ids())
async def test_tracker_store_storage_and_retrieval(store: TrackerStore):
    tracker = await store.get_or_create_tracker("some-id")
    # the retrieved tracker should be empty
    assert tracker.sender_id == "some-id"

    # Action listen should be in there
    assert list(tracker.events) == [ActionExecuted(ACTION_LISTEN_NAME)]

    # lets log a test message
    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(UserUttered("/greet", intent, []))
    assert tracker.latest_message.intent.get("name") == "greet"
    await store.save(tracker)

    # retrieving the same tracker should result in the same tracker
    retrieved_tracker = await store.get_or_create_tracker("some-id")
    assert retrieved_tracker.sender_id == "some-id"
    assert len(retrieved_tracker.events) == 2
    assert retrieved_tracker.latest_message.intent.get("name") == "greet"

    # getting another tracker should result in an empty tracker again
    other_tracker = await store.get_or_create_tracker("some-other-id")
    assert other_tracker.sender_id == "some-other-id"
    assert len(other_tracker.events) == 1


@pytest.mark.parametrize("store", stores_to_be_tested(), ids=stores_to_be_tested_ids())
@pytest.mark.parametrize("pair", zip(TEST_DIALOGUES, TEST_DOMAINS_FOR_DIALOGUES))
async def test_tracker_store(store, pair):
    dialogue, domainpath = pair
    domain = Domain.load(domainpath)
    tracker = tracker_from_dialogue(dialogue, domain)
    await store.save(tracker)
    restored = await store.retrieve(tracker.sender_id)
    assert restored == tracker


def test_tracker_write_to_story(tmp_path: Path, moodbot_domain: Domain):
    tracker = tracker_from_dialogue(TEST_MOODBOT_DIALOGUE, moodbot_domain)
    p = tmp_path / "export.yml"
    tracker.export_stories_to_file(str(p))
    trackers = training.load_data(
        str(p),
        moodbot_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    assert len(trackers) == 1
    recovered = trackers[0]
    assert len(recovered.events) == 11
    assert recovered.events[4].type_name == "user"
    assert recovered.events[4].intent == {
        "confidence": 1.0,
        "name": "mood_unhappy",
        "full_retrieval_intent_name": None,
    }


async def test_tracker_state_regression_without_bot_utterance(default_agent: Agent):
    sender_id = "test_tracker_state_regression_without_bot_utterance"
    for i in range(0, 2):
        await default_agent.handle_text("/greet", sender_id=sender_id)
    tracker = await default_agent.tracker_store.get_or_create_tracker(sender_id)

    # Ensures that the tracker has changed between the utterances
    # (and wasn't reset in between them)
    expected = (
        "action_session_start;action_listen;greet;utter_greet;action_listen;"
        "greet;utter_greet;action_listen"
    )
    assert (
        ";".join([e.as_story_string() for e in tracker.events if e.as_story_string()])
        == expected
    )


async def test_tracker_state_regression_with_bot_utterance(default_agent: Agent):
    sender_id = "test_tracker_state_regression_with_bot_utterance"
    for i in range(0, 2):
        await default_agent.handle_text("/greet", sender_id=sender_id)
    tracker = await default_agent.tracker_store.get_or_create_tracker(sender_id)

    expected = [
        "action_session_start",
        None,
        "action_listen",
        "greet",
        None,  # DefinePrevUserUtteredFeaturization
        "utter_greet",
        None,
        "action_listen",
        "greet",
        None,  # DefinePrevUserUtteredFeaturization
        "utter_greet",
        None,
        "action_listen",
    ]

    assert [e.as_story_string() for e in tracker.events] == expected


async def test_bot_utterance_comes_after_action_event(default_agent: Agent):
    sender_id = "test_bot_utterance_comes_after_action_event"

    await default_agent.handle_text("/greet", sender_id=sender_id)

    tracker = await default_agent.tracker_store.get_or_create_tracker(sender_id)

    # important is, that the 'bot' comes after the second 'action' and not
    # before
    expected = [
        "action",
        "session_started",
        "action",
        "user",
        "user_featurization",
        "action",
        "bot",
        "action",
    ]

    assert [e.type_name for e in tracker.events] == expected


@pytest.mark.parametrize(
    "entities, expected_values",
    [
        ([{"value": "greet", "entity": "entity_name"}], ["greet"]),
        (
            [
                {"value": "greet", "entity": "entity_name"},
                {"value": "bye", "entity": "other"},
            ],
            ["greet"],
        ),
        (
            [
                {"value": "greet", "entity": "entity_name"},
                {"value": "bye", "entity": "entity_name"},
            ],
            ["greet", "bye"],
        ),
        (
            [
                {"value": "greet", "entity": "entity_name", "role": "role"},
                {"value": "bye", "entity": "entity_name"},
            ],
            ["greet"],
        ),
        (
            [
                {"value": "greet", "entity": "entity_name", "group": "group"},
                {"value": "bye", "entity": "entity_name"},
            ],
            ["greet"],
        ),
        (
            [
                {"value": "greet", "entity": "entity_name"},
                {"value": "bye", "entity": "entity_name", "group": "group"},
            ],
            # bye has group set, so it should not be extracted
            ["greet"],
        ),
    ],
)
def test_get_latest_entity_values(
    entities: List[Dict[Text, Any]], expected_values: List[Text], domain: Domain
):
    entity_type = entities[0].get("entity")
    entity_role = entities[0].get("role")
    entity_group = entities[0].get("group")

    tracker = DialogueStateTracker("default", domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0
    assert list(tracker.get_latest_entity_values(entity_type)) == []

    intent = {"name": "greet", PREDICTED_CONFIDENCE_KEY: 1.0}
    tracker.update(UserUttered("/greet", intent, entities))

    assert (
        list(
            tracker.get_latest_entity_values(
                entity_type, entity_role=entity_role, entity_group=entity_group
            )
        )
        == expected_values
    )
    assert list(tracker.get_latest_entity_values("unknown")) == []


async def test_tracker_update_slots_with_entity(domain: Domain):
    tracker = DialogueStateTracker("default", domain.slots)

    test_entity = domain.entities[0]
    expected_slot_value = "test user"

    intent = {"name": "greet", PREDICTED_CONFIDENCE_KEY: 1.0}
    tracker.update(
        UserUttered(
            "/greet",
            intent,
            [
                {
                    "start": 1,
                    "end": 5,
                    "value": expected_slot_value,
                    "entity": test_entity,
                    "extractor": "manual",
                }
            ],
        ),
        domain,
    )
    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    tracker.update_with_events(events, domain)

    assert tracker.get_slot(test_entity) == expected_slot_value


def test_restart_event(domain: Domain):
    tracker = DialogueStateTracker("default", domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent = {"name": "greet", PREDICTED_CONFIDENCE_KEY: 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("/greet", intent, []))
    tracker.update(ActionExecuted("my_action"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    assert len(tracker.events) == 4
    assert tracker.latest_message.text == "/greet"
    assert len(list(tracker.generate_all_prior_trackers())) == 4

    tracker.update(Restarted())

    assert len(tracker.events) == 5
    assert tracker.followup_action == ACTION_SESSION_START_NAME

    tracker.update(SessionStarted())

    assert tracker.followup_action == ACTION_LISTEN_NAME
    assert tracker.latest_message.text is None
    assert len(list(tracker.generate_all_prior_trackers())) == 1

    dialogue = tracker.as_dialogue()

    recovered = DialogueStateTracker("default", domain.slots)
    recovered.recreate_from_dialogue(dialogue)

    assert recovered.current_state() == tracker.current_state()
    assert len(recovered.events) == 6
    assert recovered.latest_message.text is None
    assert len(list(recovered.generate_all_prior_trackers())) == 1


def test_session_start(domain: Domain):
    tracker = DialogueStateTracker("default", domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    # add a SessionStarted event
    tracker.update(SessionStarted())

    # tracker has one event
    assert len(tracker.events) == 1


def test_revert_action_event(domain: Domain):
    tracker = DialogueStateTracker("default", domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent = {"name": "greet", PREDICTED_CONFIDENCE_KEY: 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("/greet", intent, []))
    tracker.update(ActionExecuted("my_action"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    # Expecting count of 4:
    #   +3 executed actions
    #   +1 final state
    assert tracker.latest_action.get(ACTION_NAME) == ACTION_LISTEN_NAME
    assert len(list(tracker.generate_all_prior_trackers())) == 4

    tracker.update(ActionReverted())

    # Expecting count of 3:
    #   +3 executed actions
    #   +1 final state
    #   -1 reverted action
    assert tracker.latest_action.get(ACTION_NAME) == "my_action"
    assert len(list(tracker.generate_all_prior_trackers())) == 3

    dialogue = tracker.as_dialogue()

    recovered = DialogueStateTracker("default", domain.slots)
    recovered.recreate_from_dialogue(dialogue)

    assert recovered.current_state() == tracker.current_state()
    assert tracker.latest_action.get(ACTION_NAME) == "my_action"
    assert len(list(tracker.generate_all_prior_trackers())) == 3


def test_revert_user_utterance_event(domain: Domain):
    tracker = DialogueStateTracker("default", domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent1 = {"name": "greet", PREDICTED_CONFIDENCE_KEY: 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("/greet", intent1, []))
    tracker.update(ActionExecuted("my_action_1"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    intent2 = {"name": "goodbye", PREDICTED_CONFIDENCE_KEY: 1.0}
    tracker.update(UserUttered("/goodbye", intent2, []))
    tracker.update(ActionExecuted("my_action_2"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    # Expecting count of 6:
    #   +5 executed actions
    #   +1 final state
    assert tracker.latest_action.get(ACTION_NAME) == ACTION_LISTEN_NAME
    assert len(list(tracker.generate_all_prior_trackers())) == 6

    tracker.update(UserUtteranceReverted())

    # Expecting count of 3:
    #   +5 executed actions
    #   +1 final state
    #   -2 rewound actions associated with the /goodbye
    #   -1 rewound action from the listen right before /goodbye
    assert tracker.latest_action.get(ACTION_NAME) == "my_action_1"
    assert len(list(tracker.generate_all_prior_trackers())) == 3

    dialogue = tracker.as_dialogue()

    recovered = DialogueStateTracker("default", domain.slots)
    recovered.recreate_from_dialogue(dialogue)

    assert recovered.current_state() == tracker.current_state()
    assert tracker.latest_action.get(ACTION_NAME) == "my_action_1"
    assert len(list(tracker.generate_all_prior_trackers())) == 3


def test_traveling_back_in_time(domain: Domain):
    tracker = DialogueStateTracker("default", domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent = {"name": "greet", PREDICTED_CONFIDENCE_KEY: 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("/greet", intent, []))

    time.sleep(1)
    time_for_timemachine = time.time()
    time.sleep(1)

    tracker.update(ActionExecuted("my_action"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    # Expecting count of 4:
    #   +3 executed actions
    #   +1 final state
    assert tracker.latest_action.get(ACTION_NAME) == ACTION_LISTEN_NAME
    assert len(tracker.events) == 4
    assert len(list(tracker.generate_all_prior_trackers())) == 4

    tracker = tracker.travel_back_in_time(time_for_timemachine)

    # Expecting count of 2:
    #   +1 executed actions
    #   +1 final state
    assert tracker.latest_action.get(ACTION_NAME) == ACTION_LISTEN_NAME
    assert len(tracker.events) == 2
    assert len(list(tracker.generate_all_prior_trackers())) == 2


def test_tracker_init_copy(domain: Domain):
    sender_id = "some-id"
    tracker = DialogueStateTracker(sender_id, domain.slots)
    tracker_copy = tracker.init_copy()
    assert tracker.sender_id == tracker_copy.sender_id


def _load_tracker_from_json(tracker_dump: Text, domain: Domain) -> DialogueStateTracker:
    """Read the json dump from the file and instantiate a tracker it."""

    tracker_json = json.loads(rasa.shared.utils.io.read_file(tracker_dump))
    sender_id = tracker_json.get("sender_id", DEFAULT_SENDER_ID)
    return DialogueStateTracker.from_dict(
        sender_id, tracker_json.get("events", []), domain.slots
    )


def test_dump_and_restore_as_json(
    default_agent: Agent, tmp_path: Path, stories_path: Text
):
    trackers = load_data(stories_path, default_agent.domain)

    for tracker in trackers:
        out_path = tmp_path / "dumped_tracker.json"

        dumped = tracker.current_state(EventVerbosity.AFTER_RESTART)
        rasa.shared.utils.io.dump_obj_as_json_to_file(str(out_path), dumped)

        restored_tracker = _load_tracker_from_json(str(out_path), default_agent.domain)

        assert restored_tracker == tracker


def test_read_json_dump(default_agent: Agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.shared.utils.io.read_file(tracker_dump))

    restored_tracker = _load_tracker_from_json(tracker_dump, default_agent.domain)

    assert len(restored_tracker.events) == 7
    assert restored_tracker.latest_action.get(ACTION_NAME) == "action_listen"
    assert not restored_tracker.is_paused()
    assert restored_tracker.sender_id == "mysender"
    assert restored_tracker.events[-1].timestamp == 1517821726.211042

    restored_state = restored_tracker.current_state(EventVerbosity.AFTER_RESTART)
    assert restored_state == tracker_json


def test_current_state_after_restart(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.shared.utils.io.read_file(tracker_dump))

    tracker_json["events"].insert(3, {"event": "restart"})

    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        default_agent.domain.slots,
    )

    events_after_restart = [e.as_dict() for e in list(tracker.events)[4:]]

    state = tracker.current_state(EventVerbosity.AFTER_RESTART)
    assert state.get("events") == events_after_restart


def test_current_state_all_events(empty_agent: Agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.shared.utils.io.read_file(tracker_dump))

    tracker_json["events"].insert(3, {"event": "restart"})

    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        empty_agent.domain.slots,
    )

    evts = [e.as_dict() for e in tracker.events]

    state = tracker.current_state(EventVerbosity.ALL)
    assert state.get("events") == evts


def test_current_state_no_events(empty_agent: Agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.shared.utils.io.read_file(tracker_dump))

    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        empty_agent.domain.slots,
    )

    state = tracker.current_state(EventVerbosity.NONE)
    assert state.get("events") is None


def test_current_state_applied_events(empty_agent: Agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.shared.utils.io.read_file(tracker_dump))

    # add some events that result in other events not being applied anymore
    tracker_json["events"].insert(1, {"event": "restart"})
    tracker_json["events"].insert(7, {"event": "rewind"})
    tracker_json["events"].insert(8, {"event": "undo"})

    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        empty_agent.domain.slots,
    )

    evts = [e.as_dict() for e in tracker.events]
    applied_events = [evts[2], evts[9]]

    state = tracker.current_state(EventVerbosity.APPLIED)
    assert state.get("events") == applied_events


def test_session_started_not_part_of_applied_events(empty_agent: Agent):
    # take tracker dump and insert a SessionStarted event sequence
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.shared.utils.io.read_file(tracker_dump))
    tracker_json["events"].insert(
        4, {"event": ActionExecuted.type_name, "name": ACTION_SESSION_START_NAME}
    )
    tracker_json["events"].insert(5, {"event": SessionStarted.type_name})

    # initialise a tracker from this list of events
    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        empty_agent.domain.slots,
    )

    # the SessionStart event was at index 5, the tracker's `applied_events()` should
    # be the same as the list of events from index 6 onwards
    assert tracker.applied_events() == list(tracker.events)[6:]


def test_get_last_event_for():
    events = [ActionExecuted("one"), user_uttered("two", 1)]

    tracker = get_tracker(events)

    assert tracker.get_last_event_for(ActionExecuted).action_name == "one"


def test_get_last_event_with_reverted():
    events = [ActionExecuted("one"), ActionReverted(), user_uttered("two", 1)]

    tracker = get_tracker(events)

    assert tracker.get_last_event_for(ActionExecuted) is None


def test_get_last_event_for_with_skip():
    events = [ActionExecuted("one"), user_uttered("two", 1), ActionExecuted("three")]

    tracker = get_tracker(events)

    assert tracker.get_last_event_for(ActionExecuted, skip=1).action_name == "one"


def test_get_last_event_for_with_exclude():
    events = [ActionExecuted("one"), user_uttered("two", 1), ActionExecuted("three")]

    tracker = get_tracker(events)

    assert (
        tracker.get_last_event_for(
            ActionExecuted, action_names_to_exclude=["three"]
        ).action_name
        == "one"
    )


def test_last_executed_has():
    events = [
        ActionExecuted("one"),
        user_uttered("two", 1),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]

    tracker = get_tracker(events)

    assert tracker.last_executed_action_has("one") is True


def test_last_executed_has_not_name():
    events = [
        ActionExecuted("one"),
        user_uttered("two", 1),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]

    tracker = get_tracker(events)

    assert tracker.last_executed_action_has("another") is False


def test_events_metadata():
    # It should be possible to attach arbitrary metadata to any event and then
    # retrieve it after getting the tracker dict representation.
    events = [
        ActionExecuted("one", metadata={"one": 1}),
        user_uttered("two", 1, metadata={"two": 2}),
        ActionExecuted(ACTION_LISTEN_NAME, metadata={"three": 3}),
    ]

    events = get_tracker(events).current_state(EventVerbosity.ALL)["events"]
    assert events[0]["metadata"] == {"one": 1}
    assert events[1]["metadata"] == {"two": 2}
    assert events[2]["metadata"] == {"three": 3}


@pytest.mark.parametrize("key, value", [("asfa", 1), ("htb", None)])
def test_tracker_without_slots(key, value, caplog):
    event = SlotSet(key, value)
    tracker = DialogueStateTracker.from_dict("any", [])
    assert key in tracker.slots
    with caplog.at_level(logging.INFO):
        event.apply_to(tracker)
        v = tracker.get_slot(key)
        assert v == value
    assert len(caplog.records) == 0


@pytest.mark.parametrize(
    "slot_type, initial_value, value_to_set",
    [
        (FloatSlot, 4.234, 2.5),
        (BooleanSlot, True, False),
        (ListSlot, [1, 2, 3], [4, 5, 6]),
        (TextSlot, "some string", "another string"),
        (AnySlot, {"a": "nice dict"}, {"b": "better dict"}),
    ],
)
def test_tracker_does_not_modify_slots(
    slot_type: Type[Slot], initial_value: Any, value_to_set: Any
):
    slot_name = "some-slot"
    slot = slot_type(slot_name, mappings=[{}], initial_value=initial_value)
    tracker = DialogueStateTracker("some-conversation-id", [slot])

    # change the slot value in the tracker
    tracker._set_slot(slot_name, value_to_set)

    # assert that the tracker contains the slot with the modified value
    assert tracker.get_slot(slot_name) == value_to_set

    # assert that the initial slot has not been affected
    assert slot.value == initial_value


@pytest.mark.parametrize(
    "events, expected_applied_events",
    [
        (
            [
                # Form gets triggered.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("fill_whole_form"),
                # Form executes and fills slots.
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet("slot1", "value"),
                SlotSet("slot2", "value2"),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("fill_whole_form"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet("slot1", "value"),
                SlotSet("slot2", "value2"),
            ],
        ),
        (
            [
                # Form gets triggered.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("fill_whole_form"),
                # Form executes and fills all slots right away. Form finishes.
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet("slot1", "value"),
                SlotSet("slot2", "value2"),
                ActiveLoop(None),
                # Form is done. Regular conversation continues.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("intent outside form"),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("fill_whole_form"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet("slot1", "value"),
                SlotSet("slot2", "value2"),
                ActiveLoop(None),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("intent outside form"),
            ],
        ),
        (
            [
                # Form gets triggered.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                # Form executes and requests slot.
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                # User fills slot.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("bye"),
                # Form deactivates after all slots are finished.
                ActionExecuted("loop"),
                SlotSet("slot", "value"),
                ActiveLoop(None),
                SlotSet(REQUESTED_SLOT, None),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                SlotSet("slot", "value"),
                ActiveLoop(None),
                SlotSet(REQUESTED_SLOT, None),
            ],
        ),
        (
            [
                # Form was executed before and finished.
                ActionExecuted("loop"),
                ActiveLoop(None),
                # Form gets triggered again (for whatever reason)..
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                # Form executes and requests slot.
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                # User fills slot.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("bye"),
                # Form deactivates after all slots are finished.
                ActionExecuted("loop"),
                SlotSet("slot", "value"),
                ActiveLoop(None),
                SlotSet(REQUESTED_SLOT, None),
            ],
            [
                ActionExecuted("loop"),
                ActiveLoop(None),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                SlotSet("slot", "value"),
                ActiveLoop(None),
                SlotSet(REQUESTED_SLOT, None),
            ],
        ),
        (
            [
                user_uttered("trigger form"),
                ActionExecuted("form"),
                ActiveLoop("form"),
                SlotSet(REQUESTED_SLOT, "some slot"),
                BotUttered("ask slot"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("fill requested slots"),
                SlotSet("some slot", "value"),
                ActionExecuted("form"),
                SlotSet("some slot", "value"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
            [
                user_uttered("trigger form"),
                ActionExecuted("form"),
                ActiveLoop("form"),
                SlotSet(REQUESTED_SLOT, "some slot"),
                BotUttered("ask slot"),
                SlotSet("some slot", "value"),
                SlotSet("some slot", "value"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
        ),
        (
            [
                user_uttered("trigger form"),
                DefinePrevUserUtteredFeaturization(False),
                ActionExecuted("form"),
                ActiveLoop("form"),
                SlotSet(REQUESTED_SLOT, "some slot"),
                BotUttered("ask slot"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("fill requested slots"),
                SlotSet("some slot", "value"),
                DefinePrevUserUtteredFeaturization(False),
                ActionExecuted("form"),
                SlotSet("some slot", "value"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
            [
                user_uttered("trigger form"),
                DefinePrevUserUtteredFeaturization(False),
                ActionExecuted("form"),
                ActiveLoop("form"),
                SlotSet(REQUESTED_SLOT, "some slot"),
                BotUttered("ask slot"),
                SlotSet("some slot", "value"),
                SlotSet("some slot", "value"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
        ),
    ],
)
def test_applied_events_with_loop_happy_path(
    events: List[Event], expected_applied_events: List[Event]
):
    tracker = DialogueStateTracker.from_events("👋", events)
    applied = tracker.applied_events()

    assert applied == expected_applied_events


@pytest.mark.parametrize(
    "events, expected_applied_events",
    [
        (
            [
                # Form is triggered and requests slot.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                # User sends chitchat instead of answering form.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                # Form rejected execution.
                ActionExecutionRejected("loop"),
                # Action which deals with unhappy path.
                ActionExecuted("handling chitchat"),
                # We immediately return to form after executing an action to handle it.
                ActionExecuted("loop"),
                # Form happy path continues until all slots are filled.
                SlotSet(REQUESTED_SLOT, "bla"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("fill slots"),
                ActionExecuted("loop"),
                SlotSet("slot", "value"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                ActionExecutionRejected("loop"),
                ActionExecuted("handling chitchat"),
                ActionExecuted("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                SlotSet("slot", "value"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
        ),
        (
            [
                # Form gets triggered and requests slots.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                # User sends chitchat instead of answering form.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                # Form rejected execution.
                ActionExecutionRejected("loop"),
                # Unhappy path kicks in.
                ActionExecuted("ask if continue"),
                ActionExecuted(ACTION_LISTEN_NAME),
                # User decides to fill form eventually.
                user_uttered("I want to continue with form"),
                ActionExecuted("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("fill slots"),
                ActionExecuted("loop"),
                SlotSet("slot", "value"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                ActionExecutionRejected("loop"),
                ActionExecuted("ask if continue"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("I want to continue with form"),
                ActionExecuted("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                SlotSet("slot", "value"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
        ),
        (
            [
                # Form gets triggered and requests slots.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                # User sends chitchat instead of answering form.
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                # Form rejected execution.
                ActionExecutionRejected("loop"),
                # Unhappy path kicks in.
                ActionExecuted("ask if continue"),
                ActionExecuted(ACTION_LISTEN_NAME),
                # User wants to quit form.
                user_uttered("Stop the form"),
                ActionExecuted("some action"),
                ActiveLoop(None),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("outside the form"),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                ActionExecutionRejected("loop"),
                ActionExecuted("ask if continue"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("Stop the form"),
                ActionExecuted("some action"),
                ActiveLoop(None),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("outside the form"),
            ],
        ),
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                # Different action than form action after chitchat.
                # This indicates we are in an unhappy path.
                ActionExecuted("handle_chitchat"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                SlotSet(REQUESTED_SLOT, "bla"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                # Different action than form action after chitchat.
                # This indicates we are in an unhappy path.
                ActionExecuted("handle_chitchat"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
            ],
        ),
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                ActionExecuted("handle_chitchat"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("affirm"),
                ActionExecuted("loop"),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                # Different action than form action indicates unhappy path
                ActionExecuted("handle_chitchat"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("affirm"),
                ActionExecuted("loop"),
            ],
        ),
        (
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                DefinePrevUserUtteredFeaturization(False),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                DefinePrevUserUtteredFeaturization(False),
                ActionExecuted("handle_chitchat"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("affirm"),
                DefinePrevUserUtteredFeaturization(False),
                ActionExecuted("loop"),
            ],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("greet"),
                DefinePrevUserUtteredFeaturization(False),
                ActionExecuted("loop"),
                ActiveLoop("loop"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("chitchat"),
                DefinePrevUserUtteredFeaturization(False),
                # Different action than form action indicates unhappy path
                ActionExecuted("handle_chitchat"),
                ActionExecuted(ACTION_LISTEN_NAME),
                user_uttered("affirm"),
                DefinePrevUserUtteredFeaturization(False),
                ActionExecuted("loop"),
            ],
        ),
    ],
)
def test_applied_events_with_loop_unhappy_path(
    events: List[Event], expected_applied_events: List[Event]
):
    tracker = DialogueStateTracker.from_events("👋", events)
    applied = tracker.applied_events()

    assert applied == expected_applied_events


def test_reading_of_trackers_with_legacy_form_events():
    loop_name1 = "my loop"
    loop_name2 = "my form"
    tracker = DialogueStateTracker.from_dict(
        "sender",
        events_as_dict=[
            {"event": ActiveLoop.type_name, LOOP_NAME: loop_name1},
            {"event": LegacyForm.type_name, LOOP_NAME: None},
            {"event": LegacyForm.type_name, LOOP_NAME: loop_name2},
        ],
    )

    expected_events = [ActiveLoop(loop_name1), LegacyForm(None), LegacyForm(loop_name2)]
    assert list(tracker.events) == expected_events
    assert tracker.active_loop.name == loop_name2


def test_writing_trackers_with_legacy_form_events():
    loop_name = "my loop"
    tracker = DialogueStateTracker.from_events(
        "sender", evts=[ActiveLoop(loop_name), LegacyForm(None), LegacyForm("some")]
    )

    events_as_dict = [event.as_dict() for event in tracker.events]

    for event in events_as_dict:
        assert event["event"] == ActiveLoop.type_name


def test_reading_of_trackers_with_legacy_form_validation_events():
    loop_name = "form"
    tracker = DialogueStateTracker.from_dict(
        "sender",
        events_as_dict=[
            {"event": ActiveLoop.type_name, LOOP_NAME: loop_name},
            {"event": LegacyFormValidation.type_name, "name": None, "validate": True},
            {"event": LegacyFormValidation.type_name, "name": None, "validate": False},
        ],
    )

    expected_events = [
        ActiveLoop(loop_name),
        LegacyFormValidation(True),
        LegacyFormValidation(False),
    ]
    actual_events = list(tracker.events)
    assert list(tracker.events) == expected_events
    assert not actual_events[1].is_interrupted
    assert actual_events[2].is_interrupted

    assert tracker.active_loop.is_interrupted


def test_writing_trackers_with_legacy_for_validation_events():
    tracker = DialogueStateTracker.from_events(
        "sender", evts=[LegacyFormValidation(True), LegacyFormValidation(False)]
    )

    events_as_dict = [event.as_dict() for event in tracker.events]

    for event in events_as_dict:
        assert event["event"] == LoopInterrupted.type_name

    assert not events_as_dict[0][LOOP_INTERRUPTED]
    assert events_as_dict[1][LOOP_INTERRUPTED]


@pytest.mark.parametrize(
    "conversation_events,n_subtrackers",
    [
        (
            # conversation contains multiple sessions
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("hi", {"name": "greet"}),
                ActionExecuted("utter_greet"),
                # second session begins here
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("goodbye", {"name": "goodbye"}),
                ActionExecuted("utter_goodbye"),
            ],
            2,
        ),
        (
            # conversation contains only one session
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("hi", {"name": "greet"}),
                ActionExecuted("utter_greet"),
            ],
            1,
        ),
        (
            # this conversation does not contain a session
            [UserUttered("hi", {"name": "greet"}), ActionExecuted("utter_greet")],
            1,
        ),
    ],
)
def test_trackers_for_conversation_sessions(
    conversation_events: List[Event], n_subtrackers: int
):
    import rasa.shared.core.trackers as trackers_module

    tracker = DialogueStateTracker.from_events(
        "some-conversation-ID", conversation_events
    )

    subtrackers = trackers_module.get_trackers_for_conversation_sessions(tracker)

    assert len(subtrackers) == n_subtrackers


def test_policy_predictions_dont_change_persistence():
    original_user_message = UserUttered("hi", intent={"name": "greet"})
    tracker = DialogueStateTracker.from_events(
        "Vova",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("hi", intent={"name": "greet"}),
            DefinePrevUserUtteredFeaturization(True),
            EntitiesAdded(entities=[{"entity": "entity1", "value": "value1"}]),
        ],
    )

    user_message: UserUttered = list(tracker.events)[1]
    # The entities from the policy predictions are accessible
    assert user_message.entities

    actual_serialized = user_message.as_dict()

    # Assert entities predicted by policies are not persisted
    assert not actual_serialized["parse_data"]["entities"]

    expected_serialized = original_user_message.as_dict()
    # don't compare timestamps
    expected_serialized.pop("timestamp")
    actual_serialized.pop("timestamp")

    assert actual_serialized == expected_serialized


@freezegun.freeze_time("2018-01-01")
def test_policy_prediction_reflected_in_tracker_state():
    entities_predicted_by_policy = [{"entity": "entity1", "value": "value1"}]
    nlu_entities = [{"entity": "entityNLU", "value": "value100"}]

    tracker = DialogueStateTracker.from_events(
        "Tester",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                "hi",
                intent={"name": "greet"},
                entities=nlu_entities.copy(),
                message_id="unique",
                metadata={"some": "data"},
            ),
            DefinePrevUserUtteredFeaturization(True),
            EntitiesAdded(entities=entities_predicted_by_policy),
        ],
    )

    tracker_state = tracker.current_state()

    expected_state = {
        "sender_id": "Tester",
        "slots": {},
        "latest_message": {
            "intent": {"name": "greet"},
            "entities": nlu_entities + entities_predicted_by_policy,
            "text": "hi",
            "message_id": "unique",
            "metadata": {"some": "data"},
        },
        "latest_event_time": 1514764800.0,
        "followup_action": None,
        "paused": False,
        "events": None,
        "latest_input_channel": None,
        "active_loop": {},
        "latest_action": {"action_name": "action_listen"},
        "latest_action_name": "action_listen",
    }

    assert tracker_state == expected_state

    # Make sure we didn't change the actual event
    assert tracker.latest_message.parse_data["entities"] == nlu_entities


async def test_fill_slots_for_policy_entities():
    policy_entity, policy_entity_value = "policy_entity", "end-to-end"
    nlu_entity, nlu_entity_value = "nlu_entity", "nlu rocks"
    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            entities:
            - {nlu_entity}
            - {policy_entity}
            slots:
                {nlu_entity}:
                    type: text
                    mappings:
                    - type: from_entity
                      entity: {nlu_entity}
                {policy_entity}:
                    type: text
                    mappings:
                    - type: from_entity
                      entity: {policy_entity}
            """
        )
    )

    tracker = DialogueStateTracker.from_events(
        "some sender",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                "hi",
                intent={"name": "greet"},
                entities=[{"entity": nlu_entity, "value": nlu_entity_value}],
            ),
            DefinePrevUserUtteredFeaturization(True),
            EntitiesAdded(
                entities=[
                    {"entity": policy_entity, "value": policy_entity_value},
                    {"entity": nlu_entity, "value": nlu_entity_value},
                ]
            ),
        ],
        domain=domain,
        slots=domain.slots,
    )

    expected_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(
            "hi",
            intent={"name": "greet"},
            entities=[
                {"entity": nlu_entity, "value": nlu_entity_value},
                # Added by `DefinePrevUserUtteredEntities`
                {"entity": policy_entity, "value": policy_entity_value},
            ],
        ),
        DefinePrevUserUtteredFeaturization(True),
        EntitiesAdded(
            entities=[
                {"entity": policy_entity, "value": policy_entity_value},
                {"entity": nlu_entity, "value": nlu_entity_value},
            ]
        ),
        SlotSet(nlu_entity, nlu_entity_value),
        SlotSet(policy_entity, policy_entity_value),
    ]

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    tracker.update_with_events(events, domain)

    # Slots are correctly set
    assert tracker.slots[nlu_entity].value == nlu_entity_value
    assert tracker.slots[policy_entity].value == policy_entity_value

    for actual, expected in zip(tracker.events, expected_events):
        assert actual == expected


def test_tracker_fingerprinting_consistency():
    slot = TextSlot(name="name", mappings=[{}], influence_conversation=True)
    slot.value = "example"
    tr1 = DialogueStateTracker("test_sender_id", slots=[slot])
    tr2 = DialogueStateTracker("test_sender_id", slots=[slot])
    f1 = tr1.fingerprint()
    f2 = tr2.fingerprint()
    assert f1 == f2


def test_tracker_unique_fingerprint(domain: Domain):
    slot = TextSlot(name="name", mappings=[{}], influence_conversation=True)
    slot.value = "example"
    tr = DialogueStateTracker("test_sender_id", slots=[slot])
    f1 = tr.fingerprint()

    event1 = UserUttered(
        text="hello",
        parse_data={
            "intent": {"name": "greet", "confidence": 0.9604260921478271},
            "entities": [
                {"entity": "city", "value": "London"},
                {"entity": "count", "value": 1},
            ],
            "text": "hi",
            "message_id": "3f4c04602a4947098c574b107d3ccc59",
            "metadata": {},
            "intent_ranking": [
                {"name": "greet", "confidence": 0.9604260921478271},
                {"name": "goodbye", "confidence": 0.01835782080888748},
                {"name": "deny", "confidence": 0.011255578137934208},
            ],
        },
    )
    tr.update(event1)
    f2 = tr.fingerprint()
    assert f1 != f2

    event2 = ActionExecuted(action_name="action_listen")
    tr.update(event2)
    f3 = tr.fingerprint()
    assert f2 != f3


def test_tracker_fingerprint_story_reading(domain: Domain):
    def build_tracker(domain: Domain) -> DialogueStateTracker:
        story_yaml = """
            stories:
            - story: test story
              steps:
              - intent: greet
              - action: utter_greet
            """
        reader = YAMLStoryReader(domain)
        story_steps = reader.read_from_string(story_yaml)
        events = []
        for step in story_steps:
            evts = step.events
            if isinstance(evts, list):
                events += evts
            else:
                events.append(evts)

        slot = TextSlot(name="name", mappings=[{}], influence_conversation=True)
        slot.value = "example"

        tracker = DialogueStateTracker.from_events("sender_id", events, [slot])
        return tracker

    tracker1 = build_tracker(domain)
    f1 = tracker1.fingerprint()

    time.sleep(0.1)

    tracker2 = build_tracker(domain)
    f2 = tracker2.fingerprint()

    assert f1 == f2


def test_model_id_is_added_to_events():
    tracker = DialogueStateTracker("bloop", [])
    tracker.model_id = "some_id"
    tracker.update(ActionExecuted(action_name="test"))
    tracker.update_with_events([UserUttered(), SessionStarted()], None)
    assert all(e.metadata[METADATA_MODEL_ID] == "some_id" for e in tracker.events)


def test_model_id_is_not_added_to_events_with_id():
    tracker = DialogueStateTracker("bloop", [])
    tracker.model_id = "some_id"
    tracker.update(
        ActionExecuted(action_name="test", metadata={METADATA_MODEL_ID: "old_id"})
    )
    assert tracker.events[-1].metadata[METADATA_MODEL_ID] == "old_id"


@pytest.mark.parametrize(
    "event",
    [
        ActionExecuted(action_name="action_listen"),
        UserUttered(),
        SessionStarted(),
        SlotSet("my_slot", 1),
        Restarted(),
        AllSlotsReset(),
        ConversationPaused(),
        ConversationResumed(),
        StoryExported(),
        ActionReverted(),
        UserUtteranceReverted(),
        FollowupAction("my_action"),
        BotUttered("my_text", {"my_data": 1}),
        AgentUttered("my_text", "my_data"),
        ReminderScheduled("my_intent", datetime.datetime.now()),
        DefinePrevUserUtteredFeaturization(False),
        EntitiesAdded([{"entity": "entity1", "value": "value1"}]),
        ActionExecutionRejected("test_action"),
        ActiveLoop("my_form"),
        LoopInterrupted(True),
    ],
)
def test_assistant_id_is_added_to_events(event):
    assistant_id = "some_unique_assistant_name"
    tracker = DialogueStateTracker("123abcd", [])
    tracker.assistant_id = assistant_id
    tracker.update(event)
    assert all(
        event.metadata[ASSISTANT_ID_KEY] == assistant_id for event in tracker.events
    )


def test_assistant_id_is_not_added_to_events_with_assistant_id():
    assistant_id = "some_unique_assistant_name"
    tracker = DialogueStateTracker("123abcd", [])
    tracker.assistant_id = assistant_id
    tracker.update(
        ActionExecuted(action_name="test", metadata={ASSISTANT_ID_KEY: "old_name"})
    )
    assert tracker.events[-1].metadata[ASSISTANT_ID_KEY] == "old_name"
