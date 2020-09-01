import json
import logging
import os
from pathlib import Path
import tempfile
from typing import List, Text, Dict, Any, Type

import fakeredis
import pytest

import rasa.utils.io
from rasa.core import training, restore
from rasa.core.actions.action import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME
from rasa.core.agent import Agent
from rasa.core.constants import REQUESTED_SLOT
from rasa.core.domain import Domain
from rasa.core.events import (
    SlotSet,
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
)
from rasa.core.slots import FloatSlot, BooleanSlot, ListSlot, TextSlot, DataSlot, Slot
from rasa.core.tracker_store import (
    InMemoryTrackerStore,
    RedisTrackerStore,
    SQLTrackerStore,
)
from rasa.core.tracker_store import TrackerStore
from rasa.core.trackers import DialogueStateTracker, EventVerbosity
from tests.core.conftest import (
    DEFAULT_STORIES_FILE,
    EXAMPLE_DOMAINS,
    TEST_DIALOGUES,
    MockedMongoTrackerStore,
)
from tests.core.utilities import (
    tracker_from_dialogue_file,
    read_dialogue_file,
    user_uttered,
    get_tracker,
)

domain = Domain.load("examples/moodbot/domain.yml")


class MockRedisTrackerStore(RedisTrackerStore):
    def __init__(self, _domain: Domain) -> None:
        self.red = fakeredis.FakeStrictRedis()
        self.record_exp = None

        # added in redis==3.3.0, but not yet in fakeredis
        self.red.connection_pool.connection_class.health_check_interval = 0

        TrackerStore.__init__(self, _domain)


def stores_to_be_tested():
    temp = tempfile.mkdtemp()
    return [
        MockRedisTrackerStore(domain),
        InMemoryTrackerStore(domain),
        SQLTrackerStore(domain, db=os.path.join(temp, "rasa.db")),
        MockedMongoTrackerStore(domain),
    ]


def stores_to_be_tested_ids():
    return ["redis-tracker", "in-memory-tracker", "SQL-tracker", "mongo-tracker"]


def test_tracker_duplicate():
    filename = "data/test_dialogues/moodbot.json"
    dialogue = read_dialogue_file(filename)
    tracker = DialogueStateTracker(dialogue.name, domain.slots)
    tracker.recreate_from_dialogue(dialogue)
    num_actions = len(
        [event for event in dialogue.events if isinstance(event, ActionExecuted)]
    )

    # There is always one duplicated tracker more than we have actions,
    # as the tracker also gets duplicated for the
    # action that would be next (but isn't part of the operations)
    assert len(list(tracker.generate_all_prior_trackers())) == num_actions + 1


@pytest.mark.parametrize("store", stores_to_be_tested(), ids=stores_to_be_tested_ids())
def test_tracker_store_storage_and_retrieval(store):
    tracker = store.get_or_create_tracker("some-id")
    # the retrieved tracker should be empty
    assert tracker.sender_id == "some-id"

    # Action listen should be in there
    assert list(tracker.events) == [ActionExecuted(ACTION_LISTEN_NAME)]

    # lets log a test message
    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(UserUttered("/greet", intent, []))
    assert tracker.latest_message.intent.get("name") == "greet"
    store.save(tracker)

    # retrieving the same tracker should result in the same tracker
    retrieved_tracker = store.get_or_create_tracker("some-id")
    assert retrieved_tracker.sender_id == "some-id"
    assert len(retrieved_tracker.events) == 2
    assert retrieved_tracker.latest_message.intent.get("name") == "greet"

    # getting another tracker should result in an empty tracker again
    other_tracker = store.get_or_create_tracker("some-other-id")
    assert other_tracker.sender_id == "some-other-id"
    assert len(other_tracker.events) == 1


@pytest.mark.parametrize("store", stores_to_be_tested(), ids=stores_to_be_tested_ids())
@pytest.mark.parametrize("pair", zip(TEST_DIALOGUES, EXAMPLE_DOMAINS))
def test_tracker_store(store, pair):
    filename, domainpath = pair
    domain = Domain.load(domainpath)
    tracker = tracker_from_dialogue_file(filename, domain)
    store.save(tracker)
    restored = store.retrieve(tracker.sender_id)
    assert restored == tracker


async def test_tracker_write_to_story(tmp_path: Path, moodbot_domain: Domain):
    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )
    p = tmp_path / "export.md"
    tracker.export_stories_to_file(str(p))
    trackers = await training.load_data(
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
    assert recovered.events[4].intent == {"confidence": 1.0, "name": "mood_unhappy"}


async def test_tracker_state_regression_without_bot_utterance(default_agent: Agent):
    sender_id = "test_tracker_state_regression_without_bot_utterance"
    for i in range(0, 2):
        await default_agent.handle_text("/greet", sender_id=sender_id)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

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
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    expected = [
        "action_session_start",
        None,
        "action_listen",
        "greet",
        "utter_greet",
        None,
        "action_listen",
        "greet",
        "utter_greet",
        None,
        "action_listen",
    ]

    assert [e.as_story_string() for e in tracker.events] == expected


async def test_bot_utterance_comes_after_action_event(default_agent):
    sender_id = "test_bot_utterance_comes_after_action_event"

    await default_agent.handle_text("/greet", sender_id=sender_id)

    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    # important is, that the 'bot' comes after the second 'action' and not
    # before
    expected = [
        "action",
        "session_started",
        "action",
        "user",
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
            ["greet", "bye"],
        ),
    ],
)
def test_get_latest_entity_values(
    entities: List[Dict[Text, Any]], expected_values: List[Text], default_domain: Domain
):
    entity_type = entities[0].get("entity")
    entity_role = entities[0].get("role")
    entity_group = entities[0].get("group")

    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0
    assert list(tracker.get_latest_entity_values(entity_type)) == []

    intent = {"name": "greet", "confidence": 1.0}
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


def test_tracker_update_slots_with_entity(default_domain: Domain):
    tracker = DialogueStateTracker("default", default_domain.slots)

    test_entity = default_domain.entities[0]
    expected_slot_value = "test user"

    intent = {"name": "greet", "confidence": 1.0}
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
        default_domain,
    )

    assert tracker.get_slot(test_entity) == expected_slot_value


def test_restart_event(default_domain: Domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("/greet", intent, []))
    tracker.update(ActionExecuted("my_action"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    assert len(tracker.events) == 4
    assert tracker.latest_message.text == "/greet"
    assert len(list(tracker.generate_all_prior_trackers())) == 4

    tracker.update(Restarted())

    assert len(tracker.events) == 5
    assert tracker.followup_action is not None
    assert tracker.followup_action == ACTION_LISTEN_NAME
    assert tracker.latest_message.text is None
    assert len(list(tracker.generate_all_prior_trackers())) == 1

    dialogue = tracker.as_dialogue()

    recovered = DialogueStateTracker("default", default_domain.slots)
    recovered.recreate_from_dialogue(dialogue)

    assert recovered.current_state() == tracker.current_state()
    assert len(recovered.events) == 5
    assert recovered.latest_message.text is None
    assert len(list(recovered.generate_all_prior_trackers())) == 1


def test_session_start(default_domain: Domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    # add a SessionStarted event
    tracker.update(SessionStarted())

    # tracker has one event
    assert len(tracker.events) == 1


def test_revert_action_event(default_domain: Domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("/greet", intent, []))
    tracker.update(ActionExecuted("my_action"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    # Expecting count of 4:
    #   +3 executed actions
    #   +1 final state
    assert tracker.latest_action_name == ACTION_LISTEN_NAME
    assert len(list(tracker.generate_all_prior_trackers())) == 4

    tracker.update(ActionReverted())

    # Expecting count of 3:
    #   +3 executed actions
    #   +1 final state
    #   -1 reverted action
    assert tracker.latest_action_name == "my_action"
    assert len(list(tracker.generate_all_prior_trackers())) == 3

    dialogue = tracker.as_dialogue()

    recovered = DialogueStateTracker("default", default_domain.slots)
    recovered.recreate_from_dialogue(dialogue)

    assert recovered.current_state() == tracker.current_state()
    assert tracker.latest_action_name == "my_action"
    assert len(list(tracker.generate_all_prior_trackers())) == 3


def test_revert_user_utterance_event(default_domain: Domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent1 = {"name": "greet", "confidence": 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("/greet", intent1, []))
    tracker.update(ActionExecuted("my_action_1"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    intent2 = {"name": "goodbye", "confidence": 1.0}
    tracker.update(UserUttered("/goodbye", intent2, []))
    tracker.update(ActionExecuted("my_action_2"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    # Expecting count of 6:
    #   +5 executed actions
    #   +1 final state
    assert tracker.latest_action_name == ACTION_LISTEN_NAME
    assert len(list(tracker.generate_all_prior_trackers())) == 6

    tracker.update(UserUtteranceReverted())

    # Expecting count of 3:
    #   +5 executed actions
    #   +1 final state
    #   -2 rewound actions associated with the /goodbye
    #   -1 rewound action from the listen right before /goodbye
    assert tracker.latest_action_name == "my_action_1"
    assert len(list(tracker.generate_all_prior_trackers())) == 3

    dialogue = tracker.as_dialogue()

    recovered = DialogueStateTracker("default", default_domain.slots)
    recovered.recreate_from_dialogue(dialogue)

    assert recovered.current_state() == tracker.current_state()
    assert tracker.latest_action_name == "my_action_1"
    assert len(list(tracker.generate_all_prior_trackers())) == 3


def test_traveling_back_in_time(default_domain: Domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("/greet", intent, []))

    import time

    time.sleep(1)
    time_for_timemachine = time.time()
    time.sleep(1)

    tracker.update(ActionExecuted("my_action"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    # Expecting count of 4:
    #   +3 executed actions
    #   +1 final state
    assert tracker.latest_action_name == ACTION_LISTEN_NAME
    assert len(tracker.events) == 4
    assert len(list(tracker.generate_all_prior_trackers())) == 4

    tracker = tracker.travel_back_in_time(time_for_timemachine)

    # Expecting count of 2:
    #   +1 executed actions
    #   +1 final state
    assert tracker.latest_action_name == ACTION_LISTEN_NAME
    assert len(tracker.events) == 2
    assert len(list(tracker.generate_all_prior_trackers())) == 2


async def test_dump_and_restore_as_json(default_agent: Agent, tmp_path: Path):
    trackers = await default_agent.load_data(DEFAULT_STORIES_FILE)

    for tracker in trackers:
        out_path = tmp_path / "dumped_tracker.json"

        dumped = tracker.current_state(EventVerbosity.AFTER_RESTART)
        rasa.utils.io.dump_obj_as_json_to_file(str(out_path), dumped)

        restored_tracker = restore.load_tracker_from_json(
            str(out_path), default_agent.domain
        )

        assert restored_tracker == tracker


def test_read_json_dump(default_agent: Agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))

    restored_tracker = restore.load_tracker_from_json(
        tracker_dump, default_agent.domain
    )

    assert len(restored_tracker.events) == 7
    assert restored_tracker.latest_action_name == "action_listen"
    assert not restored_tracker.is_paused()
    assert restored_tracker.sender_id == "mysender"
    assert restored_tracker.events[-1].timestamp == 1517821726.211042

    restored_state = restored_tracker.current_state(EventVerbosity.AFTER_RESTART)
    assert restored_state == tracker_json


def test_current_state_after_restart(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))

    tracker_json["events"].insert(3, {"event": "restart"})

    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        default_agent.domain.slots,
    )

    events_after_restart = [e.as_dict() for e in list(tracker.events)[4:]]

    state = tracker.current_state(EventVerbosity.AFTER_RESTART)
    assert state.get("events") == events_after_restart


def test_current_state_all_events(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))

    tracker_json["events"].insert(3, {"event": "restart"})

    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        default_agent.domain.slots,
    )

    evts = [e.as_dict() for e in tracker.events]

    state = tracker.current_state(EventVerbosity.ALL)
    assert state.get("events") == evts


def test_current_state_no_events(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))

    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        default_agent.domain.slots,
    )

    state = tracker.current_state(EventVerbosity.NONE)
    assert state.get("events") is None


def test_current_state_applied_events(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))

    # add some events that result in other events not being applied anymore
    tracker_json["events"].insert(1, {"event": "restart"})
    tracker_json["events"].insert(7, {"event": "rewind"})
    tracker_json["events"].insert(8, {"event": "undo"})

    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        default_agent.domain.slots,
    )

    evts = [e.as_dict() for e in tracker.events]
    applied_events = [evts[2], evts[9]]

    state = tracker.current_state(EventVerbosity.APPLIED)
    assert state.get("events") == applied_events


def test_session_started_not_part_of_applied_events(default_agent: Agent):
    # take tracker dump and insert a SessionStarted event sequence
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))
    tracker_json["events"].insert(
        4, {"event": ActionExecuted.type_name, "name": ACTION_SESSION_START_NAME}
    )
    tracker_json["events"].insert(5, {"event": SessionStarted.type_name})

    # initialise a tracker from this list of events
    tracker = DialogueStateTracker.from_dict(
        tracker_json.get("sender_id"),
        tracker_json.get("events", []),
        default_agent.domain.slots,
    )

    # the SessionStart event was at index 5, the tracker's `applied_events()` should
    # be the same as the list of events from index 6 onwards
    assert tracker.applied_events() == list(tracker.events)[6:]


async def test_tracker_dump_e2e_story(default_agent):
    sender_id = "test_tracker_dump_e2e_story"

    await default_agent.handle_text("/greet", sender_id=sender_id)
    await default_agent.handle_text("/goodbye", sender_id=sender_id)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    story = tracker.export_stories(e2e=True)
    assert story.strip().split("\n") == [
        "## test_tracker_dump_e2e_story",
        "* greet: /greet",
        "    - utter_greet",
        "* goodbye: /goodbye",
    ]


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
        (DataSlot, {"a": "nice dict"}, {"b": "better dict"}),
    ],
)
def test_tracker_does_not_modify_slots(
    slot_type: Type[Slot], initial_value: Any, value_to_set: Any
):
    slot_name = "some-slot"
    slot = slot_type(slot_name, initial_value)
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
            {"event": ActiveLoop.type_name, "name": loop_name1},
            {"event": LegacyForm.type_name, "name": None},
            {"event": LegacyForm.type_name, "name": loop_name2},
        ],
    )

    expected_events = [ActiveLoop(loop_name1), LegacyForm(None), LegacyForm(loop_name2)]
    assert list(tracker.events) == expected_events
    assert tracker.active_loop["name"] == loop_name2


def test_writing_trackers_with_legacy_form_events():
    loop_name = "my loop"
    tracker = DialogueStateTracker.from_events(
        "sender", evts=[ActiveLoop(loop_name), LegacyForm(None), LegacyForm("some")]
    )

    events_as_dict = [event.as_dict() for event in tracker.events]

    for event in events_as_dict:
        assert event["event"] == ActiveLoop.type_name


def test_change_form_to_deprecation_warning():
    tracker = DialogueStateTracker.from_events("conversation", evts=[])
    new_form = "new form"
    with pytest.warns(DeprecationWarning):
        tracker.change_form_to(new_form)

    assert tracker.active_loop_name() == new_form
