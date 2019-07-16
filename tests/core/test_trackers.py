import asyncio
import json

import fakeredis
import pytest
import tempfile
import os
import logging

import rasa.utils.io
from rasa.core import training, restore
from rasa.core import utils
from rasa.core.slots import Slot
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import Domain
from rasa.core.events import (
    SlotSet,
    UserUttered,
    ActionExecuted,
    Restarted,
    ActionReverted,
    UserUtteranceReverted,
)
from rasa.core.tracker_store import (
    InMemoryTrackerStore,
    RedisTrackerStore,
    SQLTrackerStore,
)
from rasa.core.tracker_store import TrackerStore
from rasa.core.trackers import DialogueStateTracker, EventVerbosity
from tests.core.conftest import DEFAULT_STORIES_FILE, EXAMPLE_DOMAINS, TEST_DIALOGUES
from tests.core.utilities import (
    tracker_from_dialogue_file,
    read_dialogue_file,
    user_uttered,
    get_tracker,
)

domain = Domain.load("examples/moodbot/domain.yml")


@pytest.fixture(scope="module")
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)
    yield loop
    loop.close()


class MockRedisTrackerStore(RedisTrackerStore):
    def __init__(self, domain):
        self.red = fakeredis.FakeStrictRedis()
        self.record_exp = None
        TrackerStore.__init__(self, domain)


def stores_to_be_tested():
    temp = tempfile.mkdtemp()
    return [
        MockRedisTrackerStore(domain),
        InMemoryTrackerStore(domain),
        SQLTrackerStore(domain, db=os.path.join(temp, "rasa.db")),
    ]


def stores_to_be_tested_ids():
    return ["redis-tracker", "in-memory-tracker", "SQL-tracker"]


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


async def test_tracker_write_to_story(tmpdir, moodbot_domain):
    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/moodbot.json", moodbot_domain
    )
    p = tmpdir.join("export.md")
    tracker.export_stories_to_file(p.strpath)
    trackers = await training.load_data(
        p.strpath,
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


async def test_tracker_state_regression_without_bot_utterance(default_agent):
    sender_id = "test_tracker_state_regression_without_bot_utterance"
    for i in range(0, 2):
        await default_agent.handle_message("/greet", sender_id=sender_id)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    # Ensures that the tracker has changed between the utterances
    # (and wasn't reset in between them)
    expected = "action_listen;greet;utter_greet;action_listen;greet;action_listen"
    assert (
        ";".join([e.as_story_string() for e in tracker.events if e.as_story_string()])
        == expected
    )


async def test_tracker_state_regression_with_bot_utterance(default_agent):
    sender_id = "test_tracker_state_regression_with_bot_utterance"
    for i in range(0, 2):
        await default_agent.handle_message("/greet", sender_id=sender_id)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    expected = [
        "action_listen",
        "greet",
        "utter_greet",
        None,
        "action_listen",
        "greet",
        "action_listen",
    ]

    assert [e.as_story_string() for e in tracker.events] == expected


async def test_bot_utterance_comes_after_action_event(default_agent):
    sender_id = "test_bot_utterance_comes_after_action_event"

    await default_agent.handle_message("/greet", sender_id=sender_id)

    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    # important is, that the 'bot' comes after the second 'action' and not
    # before
    expected = ["action", "user", "action", "bot", "action"]

    assert [e.type_name for e in tracker.events] == expected


def test_tracker_entity_retrieval(default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0
    assert list(tracker.get_latest_entity_values("entity_name")) == []

    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(
        UserUttered(
            "/greet",
            intent,
            [
                {
                    "start": 1,
                    "end": 5,
                    "value": "greet",
                    "entity": "entity_name",
                    "extractor": "manual",
                }
            ],
        )
    )
    assert list(tracker.get_latest_entity_values("entity_name")) == ["greet"]
    assert list(tracker.get_latest_entity_values("unknown")) == []


def test_tracker_update_slots_with_entity(default_domain):
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


def test_restart_event(default_domain):
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


def test_revert_action_event(default_domain):
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


def test_revert_user_utterance_event(default_domain):
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


def test_traveling_back_in_time(default_domain):
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


async def test_dump_and_restore_as_json(default_agent, tmpdir_factory):
    trackers = await default_agent.load_data(DEFAULT_STORIES_FILE)

    for tracker in trackers:
        out_path = tmpdir_factory.mktemp("tracker").join("dumped_tracker.json")

        dumped = tracker.current_state(EventVerbosity.AFTER_RESTART)
        utils.dump_obj_as_json_to_file(out_path.strpath, dumped)

        restored_tracker = restore.load_tracker_from_json(
            out_path.strpath, default_agent.domain
        )

        assert restored_tracker == tracker


def test_read_json_dump(default_agent):
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


async def test_tracker_dump_e2e_story(default_agent):
    sender_id = "test_tracker_dump_e2e_story"

    await default_agent.handle_message("/greet", sender_id=sender_id)
    await default_agent.handle_message("/goodbye", sender_id=sender_id)
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
