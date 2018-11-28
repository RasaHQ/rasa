import glob
import json

import fakeredis
import pytest

from rasa_core import training, restore
from rasa_core import utils
from rasa_core.actions.action import ActionListen, ACTION_LISTEN_NAME
from rasa_core.channels import UserMessage
from rasa_core.domain import Domain
from rasa_core.events import (
    UserUttered, ActionExecuted, Restarted, ActionReverted,
    UserUtteranceReverted)
from rasa_core.tracker_store import InMemoryTrackerStore, RedisTrackerStore
from rasa_core.tracker_store import (
    TrackerStore)
from rasa_core.trackers import DialogueStateTracker, EventVerbosity
from tests.conftest import DEFAULT_STORIES_FILE
from tests.utilities import tracker_from_dialogue_file, read_dialogue_file

domain = Domain.load("data/test_domains/default.yml")


class MockRedisTrackerStore(RedisTrackerStore):
    def __init__(self, domain):
        self.red = fakeredis.FakeStrictRedis()
        self.record_exp = None
        TrackerStore.__init__(self, domain)


def stores_to_be_tested():
    return [MockRedisTrackerStore(domain),
            InMemoryTrackerStore(domain)]


def stores_to_be_tested_ids():
    return ["redis-tracker",
            "in-memory-tracker"]


def test_tracker_duplicate():
    filename = "data/test_dialogues/inform_no_change.json"
    dialogue = read_dialogue_file(filename)
    tracker = DialogueStateTracker(dialogue.name, domain.slots)
    tracker.recreate_from_dialogue(dialogue)
    num_actions = len([event
                       for event in dialogue.events
                       if isinstance(event, ActionExecuted)])

    # There is always one duplicated tracker more than we have actions,
    # as the tracker also gets duplicated for the
    # action that would be next (but isn't part of the operations)
    assert len(list(tracker.generate_all_prior_trackers())) == num_actions + 1


@pytest.mark.parametrize("store", stores_to_be_tested(),
                         ids=stores_to_be_tested_ids())
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


@pytest.mark.parametrize("store", stores_to_be_tested(),
                         ids=stores_to_be_tested_ids())
@pytest.mark.parametrize("filename", glob.glob('data/test_dialogues/*json'))
def test_tracker_store(filename, store):
    tracker = tracker_from_dialogue_file(filename, domain)
    store.save(tracker)
    restored = store.retrieve(tracker.sender_id)
    assert restored == tracker


def test_tracker_write_to_story(tmpdir, default_domain):
    tracker = tracker_from_dialogue_file(
        "data/test_dialogues/enter_name.json", default_domain)
    p = tmpdir.join("export.md")
    tracker.export_stories_to_file(p.strpath)
    trackers = training.load_data(
        p.strpath,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False
    )
    assert len(trackers) == 1
    recovered = trackers[0]
    assert len(recovered.events) == 7
    assert recovered.events[5].type_name == "slot"
    assert recovered.events[5].key == "name"
    assert recovered.events[5].value == "holger"


def test_tracker_state_regression_without_bot_utterance(default_agent):
    sender_id = "test_tracker_state_regression_without_bot_utterance"
    for i in range(0, 2):
        default_agent.handle_message("/greet", sender_id=sender_id)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    # Ensures that the tracker has changed between the utterances
    # (and wasn't reset in between them)
    expected = ("action_listen;"
                "greet;utter_greet;action_listen;"
                "greet;action_listen")
    assert ";".join([e.as_story_string() for e in
                     tracker.events if e.as_story_string()]) == expected


def test_tracker_state_regression_with_bot_utterance(default_agent):
    sender_id = "test_tracker_state_regression_with_bot_utterance"
    for i in range(0, 2):
        default_agent.handle_message("/greet", sender_id=sender_id)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    expected = ["action_listen", "greet", "utter_greet", None,
                "action_listen", "greet", "action_listen"]

    assert [e.as_story_string() for e in tracker.events] == expected


def test_bot_utterance_comes_after_action_event(default_agent):
    sender_id = "test_bot_utterance_comes_after_action_event"

    default_agent.handle_message("/greet", sender_id=sender_id)

    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    # important is, that the 'bot' comes after the second 'action' and not
    # before
    expected = ['action', 'user', 'action', 'bot', 'action']

    assert [e.type_name for e in tracker.events] == expected


def test_tracker_entity_retrieval(default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0
    assert list(tracker.get_latest_entity_values("entity_name")) == []

    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(UserUttered("/greet", intent, [{
        "start": 1,
        "end": 5,
        "value": "greet",
        "entity": "entity_name",
        "extractor": "manual"
    }]))
    assert list(tracker.get_latest_entity_values("entity_name")) == ["greet"]
    assert list(tracker.get_latest_entity_values("unknown")) == []


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


def test_dump_and_restore_as_json(default_agent, tmpdir_factory):
    trackers = default_agent.load_data(DEFAULT_STORIES_FILE)

    for tracker in trackers:
        out_path = tmpdir_factory.mktemp("tracker").join("dumped_tracker.json")

        dumped = tracker.current_state(EventVerbosity.AFTER_RESTART)
        utils.dump_obj_as_json_to_file(out_path.strpath, dumped)

        restored_tracker = restore.load_tracker_from_json(out_path.strpath,
                                                          default_agent.domain)

        assert restored_tracker == tracker


def test_read_json_dump(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(utils.read_file(tracker_dump))

    restored_tracker = restore.load_tracker_from_json(tracker_dump,
                                                      default_agent.domain)

    assert len(restored_tracker.events) == 7
    assert restored_tracker.latest_action_name == "action_listen"
    assert not restored_tracker.is_paused()
    assert restored_tracker.sender_id == "mysender"
    assert restored_tracker.events[-1].timestamp == 1517821726.211042

    restored_state = restored_tracker.current_state(
        EventVerbosity.AFTER_RESTART)
    assert restored_state == tracker_json


def test_current_state_after_restart(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(utils.read_file(tracker_dump))

    tracker_json["events"].insert(3, {"event": "restart"})

    tracker = DialogueStateTracker.from_dict(tracker_json.get("sender_id"),
                                             tracker_json.get("events", []),
                                             default_agent.domain.slots)

    events_after_restart = [e.as_dict() for e in list(tracker.events)[4:]]

    state = tracker.current_state(EventVerbosity.AFTER_RESTART)
    assert state.get("events") == events_after_restart


def test_current_state_all_events(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(utils.read_file(tracker_dump))

    tracker_json["events"].insert(3, {"event": "restart"})

    tracker = DialogueStateTracker.from_dict(tracker_json.get("sender_id"),
                                             tracker_json.get("events", []),
                                             default_agent.domain.slots)

    evts = [e.as_dict() for e in tracker.events]

    state = tracker.current_state(EventVerbosity.ALL)
    assert state.get("events") == evts


def test_current_state_no_events(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(utils.read_file(tracker_dump))

    tracker = DialogueStateTracker.from_dict(tracker_json.get("sender_id"),
                                             tracker_json.get("events", []),
                                             default_agent.domain.slots)

    state = tracker.current_state(EventVerbosity.NONE)
    assert state.get("events") is None


def test_current_state_applied_events(default_agent):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(utils.read_file(tracker_dump))

    # add some events that result in other events not being applied anymore
    tracker_json["events"].insert(1, {"event": "restart"})
    tracker_json["events"].insert(7, {"event": "rewind"})
    tracker_json["events"].insert(8, {"event": "undo"})

    tracker = DialogueStateTracker.from_dict(tracker_json.get("sender_id"),
                                             tracker_json.get("events", []),
                                             default_agent.domain.slots)

    evts = [e.as_dict() for e in tracker.events]
    applied_events = [evts[2], evts[9]]

    state = tracker.current_state(EventVerbosity.APPLIED)
    assert state.get("events") == applied_events


def test_tracker_dump_e2e_story(default_agent):
    sender_id = "test_tracker_dump_e2e_story"

    default_agent.handle_message("/greet", sender_id=sender_id)
    default_agent.handle_message("/goodbye", sender_id=sender_id)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    story = tracker.export_stories(e2e=True)
    assert story.strip().split('\n') == [
        "## test_tracker_dump_e2e_story",
        "* greet: /greet",
        "    - utter_greet",
        "* goodbye: /goodbye"]
