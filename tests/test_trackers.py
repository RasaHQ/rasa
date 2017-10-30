from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob

import pytest

from rasa_core.actions.action import ActionListen, ACTION_LISTEN_NAME
from rasa_core.agent import Agent
from rasa_core.conversation import Topic
from rasa_core.domain import TemplateDomain
from rasa_core.events import UserUttered, TopicSet, ActionExecuted, SlotSet, \
    Restarted, ActionReverted
from rasa_core.featurizers import BinaryFeaturizer
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.tracker_store import InMemoryTrackerStore, RedisTrackerStore
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training_utils import extract_stories_from_file, STORY_START
from tests.utilities import tracker_from_dialogue_file, read_dialogue_file

domain = TemplateDomain.load("data/test_domains/default_with_topic.yml")


def stores_to_be_tested():
    return [RedisTrackerStore(domain, mock=True),
            InMemoryTrackerStore(domain)]


def stores_to_be_tested_ids():
    return ["redis-tracker",
            "in-memory-tracker"]


def test_tracker_duplicate():
    filename = "data/test_dialogues/inform_no_change.json"
    dialogue = read_dialogue_file(filename)
    dialogue_topics = set([Topic(t.topic)
                           for t in dialogue.events
                           if isinstance(t, TopicSet)])
    domain.topics.extend(dialogue_topics)
    tracker = DialogueStateTracker(dialogue.name, domain.slots,
                                   domain.topics, domain.default_topic)
    tracker.recreate_from_dialogue(dialogue)
    num_actions = len([event
                       for event in dialogue.events
                       if isinstance(event, ActionExecuted)])

    # There is always one duplicated tracker more than we have actions,
    # as the tracker also gets duplicated for the
    # action that would be next (but isn't part of the operations)
    assert len(list(tracker.generate_all_prior_states())) == num_actions + 1


@pytest.mark.parametrize("store", stores_to_be_tested(),
                         ids=stores_to_be_tested_ids())
def test_tracker_store_storage_and_retrieval(store):
    tracker = store.get_or_create_tracker("some-id")
    # the retrieved tracker should be empty
    assert tracker.sender_id == "some-id"

    # Action listen should be in there
    assert list(tracker.events) == [ActionExecuted(ActionListen().name())]

    # lets log a test message
    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(UserUttered("_greet", intent, []))
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
    stories = extract_stories_from_file(p.strpath, default_domain)
    assert len(stories) == 1
    assert len(stories[0].story_steps) == 1
    assert len(stories[0].story_steps[0].events) == 4
    assert stories[0].story_steps[0].start_checkpoint_name() == STORY_START
    assert stories[0].story_steps[0].events[3] == SlotSet("location", "central")


def test_tracker_state_regression(default_agent):
    sender = "test_tracker_state_regression"
    n_actions = []
    for i in range(0, 2):
        default_agent.handle_message("_greet", sender=sender)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender)

    # Ensures that the tracker has changed between the utterances
    # (and wasn't reset in between them)
    expected = ("action_listen;"
                "_greet;utter_greet;action_listen;"
                "_greet;action_listen")
    assert ";".join([e.as_story_string() for e in tracker.events]) == expected


def test_restart_event(default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots,
                                   default_domain.topics,
                                   default_domain.default_topic)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("_greet", intent, []))
    tracker.update(ActionExecuted("my_action"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    assert len(tracker.events) == 4
    assert tracker.latest_message.text == "_greet"
    assert len(list(tracker.generate_all_prior_states())) == 4

    tracker.update(Restarted())

    assert len(tracker.events) == 6
    assert tracker.latest_message.text is None
    assert len(list(tracker.generate_all_prior_states())) == 2

    dialogue = tracker.as_dialogue()

    recovered = DialogueStateTracker("default", default_domain.slots,
                                     default_domain.topics,
                                     default_domain.default_topic)
    recovered.recreate_from_dialogue(dialogue)

    assert recovered.current_state() == tracker.current_state()
    assert len(recovered.events) == 6
    assert recovered.latest_message.text is None
    assert len(list(recovered.generate_all_prior_states())) == 2


def test_revert_action_event(default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots,
                                   default_domain.topics,
                                   default_domain.default_topic)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    intent = {"name": "greet", "confidence": 1.0}
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))
    tracker.update(UserUttered("_greet", intent, []))
    tracker.update(ActionExecuted("my_action"))
    tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

    assert tracker.latest_action_name == ACTION_LISTEN_NAME
    assert len(list(tracker.generate_all_prior_states())) == 4

    tracker.update(ActionReverted())

    assert tracker.latest_action_name == "my_action"
    assert len(list(tracker.generate_all_prior_states())) == 3

    dialogue = tracker.as_dialogue()

    recovered = DialogueStateTracker("default", default_domain.slots,
                                     default_domain.topics,
                                     default_domain.default_topic)
    recovered.recreate_from_dialogue(dialogue)

    assert recovered.current_state() == tracker.current_state()
    assert tracker.latest_action_name == "my_action"
    assert len(list(tracker.generate_all_prior_states())) == 3
