import json

import pytest

from rasa.core.tracker_stores.tracker_store import InMemoryTrackerStore
from rasa.shared.core.conversation import Dialogue
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SessionStarted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from tests.core.utilities import (
    tracker_from_dialogue,
    tracker_from_dialogue_with_user_id,
)
from tests.dialogues import (
    TEST_DEFAULT_DIALOGUE,
    TEST_DIALOGUE_WITH_USER_ID,
    TEST_DIALOGUES,
    TEST_DOMAINS_FOR_DIALOGUES,
)


@pytest.mark.parametrize("pair", zip(TEST_DIALOGUES, TEST_DOMAINS_FOR_DIALOGUES))
async def test_inmemory_tracker_store(pair):
    dialogue, domainpath = pair
    domain = Domain.load(domainpath)
    tracker = tracker_from_dialogue(dialogue, domain)
    tracker_store = InMemoryTrackerStore(domain)
    await tracker_store.save(tracker)
    restored = await tracker_store.retrieve(tracker.sender_id)
    assert restored == tracker


def test_tracker_default(domain: Domain):
    tracker = tracker_from_dialogue(TEST_DEFAULT_DIALOGUE, domain)
    assert tracker.get_slot("name") == "Peter"
    assert tracker.get_slot("price") is None  # slot doesn't exist!


def test_dialogue_from_parameters(domain: Domain):
    tracker = tracker_from_dialogue(TEST_DEFAULT_DIALOGUE, domain)
    serialised_dialogue = InMemoryTrackerStore.serialise_tracker(tracker)
    deserialised_dialogue = Dialogue.from_parameters(json.loads(serialised_dialogue))

    assert deserialised_dialogue.user_id is None
    assert tracker.as_dialogue().as_dict()["user_id"] is None
    assert tracker.as_dialogue().as_dict() == deserialised_dialogue.as_dict()


def test_dialogue_from_parameters_with_user_id(domain: Domain):
    tracker = tracker_from_dialogue_with_user_id(TEST_DIALOGUE_WITH_USER_ID, domain)
    serialised_dialogue = InMemoryTrackerStore.serialise_tracker(tracker)
    deserialised_dialogue = Dialogue.from_parameters(json.loads(serialised_dialogue))

    assert deserialised_dialogue.user_id == "test_user_id"
    assert tracker.as_dialogue().as_dict()["user_id"] == "test_user_id"
    assert tracker.as_dialogue().as_dict() == deserialised_dialogue.as_dict()


def test_dialogue_as_dict_includes_conversation_started_timestamp(domain: Domain):
    tracker = tracker_from_dialogue(TEST_DEFAULT_DIALOGUE, domain)
    tracker.ensure_conversation_started_timestamp()

    dialogue = tracker.as_dialogue()
    dialogue_dict = dialogue.as_dict()

    assert "conversation_started_timestamp" in dialogue_dict
    assert (
        dialogue_dict["conversation_started_timestamp"]
        == tracker.conversation_started_timestamp
    )


def test_dialogue_from_parameters_with_conversation_started_timestamp(
    domain: Domain,
):
    tracker = tracker_from_dialogue(TEST_DEFAULT_DIALOGUE, domain)
    tracker.ensure_conversation_started_timestamp()

    serialised_dialogue = InMemoryTrackerStore.serialise_tracker(tracker)
    deserialised_dialogue = Dialogue.from_parameters(json.loads(serialised_dialogue))

    assert (
        deserialised_dialogue.conversation_started_timestamp
        == tracker.conversation_started_timestamp
    )
    assert tracker.as_dialogue().as_dict() == deserialised_dialogue.as_dict()


def test_dialogue_from_parameters_without_conversation_started_timestamp(
    domain: Domain,
):
    dialogue_dict = {
        "name": "test_sender",
        "events": [{"event": "session_started", "timestamp": 1234567890.0}],
        "user_id": None,
    }

    dialogue = Dialogue.from_parameters(dialogue_dict)

    assert dialogue.conversation_started_timestamp is None


def test_dialogue_round_trip_serialization_with_conversation_started_timestamp(
    domain: Domain,
):
    tracker = DialogueStateTracker.from_events(
        "test_sender",
        [SessionStarted(), UserUttered("Hello")],
        domain=domain,
    )
    tracker.ensure_conversation_started_timestamp()
    original_timestamp = tracker.conversation_started_timestamp

    # Serialize
    dialogue = tracker.as_dialogue()
    dialogue_dict = dialogue.as_dict()

    # Deserialize
    restored_dialogue = Dialogue.from_parameters(dialogue_dict)

    assert restored_dialogue.conversation_started_timestamp == original_timestamp
