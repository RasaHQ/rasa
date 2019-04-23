import json

import jsonpickle
import pytest

import rasa.utils.io
from rasa.core.domain import Domain
from rasa.core.tracker_store import InMemoryTrackerStore
from tests.core.utilities import tracker_from_dialogue_file
from tests.core.conftest import TEST_DIALOGUES, EXAMPLE_DOMAINS


@pytest.mark.parametrize("filename", TEST_DIALOGUES)
def test_dialogue_serialisation(filename):
    dialogue_json = rasa.utils.io.read_file(filename)
    restored = json.loads(dialogue_json)
    tracker = tracker_from_dialogue_file(filename)
    en_de_coded = json.loads(jsonpickle.encode(tracker.as_dialogue()))
    assert restored == en_de_coded


@pytest.mark.parametrize("pair", zip(TEST_DIALOGUES, EXAMPLE_DOMAINS))
def test_inmemory_tracker_store(pair):
    filename, domainpath = pair
    domain = Domain.load(domainpath)
    tracker = tracker_from_dialogue_file(filename, domain)
    tracker_store = InMemoryTrackerStore(domain)
    tracker_store.save(tracker)
    restored = tracker_store.retrieve(tracker.sender_id)
    assert restored == tracker


def test_tracker_restaurant():
    domain = Domain.load("examples/restaurantbot/domain.yml")
    filename = "data/test_dialogues/restaurantbot.json"
    tracker = tracker_from_dialogue_file(filename, domain)
    assert tracker.get_slot("price") == "lo"
    assert tracker.get_slot("name") is None  # slot doesn't exist!
