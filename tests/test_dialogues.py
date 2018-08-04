from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import json

import io
import jsonpickle
import pytest

from rasa_core.domain import TemplateDomain
from rasa_core.tracker_store import InMemoryTrackerStore
from tests.utilities import tracker_from_dialogue_file


@pytest.mark.parametrize("filename", glob.glob('data/test_dialogues/*json'))
def test_dialogue_serialisation(filename):
    print("testing file: {0}".format(filename))
    with io.open(filename, "r") as f:
        dialogue_json = f.read()
    restored = json.loads(dialogue_json)
    tracker = tracker_from_dialogue_file(filename)
    en_de_coded = json.loads(jsonpickle.encode(tracker.as_dialogue()))
    assert restored == en_de_coded


@pytest.mark.parametrize("filename", glob.glob('data/test_dialogues/*json'))
def test_inmemory_tracker_store(filename):
    domain = TemplateDomain.load("data/test_domains/default.yml")
    tracker = tracker_from_dialogue_file(filename, domain)
    tracker_store = InMemoryTrackerStore(domain)
    tracker_store.save(tracker)
    restored = tracker_store.retrieve(tracker.sender_id)
    assert restored == tracker


def test_tracker_restaurant():
    domain = TemplateDomain.load("data/test_domains/default_with_slots.yml")
    filename = 'data/test_dialogues/enter_name.json'
    tracker = tracker_from_dialogue_file(filename, domain)
    assert tracker.get_slot("name") == "holger"
    assert tracker.get_slot("location") is None     # slot doesn't exist!
