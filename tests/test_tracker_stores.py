from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels import UserMessage
from rasa_core.domain import TemplateDomain
from rasa_core.events import SlotSet
from rasa_core.tracker_store import InMemoryTrackerStore

domain = TemplateDomain.load("data/test_domains/default_with_topic.yml")


def test_get_or_create():
    slot_key = 'location'
    slot_val = 'Easter Island'
    store = InMemoryTrackerStore(domain)

    tracker = store.get_or_create_tracker(UserMessage.DEFAULT_SENDER)
    ev = SlotSet(slot_key, slot_val)
    tracker.update(ev)
    assert tracker.get_slot(slot_key) == slot_val

    store.save(tracker)

    again = store.get_or_create_tracker(UserMessage.DEFAULT_SENDER)
    assert again.get_slot(slot_key) == slot_val
