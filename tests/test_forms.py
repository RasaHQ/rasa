from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.events import ActionExecuted, UserUttered, BotUttered, SlotSet
from rasa_core.actions.forms import FormAction, EntityFormField, FreeTextFormField
from rasa_core.domain import TemplateDomain
from rasa_core.tracker_store import InMemoryTrackerStore
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Dispatcher


class ActionSearchRestaurants(FormAction):

    REQUIRED_FIELDS = [
        EntityFormField("cuisine", "cuisine"),
        EntityFormField("number", "people")
    ]
    RANDOMIZE = False

    def name(self):
        return 'action_search_restaurants'

    def submit(self, dispatcher, tracker, domain):
        pass


class ActionSearchPeople(FormAction):

    REQUIRED_FIELDS = [
	FreeTextFormField("person_name")
    ]

    def name(self):
        return 'action_search_people'

    def submit(self, dispatcher, tracker, domain):
        pass



def test_restaurant_form():
    domain = TemplateDomain.load("data/test_domains/restaurant_form.yml")
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-restaurant"
    dispatcher = Dispatcher(sender_id, out, domain)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # first user utterance
    tracker.update(UserUttered("", intent={"name": "inform"}))
    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "requested_slot"
    assert events[0].value == "cuisine"
    tracker.update(events[0])

    # second user utterance
    tracker.update(
        UserUttered("",
                    intent={"name": "inform"},
                    entities=[{"entity": "cuisine", "value": "chinese"}]))

    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    assert len(events) == 2
    assert isinstance(events[0], SlotSet)
    assert isinstance(events[1], SlotSet)

    assert events[0].key == "cuisine"
    assert events[0].value == "chinese"

    assert events[1].key == "requested_slot"
    assert events[1].value == "people"


def test_restaurant_form_unhappy_1():
    domain = TemplateDomain.load("data/test_domains/restaurant_form.yml")
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-restaurant"
    dispatcher = Dispatcher(sender_id, out, domain)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # first user utterance
    tracker.update(UserUttered("", intent={"name": "inform"}))
    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "requested_slot"
    assert events[0].value == "cuisine"
    tracker.update(events[0])

    # second user utterance does not provide what's asked
    tracker.update(
        UserUttered("",
                    intent={"name": "inform"}))

    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    assert len(events) == 2
    assert isinstance(events[0], SlotSet)

    # same slot requested again
    assert events[0].key == "requested_slot"
    assert events[0].value == "cuisine"


def test_people_form():
    domain = TemplateDomain.load("data/test_domains/people_form.yml")
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-people"
    dispatcher = Dispatcher(sender_id, out, domain)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # first user utterance
    tracker.update(UserUttered("", intent={"name": "inform"}))
    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "requested_slot"
    assert events[0].value == "person_name"
    tracker.update(events[0])

    # second user utterance
    name = "Rasa Due"
    tracker.update(
        UserUttered(name,
                    intent={"name": "inform"}))

    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)

    assert events[0].key == "person_name"
    assert events[0].value == name
