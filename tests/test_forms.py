from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.events import (
    ActionExecuted,
    BotUttered,
    SlotSet,
    UserUttered
)
from rasa_core.actions.forms import (
    BooleanFormField,
    EntityFormField,
    FormAction,
    FreeTextFormField
)
from rasa_core.domain import TemplateDomain
from rasa_core.tracker_store import InMemoryTrackerStore
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Dispatcher


class RestaurantAPI(object):
    def search(self, *args):
        return


class ActionSearchRestaurants(FormAction):

    RANDOMIZE = False

    @staticmethod
    def required_fields():
        return [
            EntityFormField("cuisine", "cuisine"),
            EntityFormField("number", "people"),
            BooleanFormField("vegetarian", "affirm", "deny")
        ]

    def name(self):
        return 'action_search_restaurants'

    def submit(self, dispatcher, tracker, domain):
        results = RestaurantAPI.search(
            tracker.get_slot("cuisine"),
            tracker.get_slot("people"),
            tracker.get_slot("vegetarian"))
        return [SlotSet("search_results": results)]


class ActionSearchPeople(FormAction):

    @staticmethod
    def required_fields():
        return [
            FreeTextFormField("person_name")
        ]

    def name(self):
        return 'action_search_people'

    def submit(self, dispatcher, tracker, domain):
        return []


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
    entities = [{"entity": "cuisine", "value": "chinese"}]
    tracker.update(
        UserUttered("",
                    intent={"name": "inform"},
                    entities=entities))

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
    print([(e.key, e.value) for e in events])
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)

    # same slot requested again
    assert events[0].key == "requested_slot"
    assert events[0].value == "cuisine"


def test_restaurant_form_unhappy_2():
    domain = TemplateDomain.load("data/test_domains/restaurant_form.yml")
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-restaurant"
    dispatcher = Dispatcher(sender_id, out, domain)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # second user utterance
    entities = [
        {"entity": "cuisine", "value": "chinese"},
        {"entity": "people", "value": 8}]

    tracker.update(
        UserUttered("",
                    intent={"name": "inform"},
                    entities=entities))

    # store all entities as slots
    for e in domain.slots_for_entities(entities):
        tracker.update(e)
    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)

    cuisine = tracker.get_slot("cuisine")
    people = tracker.get_slot("people")
    assert cuisine == "chinese"
    assert people == 8

    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "requested_slot"
    assert events[0].value == "vegetarian"
    tracker.update(events[0])

    # second user utterance does not provide what's asked
    tracker.update(
        UserUttered("",
                    intent={"name": "random"}))

    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    s = events[0].as_story_string()
    assert len(events) == 1
    assert events[0].key == "requested_slot"
    assert events[0].value == "vegetarian"


def test_people_form():
    domain = TemplateDomain.load("data/test_domains/people_form.yml")
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-people"
    dispatcher = Dispatcher(sender_id, out, domain)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # first user utterance
    tracker.update(UserUttered("", intent={"name": "inform"}))
    events = ActionSearchPeople().run(dispatcher, tracker, domain)
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

    events = ActionSearchPeople().run(dispatcher, tracker, domain)
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)

    assert events[0].key == "person_name"
    assert events[0].value == name
