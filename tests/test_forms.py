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
from rasa_core.nlg import TemplatedNaturalLanguageGenerator
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
        results = RestaurantAPI().search(
            tracker.get_slot("cuisine"),
            tracker.get_slot("people"),
            tracker.get_slot("vegetarian"))
        return [SlotSet("search_results", results)]


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


class ActionSearchTravel(FormAction):

    RANDOMIZE = False

    @staticmethod
    def required_fields():
        return [
            EntityFormField("GPE", "GPE_origin"),
            EntityFormField("GPE", "GPE_destination")
        ]

    def name(self):
        return 'action_search_travel'

    def submit(self, dispatcher, tracker, domain):
        return []


class ActionSearchQuery(FormAction):

    RANDOMIZE = False

    @staticmethod
    def required_fields():
        return [
            FreeTextFormField("username"),
            FreeTextFormField("query")
        ]

    def name(self):
        return 'action_perform_query'

    def submit(self, dispatcher, tracker, domain):
        return []


def test_restaurant_form():
    domain = TemplateDomain.load("data/test_domains/restaurant_form.yml")
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-restaurant"
    dispatcher = Dispatcher(sender_id, out, nlg)
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
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-restaurant"
    dispatcher = Dispatcher(sender_id, out, nlg)
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
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-restaurant"
    dispatcher = Dispatcher(sender_id, out, nlg)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # first user utterance
    entities = [
        {"entity": "cuisine", "value": "chinese"},
        {"entity": "number", "value": 8}]

    tracker.update(
        UserUttered("",
                    intent={"name": "inform"},
                    entities=entities))

    # store all entities as slots
    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)

    for e in events:
        tracker.update(e)

    cuisine = tracker.get_slot("cuisine")
    people = tracker.get_slot("people")
    assert cuisine == "chinese"
    assert people == 8

    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    assert len(events) == 3
    assert isinstance(events[0], SlotSet)
    assert events[2].key == "requested_slot"
    assert events[2].value == "vegetarian"
    tracker.update(events[2])

    # second user utterance does not provide what's asked
    tracker.update(
        UserUttered("",
                    intent={"name": "random"}))

    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    s = events[0].as_story_string()
    assert len(events) == 1
    assert events[0].key == "requested_slot"
    assert events[0].value == "vegetarian"


def test_restaurant_form_skipahead():
    domain = TemplateDomain.load("data/test_domains/restaurant_form.yml")
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-restaurant"
    dispatcher = Dispatcher(sender_id, out, nlg)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # first user utterance
    entities = [{"entity": "cuisine", "value": "chinese"},
                {"entity": "number", "value": 8}]
    tracker.update(UserUttered("",
                               intent={"name": "inform"},
                               entities=entities))

    events = ActionSearchRestaurants().run(dispatcher, tracker, domain)
    s = events[0].as_story_string()
    print(events[0].as_story_string())
    print(events[1].as_story_string())
    assert len(events) == 3
    assert events[2].key == "requested_slot"
    assert events[2].value == "vegetarian"


def test_people_form():
    domain = TemplateDomain.load("data/test_domains/people_form.yml")
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-people"
    dispatcher = Dispatcher(sender_id, out, nlg)
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


def test_travel_form():
    domain = TemplateDomain.load("data/test_domains/travel_form.yml")
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-travel"
    dispatcher = Dispatcher(sender_id, out, nlg)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # first user utterance
    tracker.update(UserUttered("", intent={"name": "inform"}))
    events = ActionSearchTravel().run(dispatcher, tracker, domain)
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "requested_slot"
    assert events[0].value == "GPE_origin"
    tracker.update(events[0])

    # second user utterance
    entities = [{"entity": "GPE", "value": "Berlin"}]
    tracker.update(UserUttered("",
                               intent={"name": "inform"},
                               entities=entities))
    events = ActionSearchTravel().run(dispatcher, tracker, domain)
    for e in events:
        print(e.as_story_string())
    assert len(events) == 2
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "GPE_origin"
    assert events[0].value == "Berlin"
    assert events[1].key == "requested_slot"
    assert events[1].value == "GPE_destination"


def test_query_form_set_username_directly():
    domain = TemplateDomain.load("data/test_domains/query_form.yml")
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-form"
    dispatcher = Dispatcher(sender_id, out, nlg)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # pre-fill username slot
    username = "Monty"
    tracker.update(SlotSet('username', username))

    # first user utterance
    tracker.update(UserUttered("", intent={"name": "inform"}))
    events = ActionSearchQuery().run(dispatcher, tracker, domain)
    last_message = dispatcher.latest_bot_messages[-1]
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "requested_slot"
    assert events[0].value == "query"
    assert username in last_message.text


def test_query_form_set_username_in_form():
    domain = TemplateDomain.load("data/test_domains/query_form.yml")
    nlg = TemplatedNaturalLanguageGenerator(domain.templates)
    tracker_store = InMemoryTrackerStore(domain)
    out = CollectingOutputChannel()
    sender_id = "test-form"
    dispatcher = Dispatcher(sender_id, out, nlg)
    tracker = tracker_store.get_or_create_tracker(sender_id)

    # first user utterance
    tracker.update(UserUttered("", intent={"name": "inform"}))
    events = ActionSearchQuery().run(dispatcher, tracker, domain)
    last_message = dispatcher.latest_bot_messages[-1]
    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "requested_slot"
    assert events[0].value == "username"
    assert last_message.text == 'what is your name?'
    tracker.update(events[0])

    # second user utterance
    username = 'Monty'
    tracker.update(UserUttered(username, intent={"name": "inform"}))
    events = ActionSearchQuery().run(dispatcher, tracker, domain)
    last_message = dispatcher.latest_bot_messages[-1]
    assert len(events) == 2
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "username"
    assert events[0].value == username
    assert events[1].key == "requested_slot"
    assert events[1].value == "query"
    assert username in last_message.text
