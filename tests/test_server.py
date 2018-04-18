# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import uuid
from builtins import str

import pytest
from freezegun import freeze_time
from treq.testing import StubTreq

import rasa_core
from rasa_core.agent import Agent
from rasa_core.events import UserUttered, BotUttered, SlotSet, TopicSet, Event
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.server import RasaCoreServer
from tests.conftest import DEFAULT_STORIES_FILE

# a couple of event instances that we can use for testing
test_events = [
    Event.from_parameters({"event": UserUttered.type_name,
                           "text": "/goodbye",
                           "parse_data": {
                               "intent": {
                                   "confidence": 1.0, "name": "greet"},
                               "entities": []}
                           }),
    BotUttered("Welcome!", {"test": True}),
    TopicSet("question"),
    SlotSet("cuisine", 34),
    SlotSet("cuisine", "34"),
    SlotSet("location", None),
    SlotSet("location", [34, "34", None]),
]


@pytest.fixture(scope="module")
def app(core_server):
    """This fixture makes use of the IResource interface of the
    Klein application to mock Rasa Core server."""
    return StubTreq(core_server.app.resource())


@pytest.fixture(scope="module")
def core_server(tmpdir_factory):
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent("data/test_domains/default_with_topic.yml",
                  policies=[MemoizationPolicy()])

    agent.train(DEFAULT_STORIES_FILE, max_history=3)
    agent.persist(model_path)

    return RasaCoreServer(model_path, interpreter=RegexInterpreter())


@pytest.inlineCallbacks
def test_root(app):
    response = yield app.get("http://dummy/")
    content = yield response.text()
    assert response.code == 200 and content.startswith("hello")


@pytest.inlineCallbacks
def test_version(app):
    response = yield app.get("http://dummy/version")
    content = yield response.json()
    assert response.code == 200
    assert content.get("version") == rasa_core.__version__


@freeze_time("2018-01-01")
@pytest.inlineCallbacks
def test_requesting_non_existent_tracker(app):
    response = yield app.get("http://dummy/conversations/madeupid/tracker")
    content = yield response.json()
    assert response.code == 200
    assert content["paused"] is False
    assert content["slots"] == {"location": None, "cuisine": None}
    assert content["sender_id"] == "madeupid"
    assert content["events"] == [{"event": "action",
                                  "name": "action_listen",
                                  "timestamp": 1514764800}]
    assert content["latest_message"] == {"text": None,
                                         "intent": {},
                                         "entities": []}


@pytest.inlineCallbacks
def test_continue_on_non_existent_conversation(app):
    data = json.dumps({"events": [], "executed_action": None})
    response = yield app.post("http://dummy/conversations/myid/continue",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200
    assert content["next_action"] == "action_listen"
    assert content["tracker"]["events"] is None
    assert content["tracker"]["paused"] is False
    assert content["tracker"]["sender_id"] == "myid"
    assert content["tracker"]["slots"] == {"location": None, "cuisine": None}
    assert content["tracker"]["latest_message"] == {"text": None,
                                                    "intent": {},
                                                    "entities": []}


@pytest.inlineCallbacks
def test_parse(app):
    data = json.dumps({"query": "/greet"})
    response = yield app.post("http://dummy/conversations/myid/parse",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200
    assert content["next_action"] == "utter_greet"
    assert content["tracker"]["events"] is None
    assert content["tracker"]["paused"] is False
    assert content["tracker"]["sender_id"] == "myid"
    assert content["tracker"]["slots"] == {"location": None, "cuisine": None}
    assert content["tracker"]["latest_message"]["text"] == "/greet"
    assert content["tracker"]["latest_message"]["intent"] == {
        "confidence": 1.0,
        "name": "greet"}


@pytest.inlineCallbacks
def test_continue(app):
    data = json.dumps({"query": "/greet"})
    response = yield app.post("http://dummy/conversations/myid/parse",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200

    data = json.dumps({"events": [], "executed_action": "utter_greet"})
    response = yield app.post("http://dummy/conversations/myid/continue",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200

    assert content["next_action"] == "action_listen"
    assert content["tracker"]["events"] is None
    assert content["tracker"]["paused"] is False
    assert content["tracker"]["sender_id"] == "myid"
    assert content["tracker"]["slots"] == {"location": None, "cuisine": None}
    assert content["tracker"]["latest_message"]["text"] == "/greet"
    assert content["tracker"]["latest_message"]["intent"] == {
        "confidence": 1.0,
        "name": "greet"}


@pytest.mark.parametrize("event", test_events)
@pytest.inlineCallbacks
def test_pushing_events(core_server, app, event):
    cid = str(uuid.uuid1())
    conversation = "http://dummy/conversations/{}".format(cid)
    data = json.dumps({"query": "/greet"})
    response = yield app.post("{}/parse".format(conversation),
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200

    data = json.dumps({"events": [], "executed_action": "utter_greet"})
    response = yield app.post("{}/continue".format(conversation),
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200

    data = json.dumps([event.as_dict()])
    response = yield app.post("{}/tracker/events".format(conversation),
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200

    tracker = core_server.agent.tracker_store.retrieve(cid)
    assert tracker is not None
    assert len(tracker.events) == 5
    assert tracker.events[4] == event


@pytest.inlineCallbacks
def test_put_tracker(core_server, app):
    data = json.dumps([event.as_dict() for event in test_events])
    response = yield app.put("http://dummy/conversations/pushtracker/tracker",
                             data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200
    assert len(content["events"]) == len(test_events)
    assert content["sender_id"] == "pushtracker"

    tracker = core_server.agent.tracker_store.retrieve("pushtracker")
    assert tracker is not None
    assert len(tracker.events) == len(test_events)
    assert list(tracker.events) == test_events


@pytest.inlineCallbacks
def test_list_conversations(app):
    data = json.dumps({"query": "/greet"})
    response = yield app.post("http://dummy/conversations/myid/parse",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200

    response = yield app.get("http://dummy/conversations")
    content = yield response.json()
    assert response.code == 200

    assert len(content) > 0
    assert "myid" in content
