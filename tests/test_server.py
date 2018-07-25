# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import signal
import uuid
from multiprocessing import Process

import pytest
from builtins import str
from freezegun import freeze_time
from pytest_localserver.http import WSGIServer

import rasa_core
from rasa_core import server, events
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.agent import Agent
from rasa_core.channels import UserMessage
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.events import (
    UserUttered, BotUttered, SlotSet, Event, ActionExecuted)
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.memoization import AugmentedMemoizationPolicy
from rasa_core.remote import RasaCoreClient, RemoteAgent
from rasa_core.utils import EndpointConfig
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
    SlotSet("cuisine", 34),
    SlotSet("cuisine", "34"),
    SlotSet("location", None),
    SlotSet("location", [34, "34", None]),
]


@pytest.fixture(scope="module")
def http_app(request, core_server):
    http_server = WSGIServer(application=core_server)
    http_server.start()

    request.addfinalizer(http_server.stop)
    return http_server.url


@pytest.fixture(scope="module")
def core_server(tmpdir_factory):
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent("data/test_domains/default.yml",
                  policies=[AugmentedMemoizationPolicy(max_history=3)])

    training_data = agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    agent.persist(model_path)

    return server.create_app(model_path,
                             interpreter=RegexInterpreter())


@pytest.fixture(scope="module")
def app(core_server):
    return core_server.test_client()


def test_root(app):
    response = app.get("http://dummy/")
    content = response.get_data(as_text=True)
    assert response.status_code == 200 and content.startswith("hello")


def test_version(app):
    response = app.get("http://dummy/version")
    content = response.get_json()
    assert response.status_code == 200
    assert content.get("version") == rasa_core.__version__


@freeze_time("2018-01-01")
def test_requesting_non_existent_tracker(app):
    response = app.get("http://dummy/conversations/madeupid/tracker")
    content = response.get_json()
    assert response.status_code == 200
    assert content["paused"] is False
    assert content["slots"] == {"location": None, "cuisine": None}
    assert content["sender_id"] == "madeupid"
    assert content["events"] == [{"event": "action",
                                  "name": "action_listen",
                                  "timestamp": 1514764800}]
    assert content["latest_message"] == {"text": None,
                                         "intent": {},
                                         "entities": []}


def test_continue_on_non_existent_conversation(app):
    data = json.dumps({"events": [], "executed_action": None})
    response = app.post("http://dummy/conversations/myid/continue",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200
    assert content["next_action"] == "action_listen"
    assert content["tracker"]["events"] is None
    assert content["tracker"]["paused"] is False
    assert content["tracker"]["sender_id"] == "myid"
    assert content["tracker"]["slots"] == {"location": None, "cuisine": None}
    assert content["tracker"]["latest_message"] == {"text": None,
                                                    "intent": {},
                                                    "entities": []}


def test_parse(app):
    data = json.dumps({"query": "/greet"})
    response = app.post("http://dummy/conversations/myid/parse",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200
    assert content["next_action"] == "utter_greet"
    assert content["tracker"]["events"] is None
    assert content["tracker"]["paused"] is False
    assert content["tracker"]["sender_id"] == "myid"
    assert content["tracker"]["slots"] == {"location": None, "cuisine": None}
    assert content["tracker"]["latest_message"]["text"] == "/greet"
    assert content["tracker"]["latest_message"]["intent"] == {
        "confidence": 1.0,
        "name": "greet"}


def test_continue(app):
    data = json.dumps({"query": "/greet"})
    response = app.post("http://dummy/conversations/myid/parse",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    data = json.dumps({"events": [], "executed_action": "utter_greet"})
    response = app.post("http://dummy/conversations/myid/continue",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

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
def test_pushing_events(app, event):
    cid = str(uuid.uuid1())
    conversation = "http://dummy/conversations/{}".format(cid)
    data = json.dumps({"query": "/greet"})
    response = app.post("{}/parse".format(conversation),
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    data = json.dumps({"events": [], "executed_action": "utter_greet"})
    response = app.post("{}/continue".format(conversation),
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    data = json.dumps([event.as_dict()])
    response = app.post("{}/tracker/events".format(conversation),
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    tracker_response = app.get("http://dummy/conversations/{}/tracker"
                               "".format(cid))
    tracker = tracker_response.get_json()
    assert tracker is not None
    assert len(tracker.get("events")) == 5

    evt = tracker.get("events")[4]
    assert Event.from_parameters(evt) == event


def test_put_tracker(app):
    data = json.dumps([event.as_dict() for event in test_events])
    response = app.put("http://dummy/conversations/pushtracker/tracker",
                       data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200
    assert len(content["events"]) == len(test_events)
    assert content["sender_id"] == "pushtracker"

    tracker_response = app.get("http://dummy/conversations/pushtracker/tracker")
    tracker = tracker_response.get_json()
    assert tracker is not None
    evts = tracker.get("events")
    assert events.deserialise_events(evts) == test_events


def test_list_conversations(app):
    data = json.dumps({"query": "/greet"})
    response = app.post("http://dummy/conversations/myid/parse",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    response = app.get("http://dummy/conversations")
    content = response.get_json()
    assert response.status_code == 200

    assert len(content) > 0
    assert "myid" in content


def test_remote_client(http_app, default_agent, tmpdir):
    model_path = tmpdir.join("persisted_model").strpath

    default_agent.persist(model_path)

    remote_agent = RemoteAgent.load(model_path,
                                    EndpointConfig(http_app))

    message = UserMessage("""/greet{"name":"Rasa"}""",
                          output_channel=CollectingOutputChannel())

    remote_agent.process_message(message)

    tracker = remote_agent.core_client.tracker_json("default")

    assert len(tracker.get("events")) == 6

    # listen
    assert tracker["events"][0]["name"] == "action_listen"
    # this should be the utterance
    assert tracker["events"][1]["text"] == """/greet{"name":"Rasa"}"""
    # set slot event
    assert tracker["events"][2]["value"] == "Rasa"
    # utter action
    assert tracker["events"][3]["name"] == "utter_greet"
    # this should be the bot utterance
    assert tracker["events"][4]["text"] == "hey there Rasa!"
    # listen
    assert tracker["events"][5]["name"] == "action_listen"


def test_remote_status(http_app):
    client = RasaCoreClient(EndpointConfig(http_app))

    status = client.status()

    assert status.get("version") == rasa_core.__version__


def test_remote_clients(http_app):
    client = RasaCoreClient(EndpointConfig(http_app))

    cid = str(uuid.uuid1())
    client.parse("/greet", cid)

    clients = client.clients()

    assert cid in clients


def test_remote_append_events(http_app):
    client = RasaCoreClient(EndpointConfig(http_app))

    cid = str(uuid.uuid1())

    client.append_events_to_tracker(cid, test_events[:2])

    tracker = client.tracker_json(cid)

    evts = tracker.get("events")
    expected = [ActionExecuted(ACTION_LISTEN_NAME)] + test_events[:2]
    assert events.deserialise_events(evts) == expected
