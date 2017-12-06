# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

import pytest
from treq.testing import StubTreq

import rasa_core
from rasa_core.agent import Agent
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.scoring_policy import ScoringPolicy
from rasa_core.server import RasaCoreServer


@pytest.fixture(scope="module")
def app(core_server):
    """This fixture makes use of the IResource interface of the
    Klein application to mock Rasa Core server."""
    return StubTreq(core_server.app.resource())


@pytest.fixture(scope="module")
def core_server(tmpdir_factory):
    training_data_file = 'examples/moodbot/data/stories.md'
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent("examples/moodbot/domain.yml",
                  policies=[ScoringPolicy()])

    agent.train(training_data_file, max_history=3)
    agent.persist(model_path)

    return RasaCoreServer(model_path, interpreter=RegexInterpreter())


@pytest.inlineCallbacks
def test_root(app):
    response = yield app.get("http://dummy_uri/")
    content = yield response.text()
    assert response.code == 200 and content.startswith("hello")


@pytest.inlineCallbacks
def test_version(app):
    response = yield app.get("http://dummy_uri/version")
    content = yield response.json()
    assert response.code == 200
    assert content.get("version") == rasa_core.__version__


@pytest.inlineCallbacks
def test_requesting_non_existent_tracker(app):
    response = yield app.get("http://dummy_uri/conversations/madeupid/tracker")
    content = yield response.json()
    assert response.code == 200
    assert content["paused"] is False
    assert content["slots"] == {}
    assert content["sender_id"] == "madeupid"
    assert content["events"] == [{"event": "action", "name": "action_listen"}]
    assert content["latest_message"] == {"text": None,
                                         "intent": {},
                                         "entities": []}


@pytest.inlineCallbacks
def test_continue_on_non_existent_conversation(app):
    data = json.dumps({"events": [], "executed_action": None})
    response = yield app.post("http://dummy_uri/conversations/myid/continue",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200
    assert content["next_action"] == "action_listen"
    assert content["tracker"]["events"] is None
    assert content["tracker"]["paused"] is False
    assert content["tracker"]["sender_id"] == "myid"
    assert content["tracker"]["slots"] == {}
    assert content["tracker"]["latest_message"] == {"text": None,
                                                    "intent": {},
                                                    "entities": []}


@pytest.inlineCallbacks
def test_parse(app):
    data = json.dumps({"query": "/greet"})
    response = yield app.post("http://dummy_uri/conversations/myid/parse",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200
    assert content["next_action"] == "utter_greet"
    assert content["tracker"]["events"] is None
    assert content["tracker"]["paused"] is False
    assert content["tracker"]["sender_id"] == "myid"
    assert content["tracker"]["slots"] == {}
    assert content["tracker"]["latest_message"]["text"] == "/greet"
    assert content["tracker"]["latest_message"]["intent"] == {
        "confidence": 1.0,
        "name": "greet"}


@pytest.inlineCallbacks
def test_continue(app):
    data = json.dumps({"query": "/greet"})
    response = yield app.post("http://dummy_uri/conversations/myid/parse",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200

    data = json.dumps({"events": [], "executed_action": "utter_greet"})
    response = yield app.post("http://dummy_uri/conversations/myid/continue",
                              data=data, content_type='application/json')
    content = yield response.json()
    assert response.code == 200

    assert content["next_action"] == "action_listen"
    assert content["tracker"]["events"] is None
    assert content["tracker"]["paused"] is False
    assert content["tracker"]["sender_id"] == "myid"
    assert content["tracker"]["slots"] == {}
    assert content["tracker"]["latest_message"]["text"] == "/greet"
    assert content["tracker"]["latest_message"]["intent"] == {
        "confidence": 1.0,
        "name": "greet"}


# @pytest.mark.parametrize("event", [
#     {"event": "user", "text": "/goodbye", "parse_data": {
#         "intent": {"confidence": 1.0, "name": "greet"},
#         "entities": []}},
#     {"event": "bot", "text": "Welcome!", "data": {"test": True}},
#     {"event": "topic", "topic": "Greet"},
#     {"event": "slot", "name": "my_slot", "value": 34},
#     {"event": "slot", "name": "my_slot", "value": "34"},
#     {"event": "slot", "name": "my_slot", "value": None},
#     {"event": "slot", "name": "my_slot", "value": [34, "34", None]}
# ])
# @pytest.inlineCallbacks
# def test_continue(app, event):
#     cid = "myid"
#     conversation = "http://dummy_uri/conversations/{}".format(cid)
#     data = json.dumps({"query": "/greet"})
#     response = yield app.post("{}/parse".format(conversation),
#                               data=data, content_type='application/json')
#     content = yield response.json()
#     assert response.code == 200
#
#     data = json.dumps({"events": [], "executed_action": "utter_greet"})
#     response = yield app.post("{}/continue".format(conversation),
#                               data=data, content_type='application/json')
#     content = yield response.json()
#     assert response.code == 200
#
#     data = json.dumps([event])
#     response = yield app.post("{}/tracker/events".format(conversation),
#                               data=data, content_type='application/json')
#     content = yield response.json()
#     assert response.code == 200
#
#     assert content