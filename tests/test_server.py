# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import str

import json
import pytest
import uuid
from freezegun import freeze_time

import rasa_core
from rasa_core import events, constants
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.domain import Domain
from rasa_core.events import (
    UserUttered, BotUttered, SlotSet, Event, ActionExecuted)
from rasa_core.remote import RasaCoreClient
from rasa_core.utils import EndpointConfig

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
def app(core_server):
    return core_server.test_client()


@pytest.fixture(scope="module")
def secured_app(core_server_secured):
    return core_server_secured.test_client()


def test_root(app):
    response = app.get("http://dummy/")
    content = response.get_data(as_text=True)
    assert response.status_code == 200 and content.startswith("hello")


def test_root_secured(secured_app):
    response = secured_app.get("http://dummy/")
    content = response.get_data(as_text=True)
    assert response.status_code == 200 and content.startswith("hello")


def test_version(app):
    response = app.get("http://dummy/version")
    content = response.get_json()
    assert response.status_code == 200
    assert content.get("version") == rasa_core.__version__
    assert (content.get(
        "minimum_compatible_version") == constants.MINIMUM_COMPATIBLE_VERSION)


def test_status(app):
    response = app.get("http://dummy/status")
    content = response.get_json()
    assert response.status_code == 200
    assert content.get("is_ready")
    assert content.get("model_fingerprint") is not None


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
                                  "policy": None,
                                  "confidence": None,
                                  "timestamp": 1514764800}]
    assert content["latest_message"] == {"text": None,
                                         "intent": {},
                                         "entities": []}


def test_respond(app):
    data = json.dumps({"query": "/greet"})
    response = app.post("http://dummy/conversations/myid/respond",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200
    assert content == [{'text': 'hey there!', 'recipient_id': 'myid'}]


@pytest.mark.parametrize("event", test_events)
def test_pushing_event(app, event):
    cid = str(uuid.uuid1())
    conversation = "http://dummy/conversations/{}".format(cid)
    data = json.dumps({"query": "/greet"})
    response = app.post("{}/respond".format(conversation),
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    data = json.dumps(event.as_dict())
    response = app.post("{}/tracker/events".format(conversation),
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    tracker_response = app.get("http://dummy/conversations/{}/tracker"
                               "".format(cid))
    tracker = tracker_response.get_json()
    assert tracker is not None
    assert len(tracker.get("events")) == 6

    evt = tracker.get("events")[5]
    assert Event.from_parameters(evt) == event


def test_put_tracker(app):
    data = json.dumps([event.as_dict() for event in test_events])
    response = app.put("http://dummy/conversations/pushtracker/tracker/events",
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
    response = app.post("http://dummy/conversations/myid/respond",
                        data=data, content_type='application/json')
    content = response.get_json()
    assert response.status_code == 200

    response = app.get("http://dummy/conversations")
    content = response.get_json()
    assert response.status_code == 200

    assert len(content) > 0
    assert "myid" in content


def test_remote_status(http_app):
    client = RasaCoreClient(EndpointConfig(http_app))

    status = client.status()

    assert status.get("version") == rasa_core.__version__


def test_remote_clients(http_app):
    client = RasaCoreClient(EndpointConfig(http_app))

    cid = str(uuid.uuid1())
    client.respond("/greet", cid)

    clients = client.clients()

    assert cid in clients


@pytest.mark.parametrize("event", test_events)
def test_remote_append_events(http_app, event):
    client = RasaCoreClient(EndpointConfig(http_app))

    cid = str(uuid.uuid1())

    client.append_event_to_tracker(cid, event)

    tracker = client.tracker_json(cid)

    evts = tracker.get("events")
    expected = [ActionExecuted(ACTION_LISTEN_NAME), event]
    assert events.deserialise_events(evts) == expected


def test_predict(http_app, app):
    client = RasaCoreClient(EndpointConfig(http_app))
    cid = str(uuid.uuid1())
    for event in test_events[:2]:
        client.append_event_to_tracker(cid, event)
    out = app.get('/domain', headers={'Accept': 'yml'})
    domain = Domain.from_yaml(out.get_data())
    tracker = client.tracker(cid, domain)
    event_dicts = [ev.as_dict() for ev in tracker.applied_events()]
    response = app.post('/predict',
                        json=event_dicts)
    assert response.status_code == 200


def test_list_conversations_with_jwt(secured_app):
    # token generated with secret "core" and algorithm HS256
    # on https://jwt.io/

    # {"username": "testadmin", "role": "admin"}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                         "1c2VybmFtZSI6InRlc3RhZG1pbiIsInJvbGUiOiJhZG1pbi"
                         "J9.3gp-0pEEUJpU_NoR76lVYMrW86Aedx_QULKUcw3ODbo"
    }
    response = secured_app.get("/conversations",
                               headers=jwt_header)
    assert response.status_code == 200

    # {"username": "testuser", "role": "user"}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                         "1c2VybmFtZSI6InRlc3R1c2VyIiwicm9sZSI6InVzZXIifQ"
                         ".X4wN0sLRW0Urd9E-ProsCK_IQHjuNZ5SJwm4RXiX6fQ"
    }
    response = secured_app.get("/conversations",
                               headers=jwt_header)
    assert response.status_code == 403


def test_get_tracker_with_jwt(secured_app):
    # token generated with secret "core" and algorithm HS256
    # on https://jwt.io/

    # {"username": "testadmin", "role": "admin"}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                         "1c2VybmFtZSI6InRlc3RhZG1pbiIsInJvbGUiOiJhZG1pbi"
                         "J9.3gp-0pEEUJpU_NoR76lVYMrW86Aedx_QULKUcw3ODbo"
    }
    response = secured_app.get("/conversations/testadmin/tracker",
                               headers=jwt_header)
    assert response.status_code == 200

    response = secured_app.get("/conversations/testuser/tracker",
                               headers=jwt_header)
    assert response.status_code == 200

    # {"username": "testuser", "role": "user"}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                         "1c2VybmFtZSI6InRlc3R1c2VyIiwicm9sZSI6InVzZXIifQ"
                         ".X4wN0sLRW0Urd9E-ProsCK_IQHjuNZ5SJwm4RXiX6fQ"
    }
    response = secured_app.get("/conversations/testadmin/tracker",
                               headers=jwt_header)
    assert response.status_code == 403

    response = secured_app.get("/conversations/testuser/tracker",
                               headers=jwt_header)
    assert response.status_code == 200


def test_list_conversations_with_token(secured_app):
    response = secured_app.get("/conversations?token=rasa")
    assert response.status_code == 200


def test_list_conversations_with_wrong_token(secured_app):
    response = secured_app.get("/conversations?token=Rasa")
    assert response.status_code == 401


def test_list_conversations_without_auth(secured_app):
    response = secured_app.get("/conversations")
    assert response.status_code == 401


def test_list_conversations_with_wrong_jwt(secured_app):
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                         "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIi"
                         "wiaWF0IjoxNTE2MjM5MDIyfQ.qdrr2_a7Sd80gmCWjnDomO"
                         "Gl8eZFVfKXA6jhncgRn-I"
    }
    response = secured_app.get("/conversations",
                               headers=jwt_header)
    assert response.status_code == 422
