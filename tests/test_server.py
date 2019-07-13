# -*- coding: utf-8 -*-
import json
import os
import tempfile
import uuid
from aioresponses import aioresponses

import pytest
from freezegun import freeze_time

import rasa
import rasa.constants
from rasa.core import events, utils
from rasa.core.events import Event, UserUttered, SlotSet, BotUttered
from rasa.model import unpack_model
from rasa.utils.endpoints import EndpointConfig
from tests.nlu.utilities import ResponseTest


# a couple of event instances that we can use for testing
test_events = [
    Event.from_parameters(
        {
            "event": UserUttered.type_name,
            "text": "/goodbye",
            "parse_data": {
                "intent": {"confidence": 1.0, "name": "greet"},
                "entities": [],
            },
        }
    ),
    BotUttered("Welcome!", {"test": True}),
    SlotSet("cuisine", 34),
    SlotSet("cuisine", "34"),
    SlotSet("location", None),
    SlotSet("location", [34, "34", None]),
]


@pytest.fixture
def rasa_app(rasa_server):
    return rasa_server.test_client


@pytest.fixture
def rasa_app_nlu(rasa_nlu_server):
    return rasa_nlu_server.test_client


@pytest.fixture
def rasa_app_core(rasa_core_server):
    return rasa_core_server.test_client


@pytest.fixture
def rasa_secured_app(rasa_server_secured):
    return rasa_server_secured.test_client


def test_root(rasa_app):
    _, response = rasa_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_root_secured(rasa_secured_app):
    _, response = rasa_secured_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_version(rasa_app):
    _, response = rasa_app.get("/version")
    content = response.json
    assert response.status == 200
    assert content.get("version") == rasa.__version__
    assert (
        content.get("minimum_compatible_version")
        == rasa.constants.MINIMUM_COMPATIBLE_VERSION
    )


def test_status(rasa_app):
    _, response = rasa_app.get("/status")
    assert response.status == 200
    assert "fingerprint" in response.json
    assert "model_file" in response.json


def test_status_secured(rasa_secured_app):
    _, response = rasa_secured_app.get("/status")
    assert response.status == 401


def test_status_not_ready_agent(rasa_app_nlu):
    _, response = rasa_app_nlu.get("/status")
    assert response.status == 409


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={"text": "hello ńöñàśçií"},
        ),
    ],
)
def test_parse(rasa_app, response_test):
    _, response = rasa_app.post(response_test.endpoint, json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text"])
    assert rjs["entities"] == response_test.expected_response["entities"]
    assert rjs["text"] == response_test.expected_response["text"]
    assert rjs["intent"] == response_test.expected_response["intent"]


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/model/parse?emulation_mode=wit",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse?emulation_mode=dialogflow",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse?emulation_mode=luis",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={"text": "hello ńöñàśçií"},
        ),
    ],
)
def test_parse_with_different_emulation_mode(rasa_app, response_test):
    _, response = rasa_app.post(response_test.endpoint, json=response_test.payload)
    assert response.status == 200


def test_parse_without_nlu_model(rasa_app_core):
    _, response = rasa_app_core.post("/model/parse", json={"text": "hello"})
    assert response.status == 200

    rjs = response.json
    assert all(prop in rjs for prop in ["entities", "intent", "text"])


def test_parse_on_invalid_emulation_mode(rasa_app_nlu):
    _, response = rasa_app_nlu.post(
        "/model/parse?emulation_mode=ANYTHING", json={"text": "hello"}
    )
    assert response.status == 400


def test_train_stack_success(
    rasa_app,
    default_domain_path,
    default_stories_file,
    default_stack_config,
    default_nlu_data,
):
    domain_file = open(default_domain_path)
    config_file = open(default_stack_config)
    stories_file = open(default_stories_file)
    nlu_file = open(default_nlu_data)

    payload = dict(
        domain=domain_file.read(),
        config=config_file.read(),
        stories=stories_file.read(),
        nlu=nlu_file.read(),
    )

    domain_file.close()
    config_file.close()
    stories_file.close()
    nlu_file.close()

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 200

    assert response.headers["filename"] is not None

    # save model to temporary file
    tempdir = tempfile.mkdtemp()
    model_path = os.path.join(tempdir, "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    # unpack model and ensure fingerprint is present
    model_path = unpack_model(model_path)
    assert os.path.exists(os.path.join(model_path, "fingerprint.json"))


def test_train_nlu_success(
    rasa_app, default_stack_config, default_nlu_data, default_domain_path
):
    domain_file = open(default_domain_path)
    config_file = open(default_stack_config)
    nlu_file = open(default_nlu_data)

    payload = dict(
        domain=domain_file.read(), config=config_file.read(), nlu=nlu_file.read()
    )

    config_file.close()
    nlu_file.close()

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 200

    # save model to temporary file
    tempdir = tempfile.mkdtemp()
    model_path = os.path.join(tempdir, "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    # unpack model and ensure fingerprint is present
    model_path = unpack_model(model_path)
    assert os.path.exists(os.path.join(model_path, "fingerprint.json"))


def test_train_core_success(
    rasa_app, default_stack_config, default_stories_file, default_domain_path
):
    domain_file = open(default_domain_path)
    config_file = open(default_stack_config)
    core_file = open(default_stories_file)

    payload = dict(
        domain=domain_file.read(), config=config_file.read(), nlu=core_file.read()
    )

    config_file.close()
    core_file.close()

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 200

    # save model to temporary file
    tempdir = tempfile.mkdtemp()
    model_path = os.path.join(tempdir, "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    # unpack model and ensure fingerprint is present
    model_path = unpack_model(model_path)
    assert os.path.exists(os.path.join(model_path, "fingerprint.json"))


def test_train_missing_config(rasa_app):
    payload = dict(domain="domain data", config=None)

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 400


def test_train_missing_training_data(rasa_app):
    payload = dict(domain="domain data", config="config data")

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 400


def test_train_internal_error(rasa_app):
    payload = dict(domain="domain data", config="config data", nlu="nlu data")

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 500


def test_evaluate_stories(rasa_app, default_stories_file):
    with open(default_stories_file, "r") as f:
        stories = f.read()

    _, response = rasa_app.post("/model/test/stories", data=stories)

    assert response.status == 200

    js = response.json
    assert set(js.keys()) == {
        "report",
        "precision",
        "f1",
        "accuracy",
        "actions",
        "in_training_data_fraction",
        "is_end_to_end_evaluation",
    }
    assert not js["is_end_to_end_evaluation"]
    assert set(js["actions"][0].keys()) == {
        "action",
        "predicted",
        "confidence",
        "policy",
    }


def test_evaluate_stories_not_ready_agent(rasa_app_nlu, default_stories_file):
    with open(default_stories_file, "r") as f:
        stories = f.read()

    _, response = rasa_app_nlu.post("/model/test/stories", data=stories)

    assert response.status == 409


def test_evaluate_stories_end_to_end(rasa_app, end_to_end_story_file):
    with open(end_to_end_story_file, "r") as f:
        stories = f.read()

    _, response = rasa_app.post("/model/test/stories?e2e=true", data=stories)

    assert response.status == 200
    js = response.json
    assert set(js.keys()) == {
        "report",
        "precision",
        "f1",
        "accuracy",
        "actions",
        "in_training_data_fraction",
        "is_end_to_end_evaluation",
    }
    assert js["is_end_to_end_evaluation"]
    assert set(js["actions"][0].keys()) == {
        "action",
        "predicted",
        "confidence",
        "policy",
    }


def test_evaluate_intent(rasa_app, default_nlu_data):
    with open(default_nlu_data, "r") as f:
        nlu_data = f.read()

    _, response = rasa_app.post("/model/test/intents", data=nlu_data)

    assert response.status == 200
    assert set(response.json.keys()) == {"intent_evaluation", "entity_evaluation"}


def test_evaluate_intent_on_just_nlu_model(rasa_app_nlu, default_nlu_data):
    with open(default_nlu_data, "r") as f:
        nlu_data = f.read()

    _, response = rasa_app_nlu.post("/model/test/intents", data=nlu_data)

    assert response.status == 200
    assert set(response.json.keys()) == {"intent_evaluation", "entity_evaluation"}


def test_evaluate_intent_with_query_param(
    rasa_app, trained_nlu_model, default_nlu_data
):
    _, response = rasa_app.get("/status")
    previous_model_file = response.json["model_file"]

    with open(default_nlu_data, "r") as f:
        nlu_data = f.read()

    _, response = rasa_app.post(
        "/model/test/intents?model={}".format(trained_nlu_model), data=nlu_data
    )

    assert response.status == 200
    assert set(response.json.keys()) == {"intent_evaluation", "entity_evaluation"}

    _, response = rasa_app.get("/status")
    assert previous_model_file == response.json["model_file"]


def test_predict(rasa_app):
    data = json.dumps(
        {
            "Events": {
                "value": [
                    {"event": "action", "name": "action_listen"},
                    {
                        "event": "user",
                        "text": "hello",
                        "parse_data": {
                            "entities": [],
                            "intent": {"confidence": 0.57, "name": "greet"},
                            "text": "hello",
                        },
                    },
                ]
            }
        }
    )
    _, response = rasa_app.post(
        "/model/predict", data=data, headers={"Content-Type": "application/json"}
    )
    content = response.json
    assert response.status == 200
    assert "scores" in content
    assert "tracker" in content
    assert "policy" in content


def test_retrieve_tacker_not_ready_agent(rasa_app_nlu):
    _, response = rasa_app_nlu.get("/conversations/test/tracker")
    assert response.status == 409


@freeze_time("2018-01-01")
def test_requesting_non_existent_tracker(rasa_app):
    _, response = rasa_app.get("/conversations/madeupid/tracker")
    content = response.json
    assert response.status == 200
    assert content["paused"] is False
    assert content["slots"] == {"location": None, "cuisine": None}
    assert content["sender_id"] == "madeupid"
    assert content["events"] == [
        {
            "event": "action",
            "name": "action_listen",
            "policy": None,
            "confidence": None,
            "timestamp": 1514764800,
        }
    ]
    assert content["latest_message"] == {"text": None, "intent": {}, "entities": []}


@pytest.mark.parametrize("event", test_events)
def test_pushing_event(rasa_app, event):
    cid = str(uuid.uuid1())
    conversation = "/conversations/{}".format(cid)

    data = json.dumps(event.as_dict())
    _, response = rasa_app.post(
        "{}/tracker/events".format(conversation),
        data=data,
        headers={"Content-Type": "application/json"},
    )
    assert response.json is not None
    assert response.status == 200

    _, tracker_response = rasa_app.get("/conversations/{}/tracker".format(cid))
    tracker = tracker_response.json
    assert tracker is not None
    assert len(tracker.get("events")) == 2

    evt = tracker.get("events")[1]
    assert Event.from_parameters(evt) == event


def test_push_multiple_events(rasa_app):
    cid = str(uuid.uuid1())
    conversation = "/conversations/{}".format(cid)

    events = [e.as_dict() for e in test_events]
    data = json.dumps(events)
    _, response = rasa_app.post(
        "{}/tracker/events".format(conversation),
        data=data,
        headers={"Content-Type": "application/json"},
    )
    assert response.json is not None
    assert response.status == 200

    _, tracker_response = rasa_app.get("/conversations/{}/tracker".format(cid))
    tracker = tracker_response.json
    assert tracker is not None

    # there is also an `ACTION_LISTEN` event at the start
    assert len(tracker.get("events")) == len(test_events) + 1
    assert tracker.get("events")[1:] == events


def test_put_tracker(rasa_app):
    data = json.dumps([event.as_dict() for event in test_events])
    _, response = rasa_app.put(
        "/conversations/pushtracker/tracker/events",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    content = response.json
    assert response.status == 200
    assert len(content["events"]) == len(test_events)
    assert content["sender_id"] == "pushtracker"

    _, tracker_response = rasa_app.get("/conversations/pushtracker/tracker")
    tracker = tracker_response.json
    assert tracker is not None
    evts = tracker.get("events")
    assert events.deserialise_events(evts) == test_events


def test_sorted_predict(rasa_app):
    data = json.dumps([event.as_dict() for event in test_events[:3]])
    _, response = rasa_app.put(
        "/conversations/sortedpredict/tracker/events",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    assert response.status == 200

    _, response = rasa_app.post("/conversations/sortedpredict/predict")
    scores = response.json["scores"]
    sorted_scores = sorted(scores, key=lambda k: (-k["score"], k["action"]))
    assert scores == sorted_scores


def test_get_tracker_with_jwt(rasa_secured_app):
    # token generated with secret "core" and algorithm HS256
    # on https://jwt.io/

    # {"user": {"username": "testadmin", "role": "admin"}}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdGFkbWluIiwic"
        "m9sZSI6ImFkbWluIn19.NAQr0kbtSrY7d28XTqRzawq2u"
        "QRre7IWTuIDrCn5AIw"
    }
    _, response = rasa_secured_app.get(
        "/conversations/testadmin/tracker", headers=jwt_header
    )
    assert response.status == 200

    _, response = rasa_secured_app.get(
        "/conversations/testuser/tracker", headers=jwt_header
    )
    assert response.status == 200

    # {"user": {"username": "testuser", "role": "user"}}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdHVzZXIiLCJyb"
        "2xlIjoidXNlciJ9fQ.JnMTLYd56qut2w9h7hRQlDm1n3l"
        "HJHOxxC_w7TtwCrs"
    }
    _, response = rasa_secured_app.get(
        "/conversations/testadmin/tracker", headers=jwt_header
    )
    assert response.status == 403

    _, response = rasa_secured_app.get(
        "/conversations/testuser/tracker", headers=jwt_header
    )
    assert response.status == 200


def test_list_routes(default_agent):
    from rasa import server

    app = server.create_app(default_agent, auth_token=None)

    routes = utils.list_routes(app)
    assert set(routes.keys()) == {
        "hello",
        "version",
        "status",
        "retrieve_tracker",
        "append_events",
        "replace_events",
        "retrieve_story",
        "execute_action",
        "predict",
        "add_message",
        "train",
        "evaluate_stories",
        "evaluate_intents",
        "tracker_predict",
        "parse",
        "load_model",
        "unload_model",
        "get_domain",
    }


def test_unload_model_error(rasa_app):
    _, response = rasa_app.get("/status")
    assert response.status == 200
    assert "model_file" in response.json and response.json["model_file"] is not None

    _, response = rasa_app.delete("/model")
    assert response.status == 204

    _, response = rasa_app.get("/status")
    assert response.status == 409


def test_get_domain(rasa_app):
    _, response = rasa_app.get("/domain", headers={"accept": "application/json"})

    content = response.json

    assert response.status == 200
    assert "config" in content
    assert "intents" in content
    assert "entities" in content
    assert "slots" in content
    assert "templates" in content
    assert "actions" in content


def test_get_domain_invalid_accept_header(rasa_app):
    _, response = rasa_app.get("/domain")

    assert response.status == 406


def test_load_model(rasa_app, trained_core_model):
    _, response = rasa_app.get("/status")

    assert response.status == 200
    assert "fingerprint" in response.json

    old_fingerprint = response.json["fingerprint"]

    data = {"model_file": trained_core_model}
    _, response = rasa_app.put("/model", json=data)

    assert response.status == 204

    _, response = rasa_app.get("/status")

    assert response.status == 200
    assert "fingerprint" in response.json

    assert old_fingerprint != response.json["fingerprint"]


def test_load_model_from_model_server(rasa_app, trained_core_model):
    _, response = rasa_app.get("/status")

    assert response.status == 200
    assert "fingerprint" in response.json

    old_fingerprint = response.json["fingerprint"]

    endpoint = EndpointConfig("https://example.com/model/trained_core_model")
    with open(trained_core_model, "rb") as f:
        with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
            headers = {}
            fs = os.fstat(f.fileno())
            headers["Content-Length"] = str(fs[6])
            mocked.get(
                "https://example.com/model/trained_core_model",
                content_type="application/x-tar",
                body=f.read(),
            )
            data = {"model_server": {"url": endpoint.url}}
            _, response = rasa_app.put("/model", json=data)

            assert response.status == 204

            _, response = rasa_app.get("/status")

            assert response.status == 200
            assert "fingerprint" in response.json

            assert old_fingerprint != response.json["fingerprint"]

    import rasa.core.jobs

    rasa.core.jobs.__scheduler = None


def test_load_model_invalid_request_body(rasa_app):
    _, response = rasa_app.put("/model")

    assert response.status == 400


def test_load_model_invalid_configuration(rasa_app):
    data = {"model_file": "some-random-path"}
    _, response = rasa_app.put("/model", json=data)

    assert response.status == 400
