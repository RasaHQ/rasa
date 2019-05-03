# -*- coding: utf-8 -*-
import json
import os
import tempfile
import uuid

import pytest
from freezegun import freeze_time

import rasa
import rasa.constants
from rasa.core import events
from rasa.core.events import Event, UserUttered, SlotSet, BotUttered
from rasa.model import unpack_model
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
def test_post_parse(rasa_app, response_test):
    _, response = rasa_app.post(response_test.endpoint, json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text"])
    assert rjs["entities"] == response_test.expected_response["entities"]
    assert rjs["text"] == response_test.expected_response["text"]


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
def test_post_parse_without_interpreter(rasa_app_core, response_test):
    _, response = rasa_app_core.post(response_test.endpoint, json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text"])
    assert rjs["entities"] == response_test.expected_response["entities"]
    assert rjs["text"] == response_test.expected_response["text"]


def test_post_train_stack_success(
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

    # save model to temporary file
    tempdir = tempfile.mkdtemp()
    model_path = os.path.join(tempdir, "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    # unpack model and ensure fingerprint is present
    model_path = unpack_model(model_path)
    assert os.path.exists(os.path.join(model_path, "fingerprint.json"))


def test_post_train_nlu_success(
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


def test_post_train_core_success(
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


def test_post_train_missing_config(rasa_app):
    payload = dict(domain="domain data", config=None)

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 400


def test_post_train_missing_training_data(rasa_app):
    payload = dict(domain="domain data", config="config data")

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 400


def test_post_train_internal_error(rasa_app):
    payload = dict(domain="domain data", config="config data", nlu="nlu data")

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 500


def test_evaluate(rasa_app, default_stories_file):
    with open(default_stories_file, "r") as f:
        stories = f.read()

    _, response = rasa_app.post("/model/evaluate/stories", data=stories)

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


def test_end_to_end_evaluation(rasa_app, end_to_end_story_file):
    with open(end_to_end_story_file, "r") as f:
        stories = f.read()

    _, response = rasa_app.post("/model/evaluate/stories?e2e=true", data=stories)

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


def test_intent_evaluation(rasa_app, default_nlu_data):
    with open(default_nlu_data, "r") as f:
        nlu_data = f.read()

    _, response = rasa_app.post("/model/evaluate/intents", data=nlu_data)

    assert response.status == 200
    assert set(response.json.keys()) == {"intent_evaluation", "entity_evaluation"}


def test_parse(rasa_app):
    data = json.dumps({"text": """/greet{"name": "Rasa"}"""})
    _, response = rasa_app.post(
        "/model/parse", data=data, headers={"Content-Type": "application/json"}
    )
    content = response.json
    assert response.status == 200
    assert content == {
        "entities": [{"end": 22, "entity": "name", "start": 6, "value": "Rasa"}],
        "intent": {"confidence": 1.0, "name": "greet"},
        "intent_ranking": [{"confidence": 1.0, "name": "greet"}],
        "text": '/greet{"name": "Rasa"}',
    }


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
    assert content == {
        "entities": [{"end": 22, "entity": "name", "start": 6, "value": "Rasa"}],
        "intent": {"confidence": 1.0, "name": "greet"},
        "intent_ranking": [{"confidence": 1.0, "name": "greet"}],
        "text": '/greet{"name": "Rasa"}',
    }


# def test_model_hot_reloading(app, rasa_default_train_data):
#     query = "/parse?q=hello&model=test-model"
#
#     # Model could not be found, fallback model was used instead
#     _, response = app.get(query)
#     assert response.status == 200
#     rjs = response.json
#     assert rjs["model"] == FALLBACK_MODEL_NAME
#
#     # Train a new model - model will be loaded automatically
#     train_u = "/train?model=test-model"
#     request = {
#         "language": "en",
#         "pipeline": "pretrained_embeddings_spacy",
#         "data": rasa_default_train_data,
#     }
#     model_str = yaml.safe_dump(request, default_flow_style=False, allow_unicode=True)
#     _, response = app.post(
#         train_u, headers={"Content-Type": "application/x-yml"}, data=model_str
#     )
#     assert response.status == 200, "Training should end successfully"
#
#     _, response = app.post(
#         train_u, headers={"Content-Type": "application/json"}, data=json.dumps(request)
#     )
#     assert response.status == 200, "Training should end successfully"
#
#     # Model should be there now
#     _, response = app.get(query)
#     assert response.status == 200, "Model should now exist after it got trained"
#     rjs = response.json
#     assert "test-model" in rjs["model"]
#
#
# def test_evaluate_invalid_model_error(app, rasa_default_train_data):
#     _, response = app.post("/evaluate?model=not-existing", json=rasa_default_train_data)
#
#     rjs = response.json
#     assert response.status == 500
#     assert "details" in rjs
#     assert rjs["details"]["error"] == "Model with name 'not-existing' is not loaded."
#
#
# def test_evaluate_unsupported_model_error(app_without_model, rasa_default_train_data):
#     _, response = app_without_model.post("/evaluate", json=rasa_default_train_data)
#
#     rjs = response.json
#     assert response.status == 500
#     assert "details" in rjs
#     assert rjs["details"]["error"] == "No model is loaded. Cannot evaluate."
#
#
# def test_evaluate_internal_error(app, rasa_default_train_data):
#     _, response = app.post(
#         "/evaluate", json={"data": "dummy_data_for_triggering_an_error"}
#     )
#     assert response.status == 500, "The training data format is not valid"
#
#
# def test_evaluate(app, rasa_default_train_data):
#     _, response = app.post(
#         "/evaluate?model={}".format(NLU_MODEL_NAME), json=rasa_default_train_data
#     )
#
#     rjs = response.json
#     assert "intent_evaluation" in rjs
#     assert "entity_evaluation" in rjs
#     assert all(
#         prop in rjs["intent_evaluation"]
#         for prop in ["report", "predictions", "precision", "f1_score", "accuracy"]
#     )
#     assert response.status == 200, "Evaluation should start"
#
#
# def test_unload_model_error(app):
#     request = "/models?model=my_model"
#     _, response = app.delete(request)
#     rjs = response.json
#     assert (
#         response.status == 404
#     ), "Model is not loaded and can therefore not be unloaded."
#     assert rjs["details"]["error"] == "Model with name 'my_model' is not loaded."
#
#
# def test_unload_model(app):
#     unload = "/models?model={}".format(NLU_MODEL_NAME)
#     _, response = app.delete(unload)
#     assert response.status == 204, "No Content"
#
#
# def test_status_after_unloading(app):
#     _, response = app.get("/status")
#     rjs = response.json
#     assert response.status == 200
#     assert rjs["loaded_model"] == NLU_MODEL_NAME
#
#     unload = "/models?model={}".format(NLU_MODEL_NAME)
#     _, response = app.delete(unload)
#     assert response.status == 204, "No Content"
#
#     _, response = app.get("/status")
#     rjs = response.json
#     assert response.status == 200
#     assert rjs["loaded_model"] is None


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
