import os
import time
import tempfile
import uuid
from multiprocessing import Process, Manager
from typing import List, Text, Type
from contextlib import ExitStack

from aioresponses import aioresponses

import pytest
from freezegun import freeze_time
from mock import MagicMock

import rasa
import rasa.constants
from rasa.core import events, utils
from rasa.core.channels import CollectingOutputChannel, RestInput, SlackInput
from rasa.core.channels.slack import SlackBot
from rasa.core.events import Event, UserUttered, SlotSet, BotUttered
from rasa.core.trackers import DialogueStateTracker
from rasa.model import unpack_model
from rasa.utils.endpoints import EndpointConfig
from sanic import Sanic
from sanic.testing import SanicTestClient, PORT
from tests.nlu.utilities import ResponseTest
from tests.conftest import get_test_client


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
def rasa_app_without_api(rasa_server_without_api: Sanic) -> SanicTestClient:
    return get_test_client(rasa_server_without_api)


@pytest.fixture
def rasa_app(rasa_server: Sanic) -> SanicTestClient:
    return get_test_client(rasa_server)


@pytest.fixture
def rasa_app_nlu(rasa_nlu_server: Sanic) -> SanicTestClient:
    return get_test_client(rasa_nlu_server)


@pytest.fixture
def rasa_app_core(rasa_core_server: Sanic) -> SanicTestClient:
    return get_test_client(rasa_core_server)


@pytest.fixture
def rasa_secured_app(rasa_server_secured: Sanic) -> SanicTestClient:
    return get_test_client(rasa_server_secured)


def test_root(rasa_app: SanicTestClient):
    _, response = rasa_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_root_without_enable_api(rasa_app_without_api: SanicTestClient):

    _, response = rasa_app_without_api.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_root_secured(rasa_secured_app: SanicTestClient):
    _, response = rasa_secured_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_version(rasa_app: SanicTestClient):
    _, response = rasa_app.get("/version")
    content = response.json
    assert response.status == 200
    assert content.get("version") == rasa.__version__
    assert (
        content.get("minimum_compatible_version")
        == rasa.constants.MINIMUM_COMPATIBLE_VERSION
    )


def test_status(rasa_app: SanicTestClient, trained_rasa_model: Text):
    _, response = rasa_app.get("/status")
    model_file = response.json["model_file"]
    assert response.status == 200
    assert "fingerprint" in response.json
    assert os.path.isfile(model_file)
    assert model_file == trained_rasa_model


def test_status_nlu_only(rasa_app_nlu: SanicTestClient, trained_nlu_model: Text):
    _, response = rasa_app_nlu.get("/status")
    model_file = response.json["model_file"]
    assert response.status == 200
    assert "fingerprint" in response.json
    assert "model_file" in response.json
    assert model_file == trained_nlu_model


def test_status_secured(rasa_secured_app: SanicTestClient):
    _, response = rasa_secured_app.get("/status")
    assert response.status == 401


def test_status_not_ready_agent(rasa_app: SanicTestClient):
    rasa_app.app.agent = None
    _, response = rasa_app.get("/status")
    assert response.status == 409


@pytest.fixture
def formbot_data():
    return dict(
        domain="examples/formbot/domain.yml",
        config="examples/formbot/config.yml",
        stories="examples/formbot/data/stories.md",
        nlu="examples/formbot/data/nlu.md",
    )


def test_train_status(rasa_server, rasa_app, formbot_data):
    with ExitStack() as stack:
        payload = {
            key: stack.enter_context(open(path)).read()
            for key, path in formbot_data.items()
        }

    def train(results):
        client1 = SanicTestClient(rasa_server, port=PORT + 1)
        _, train_resp = client1.post("/model/train", json=payload)
        results["train_response_code"] = train_resp.status

    # Run training process in the background
    manager = Manager()
    results = manager.dict()
    p1 = Process(target=train, args=(results,))
    p1.start()

    # Query the status endpoint a few times to ensure the test does
    # not fail prematurely due to mismatched timing of a single query.
    for i in range(10):
        time.sleep(1)
        _, status_resp = rasa_app.get("/status")
        assert status_resp.status == 200
        if status_resp.json["num_active_training_jobs"] == 1:
            break
    assert status_resp.json["num_active_training_jobs"] == 1

    p1.join()
    assert results["train_response_code"] == 200

    _, status_resp = rasa_app.get("/status")
    assert status_resp.status == 200
    assert status_resp.json["num_active_training_jobs"] == 0


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


def test_parse_without_nlu_model(rasa_app_core: SanicTestClient):
    _, response = rasa_app_core.post("/model/parse", json={"text": "hello"})
    assert response.status == 200

    rjs = response.json
    assert all(prop in rjs for prop in ["entities", "intent", "text"])


def test_parse_on_invalid_emulation_mode(rasa_app_nlu: SanicTestClient):
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
    with ExitStack() as stack:
        domain_file = stack.enter_context(open(default_domain_path))
        config_file = stack.enter_context(open(default_stack_config))
        stories_file = stack.enter_context(open(default_stories_file))
        nlu_file = stack.enter_context(open(default_nlu_data))

        payload = dict(
            domain=domain_file.read(),
            config=config_file.read(),
            stories=stories_file.read(),
            nlu=nlu_file.read(),
        )

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
    with ExitStack() as stack:
        domain_file = stack.enter_context(open(default_domain_path))
        config_file = stack.enter_context(open(default_stack_config))
        nlu_file = stack.enter_context(open(default_nlu_data))

        payload = dict(
            domain=domain_file.read(), config=config_file.read(), nlu=nlu_file.read()
        )

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
    with ExitStack() as stack:
        domain_file = stack.enter_context(open(default_domain_path))
        config_file = stack.enter_context(open(default_stack_config))
        core_file = stack.enter_context(open(default_stories_file))

        payload = dict(
            domain=domain_file.read(),
            config=config_file.read(),
            stories=core_file.read(),
        )

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


def test_train_missing_config(rasa_app: SanicTestClient):
    payload = dict(domain="domain data", config=None)

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 400


def test_train_missing_training_data(rasa_app: SanicTestClient):
    payload = dict(domain="domain data", config="config data")

    _, response = rasa_app.post("/model/train", json=payload)
    assert response.status == 400


def test_train_internal_error(rasa_app: SanicTestClient):
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


def test_evaluate_stories_not_ready_agent(
    rasa_app_nlu: SanicTestClient, default_stories_file
):
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
    assert set(response.json.keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }


def test_evaluate_intent_on_just_nlu_model(
    rasa_app_nlu: SanicTestClient, default_nlu_data
):
    with open(default_nlu_data, "r") as f:
        nlu_data = f.read()

    _, response = rasa_app_nlu.post("/model/test/intents", data=nlu_data)

    assert response.status == 200
    assert set(response.json.keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }


def test_evaluate_intent_with_query_param(
    rasa_app, trained_nlu_model, default_nlu_data
):
    _, response = rasa_app.get("/status")
    previous_model_file = response.json["model_file"]

    with open(default_nlu_data, "r") as f:
        nlu_data = f.read()

    _, response = rasa_app.post(
        f"/model/test/intents?model={trained_nlu_model}", data=nlu_data
    )

    assert response.status == 200
    assert set(response.json.keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }

    _, response = rasa_app.get("/status")
    assert previous_model_file == response.json["model_file"]


def test_predict(rasa_app: SanicTestClient):
    data = {
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
    _, response = rasa_app.post(
        "/model/predict", json=data, headers={"Content-Type": "application/json"}
    )
    content = response.json
    assert response.status == 200
    assert "scores" in content
    assert "tracker" in content
    assert "policy" in content


@freeze_time("2018-01-01")
def test_requesting_non_existent_tracker(rasa_app: SanicTestClient):
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
    assert content["latest_message"] == {
        "text": None,
        "intent": {},
        "entities": [],
        "message_id": None,
        "metadata": {},
    }


@pytest.mark.parametrize("event", test_events)
def test_pushing_event(rasa_app, event):
    cid = str(uuid.uuid1())
    conversation = f"/conversations/{cid}"

    _, response = rasa_app.post(
        f"{conversation}/tracker/events",
        json=event.as_dict(),
        headers={"Content-Type": "application/json"},
    )
    assert response.json is not None
    assert response.status == 200

    _, tracker_response = rasa_app.get(f"/conversations/{cid}/tracker")
    tracker = tracker_response.json
    assert tracker is not None
    assert len(tracker.get("events")) == 2

    evt = tracker.get("events")[1]
    assert Event.from_parameters(evt) == event


def test_push_multiple_events(rasa_app: SanicTestClient):
    cid = str(uuid.uuid1())
    conversation = f"/conversations/{cid}"

    events = [e.as_dict() for e in test_events]
    _, response = rasa_app.post(
        f"{conversation}/tracker/events",
        json=events,
        headers={"Content-Type": "application/json"},
    )
    assert response.json is not None
    assert response.status == 200

    _, tracker_response = rasa_app.get(f"/conversations/{cid}/tracker")
    tracker = tracker_response.json
    assert tracker is not None

    # there is also an `ACTION_LISTEN` event at the start
    assert len(tracker.get("events")) == len(test_events) + 1
    assert tracker.get("events")[1:] == events


def test_put_tracker(rasa_app: SanicTestClient):
    data = [event.as_dict() for event in test_events]
    _, response = rasa_app.put(
        "/conversations/pushtracker/tracker/events",
        json=data,
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


def test_sorted_predict(rasa_app: SanicTestClient):
    _create_tracker_for_sender(rasa_app, "sortedpredict")

    _, response = rasa_app.post("/conversations/sortedpredict/predict")
    scores = response.json["scores"]
    sorted_scores = sorted(scores, key=lambda k: (-k["score"], k["action"]))
    assert scores == sorted_scores


def _create_tracker_for_sender(app: SanicTestClient, sender_id: Text) -> None:
    data = [event.as_dict() for event in test_events[:3]]
    _, response = app.put(
        f"/conversations/{sender_id}/tracker/events",
        json=data,
        headers={"Content-Type": "application/json"},
    )

    assert response.status == 200


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


def test_unload_model_error(rasa_app: SanicTestClient):
    _, response = rasa_app.get("/status")
    assert response.status == 200
    assert "model_file" in response.json and response.json["model_file"] is not None

    _, response = rasa_app.delete("/model")
    assert response.status == 204


def test_get_domain(rasa_app: SanicTestClient):
    _, response = rasa_app.get("/domain", headers={"accept": "application/json"})

    content = response.json

    assert response.status == 200
    assert "config" in content
    assert "intents" in content
    assert "entities" in content
    assert "slots" in content
    assert "templates" in content
    assert "actions" in content


def test_get_domain_invalid_accept_header(rasa_app: SanicTestClient):
    _, response = rasa_app.get("/domain")

    assert response.status == 406


def test_load_model(rasa_app: SanicTestClient, trained_core_model):
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


def test_load_model_from_model_server(rasa_app: SanicTestClient, trained_core_model):
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


def test_load_model_invalid_request_body(rasa_app: SanicTestClient):
    _, response = rasa_app.put("/model")

    assert response.status == 400


def test_load_model_invalid_configuration(rasa_app: SanicTestClient):
    data = {"model_file": "some-random-path"}
    _, response = rasa_app.put("/model", json=data)

    assert response.status == 400


def test_execute(rasa_app: SanicTestClient):
    _create_tracker_for_sender(rasa_app, "test_execute")

    data = {"name": "utter_greet"}
    _, response = rasa_app.post("/conversations/test_execute/execute", json=data)

    assert response.status == 200

    parsed_content = response.json
    assert parsed_content["tracker"]
    assert parsed_content["messages"]


def test_execute_with_missing_action_name(rasa_app: SanicTestClient):
    test_sender = "test_execute_with_missing_action_name"
    _create_tracker_for_sender(rasa_app, test_sender)

    data = {"wrong-key": "utter_greet"}
    _, response = rasa_app.post(f"/conversations/{test_sender}/execute", json=data)

    assert response.status == 400


def test_execute_with_not_existing_action(rasa_app: SanicTestClient):
    test_sender = "test_execute_with_not_existing_action"
    _create_tracker_for_sender(rasa_app, test_sender)

    data = {"name": "ka[pa[opi[opj[oj[oija"}
    _, response = rasa_app.post(f"/conversations/{test_sender}/execute", json=data)

    assert response.status == 500


@pytest.mark.parametrize(
    "input_channels, output_channel_to_use, expected_channel",
    [
        (None, "slack", CollectingOutputChannel),
        ([], None, CollectingOutputChannel),
        ([RestInput()], "slack", CollectingOutputChannel),
        ([RestInput()], "rest", CollectingOutputChannel),
        ([RestInput(), SlackInput("test")], "slack", SlackBot),
    ],
)
def test_get_output_channel(
    input_channels: List[Text], output_channel_to_use, expected_channel: Type
):
    request = MagicMock()
    app = MagicMock()
    app.input_channels = input_channels
    request.app = app
    request.args = {"output_channel": output_channel_to_use}

    actual = rasa.server._get_output_channel(request, None)

    assert isinstance(actual, expected_channel)


@pytest.mark.parametrize(
    "input_channels, expected_channel",
    [
        ([], CollectingOutputChannel),
        ([RestInput()], CollectingOutputChannel),
        ([RestInput(), SlackInput("test")], SlackBot),
    ],
)
def test_get_latest_output_channel(input_channels: List[Text], expected_channel: Type):
    request = MagicMock()
    app = MagicMock()
    app.input_channels = input_channels
    request.app = app
    request.args = {"output_channel": "latest"}

    tracker = DialogueStateTracker.from_events(
        "default", [UserUttered("text", input_channel="slack")]
    )

    actual = rasa.server._get_output_channel(request, tracker)

    assert isinstance(actual, expected_channel)


def test_app_when_app_has_no_input_channels():
    request = MagicMock()

    class NoInputChannels:
        pass

    request.app = NoInputChannels()

    actual = rasa.server._get_output_channel(
        request, DialogueStateTracker.from_events("default", [])
    )
    assert isinstance(actual, CollectingOutputChannel)
