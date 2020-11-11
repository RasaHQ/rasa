import os
from multiprocessing.managers import DictProxy
from pathlib import Path
from unittest.mock import Mock, ANY

import requests
import time
import uuid
import urllib.parse

from typing import List, Text, Type, Generator, NoReturn, Dict, Optional
from contextlib import ExitStack

from _pytest import pathlib
from _pytest.monkeypatch import MonkeyPatch
from aioresponses import aioresponses

import pytest
from freezegun import freeze_time
from mock import MagicMock
from multiprocessing import Process, Manager

import rasa
import rasa.constants
import rasa.shared.constants
import rasa.shared.utils.io
import rasa.utils.io
import rasa.server
from rasa.core import utils
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core import events
from rasa.core.agent import Agent
from rasa.core.channels import (
    channel,
    CollectingOutputChannel,
    RestInput,
    SlackInput,
    CallbackInput,
)
from rasa.core.channels.slack import SlackBot
from rasa.shared.core.constants import ACTION_SESSION_START_NAME, ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain, SessionConfig
from rasa.shared.core.events import (
    Event,
    UserUttered,
    SlotSet,
    BotUttered,
    ActionExecuted,
    SessionStarted,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.model import unpack_model
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.utils.endpoints import EndpointConfig
from sanic import Sanic
from sanic.testing import SanicASGITestClient
from tests.nlu.utilities import ResponseTest
from tests.utilities import json_of_latest_request, latest_request
from ruamel.yaml import StringIO


# a couple of event instances that we can use for testing
test_events = [
    Event.from_parameters(
        {
            "event": UserUttered.type_name,
            "text": "/goodbye",
            "parse_data": {
                "intent": {"confidence": 1.0, INTENT_NAME_KEY: "greet"},
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

# sequence of events expected at the beginning of trackers
session_start_sequence: List[Event] = [
    ActionExecuted(ACTION_SESSION_START_NAME),
    SessionStarted(),
    ActionExecuted(ACTION_LISTEN_NAME),
]


@pytest.fixture
def rasa_app_without_api(rasa_server_without_api: Sanic) -> SanicASGITestClient:
    return rasa_server_without_api.asgi_client


@pytest.fixture
def rasa_app(rasa_server: Sanic) -> SanicASGITestClient:
    return rasa_server.asgi_client


@pytest.fixture
def rasa_app_nlu(rasa_nlu_server: Sanic) -> SanicASGITestClient:
    return rasa_nlu_server.asgi_client


@pytest.fixture
def rasa_app_core(rasa_core_server: Sanic) -> SanicASGITestClient:
    return rasa_core_server.asgi_client


@pytest.fixture
def rasa_secured_app(rasa_server_secured: Sanic) -> SanicASGITestClient:
    return rasa_server_secured.asgi_client


async def test_root(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


async def test_root_without_enable_api(rasa_app_without_api: SanicASGITestClient):
    _, response = await rasa_app_without_api.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


async def test_root_secured(rasa_secured_app: SanicASGITestClient):
    _, response = await rasa_secured_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


async def test_version(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.get("/version")
    content = response.json()
    assert response.status == 200
    assert content.get("version") == rasa.__version__
    assert (
        content.get("minimum_compatible_version")
        == rasa.constants.MINIMUM_COMPATIBLE_VERSION
    )


async def test_status(rasa_app: SanicASGITestClient, trained_rasa_model: Text):
    _, response = await rasa_app.get("/status")
    model_file = response.json()["model_file"]
    assert response.status == 200
    assert "fingerprint" in response.json()
    assert os.path.isfile(model_file)
    assert model_file == trained_rasa_model


async def test_status_nlu_only(
    rasa_app_nlu: SanicASGITestClient, trained_nlu_model: Text
):
    _, response = await rasa_app_nlu.get("/status")
    model_file = response.json()["model_file"]
    assert response.status == 200
    assert "fingerprint" in response.json()
    assert "model_file" in response.json()
    assert model_file == trained_nlu_model


async def test_status_secured(rasa_secured_app: SanicASGITestClient):
    _, response = await rasa_secured_app.get("/status")
    assert response.status == 401


async def test_status_not_ready_agent(rasa_app: SanicASGITestClient):
    rasa_app.app.agent = None
    _, response = await rasa_app.get("/status")
    assert response.status == 409


@pytest.fixture
def shared_statuses() -> DictProxy:
    return Manager().dict()


@pytest.fixture
def background_server(
    shared_statuses: DictProxy, tmpdir: pathlib.Path
) -> Generator[Process, None, None]:
    # Create a fake model archive which the mocked train function can return

    fake_model = Path(tmpdir) / "fake_model.tar.gz"
    fake_model.touch()
    fake_model_path = str(fake_model)

    # Fake training function which blocks until we tell it to stop blocking
    # If we can send a status request while this is blocking, we can be sure that the
    # actual training is also not blocking
    def mocked_training_function(*_, **__) -> Text:
        # Tell the others that we are now blocking
        shared_statuses["started_training"] = True
        # Block until somebody tells us to not block anymore
        while shared_statuses.get("stop_training") is not True:
            time.sleep(1)

        return fake_model_path

    def run_server() -> NoReturn:
        rasa.train = mocked_training_function

        from rasa import __main__
        import sys

        sys.argv = ["rasa", "run", "--enable-api"]
        __main__.main()

    server = Process(target=run_server)
    yield server
    server.terminate()


@pytest.fixture()
def training_request(
    shared_statuses: DictProxy, tmp_path: Path
) -> Generator[Process, None, None]:
    def send_request() -> None:
        payload = {}
        project_path = Path("examples") / "formbot"

        for file in [
            "domain.yml",
            "config.yml",
            Path("data") / "rules.yml",
            Path("data") / "stories.yml",
            Path("data") / "nlu.yml",
        ]:
            full_path = project_path / file
            # Read in as dictionaries to avoid that keys, which are specified in
            # multiple files (such as 'version'), clash.
            content = rasa.shared.utils.io.read_yaml_file(full_path)
            payload.update(content)

        concatenated_payload_file = tmp_path / "concatenated.yml"
        rasa.shared.utils.io.write_yaml(payload, concatenated_payload_file)

        payload_as_yaml = concatenated_payload_file.read_text()

        response = requests.post(
            "http://localhost:5005/model/train",
            data=payload_as_yaml,
            headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
            params={"force_training": True},
        )
        shared_statuses["training_result"] = response.status_code

    train_request = Process(target=send_request)
    yield train_request
    train_request.terminate()


# Due to unknown reasons this test can not be run in pycharm, it
# results in segfaults...will skip in that case - test will still get run on CI.
# It also doesn't run on Windows because of Process-related calls and an attempt
# to start/terminate a process. We will investigate this case further later:
# https://github.com/RasaHQ/rasa/issues/6302
@pytest.mark.skipif("PYCHARM_HOSTED" in os.environ, reason="results in segfault")
@pytest.mark.skip_on_windows
def test_train_status_is_not_blocked_by_training(
    background_server: Process, shared_statuses: DictProxy, training_request: Process
):
    background_server.start()

    def is_server_ready() -> bool:
        try:
            return requests.get("http://localhost:5005/status").status_code == 200
        except Exception:
            return False

    # wait until server is up before sending train request and status test loop
    start = time.time()
    while not is_server_ready() and time.time() - start < 60:
        time.sleep(1)

    assert is_server_ready()

    training_request.start()

    # Wait until the blocking training function was called
    start = time.time()
    while (
        shared_statuses.get("started_training") is not True and time.time() - start < 60
    ):
        time.sleep(1)

    # Check if the number of currently running trainings was incremented
    response = requests.get("http://localhost:5005/status")
    assert response.status_code == 200
    assert response.json()["num_active_training_jobs"] == 1

    # Tell the blocking training function to stop
    shared_statuses["stop_training"] = True

    start = time.time()
    while shared_statuses.get("training_result") is None and time.time() - start < 60:
        time.sleep(1)
    assert shared_statuses.get("training_result")

    # Check that the training worked correctly
    assert shared_statuses["training_result"] == 200

    # Check if the number of currently running trainings was decremented
    response = requests.get("http://localhost:5005/status")
    assert response.status_code == 200
    assert response.json()["num_active_training_jobs"] == 0


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, INTENT_NAME_KEY: "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, INTENT_NAME_KEY: "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, INTENT_NAME_KEY: "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={"text": "hello ńöñàśçií"},
        ),
    ],
)
async def test_parse(rasa_app: SanicASGITestClient, response_test: ResponseTest):
    _, response = await rasa_app.post(
        response_test.endpoint, json=response_test.payload
    )
    rjs = response.json()
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
                "intent": {"confidence": 1.0, INTENT_NAME_KEY: "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse?emulation_mode=dialogflow",
            {
                "entities": [],
                "intent": {"confidence": 1.0, INTENT_NAME_KEY: "greet"},
                "text": "hello",
            },
            payload={"text": "hello"},
        ),
        ResponseTest(
            "/model/parse?emulation_mode=luis",
            {
                "entities": [],
                "intent": {"confidence": 1.0, INTENT_NAME_KEY: "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={"text": "hello ńöñàśçií"},
        ),
    ],
)
async def test_parse_with_different_emulation_mode(
    rasa_app: SanicASGITestClient, response_test: ResponseTest
):
    _, response = await rasa_app.post(
        response_test.endpoint, json=response_test.payload
    )
    assert response.status == 200


async def test_parse_without_nlu_model(rasa_app_core: SanicASGITestClient):
    _, response = await rasa_app_core.post("/model/parse", json={"text": "hello"})
    assert response.status == 200

    rjs = response.json()
    assert all(prop in rjs for prop in ["entities", "intent", "text"])


async def test_parse_on_invalid_emulation_mode(rasa_app_nlu: SanicASGITestClient):
    _, response = await rasa_app_nlu.post(
        "/model/parse?emulation_mode=ANYTHING", json={"text": "hello"}
    )
    assert response.status == 400


async def test_train_stack_success(
    rasa_app: SanicASGITestClient,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
    default_nlu_data: Text,
    tmp_path: Path,
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

    _, response = await rasa_app.post("/model/train", json=payload)
    assert response.status == 200

    assert response.headers["filename"] is not None

    # save model to temporary file
    model_path = str(tmp_path / "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    # unpack model and ensure fingerprint is present
    model_path = unpack_model(model_path)
    assert os.path.exists(os.path.join(model_path, "fingerprint.json"))


async def test_train_nlu_success(
    rasa_app: SanicASGITestClient,
    default_stack_config: Text,
    default_nlu_data: Text,
    default_domain_path: Text,
    tmp_path: Path,
):
    domain_data = rasa.shared.utils.io.read_yaml_file(default_domain_path)
    config_data = rasa.shared.utils.io.read_yaml_file(default_stack_config)
    nlu_data = rasa.shared.utils.io.read_yaml_file(default_nlu_data)

    # combine all data into our payload
    payload = {
        key: val for d in [domain_data, config_data, nlu_data] for key, val in d.items()
    }

    data = StringIO()
    rasa.shared.utils.io.write_yaml(payload, data)

    _, response = await rasa_app.post(
        "/model/train",
        data=data.getvalue(),
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )
    assert response.status == 200

    # save model to temporary file
    model_path = str(tmp_path / "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    # unpack model and ensure fingerprint is present
    model_path = unpack_model(model_path)
    assert os.path.exists(os.path.join(model_path, "fingerprint.json"))


async def test_train_core_success(
    rasa_app: SanicASGITestClient,
    default_stack_config: Text,
    default_stories_file: Text,
    default_domain_path: Text,
    tmp_path: Path,
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

    _, response = await rasa_app.post("/model/train", json=payload)
    assert response.status == 200

    # save model to temporary file
    model_path = str(tmp_path / "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    # unpack model and ensure fingerprint is present
    model_path = unpack_model(model_path)
    assert os.path.exists(os.path.join(model_path, "fingerprint.json"))


async def test_train_with_retrieval_events_success(
    rasa_app: SanicASGITestClient, default_stack_config: Text, tmp_path: Path
):
    with ExitStack() as stack:
        domain_file = stack.enter_context(
            open("data/test_domains/default_retrieval_intents.yml")
        )
        config_file = stack.enter_context(open(default_stack_config))
        core_file = stack.enter_context(
            open("data/test_stories/stories_retrieval_intents.md")
        )
        responses_file = stack.enter_context(open("data/test_responses/default.md"))
        nlu_file = stack.enter_context(
            open("data/test_nlu/default_retrieval_intents.md")
        )

        payload = dict(
            domain=domain_file.read(),
            config=config_file.read(),
            stories=core_file.read(),
            responses=responses_file.read(),
            nlu=nlu_file.read(),
        )

    _, response = await rasa_app.post("/model/train", json=payload, timeout=60 * 5)
    assert response.status == 200
    assert_trained_model(response.body, tmp_path)


def assert_trained_model(response_body: bytes, tmp_path: Path) -> None:
    # save model to temporary file
    model_path = str(tmp_path / "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response_body)

    # unpack model and ensure fingerprint is present
    model_path = unpack_model(model_path)
    assert os.path.exists(os.path.join(model_path, "fingerprint.json"))


@pytest.mark.parametrize(
    "payload",
    [
        {"config": None, "stories": None, "nlu": None, "domain": None, "force": True},
        {
            "config": None,
            "stories": None,
            "nlu": None,
            "domain": None,
            "force": False,
            "save_to_default_model_directory": True,
        },
        {
            "config": None,
            "stories": None,
            "nlu": None,
            "domain": None,
            "save_to_default_model_directory": False,
        },
    ],
)
def test_deprecation_warnings_json_payload(payload: Dict):
    with pytest.warns(FutureWarning):
        rasa.server._validate_json_training_payload(payload)


async def test_train_with_yaml(rasa_app: SanicASGITestClient, tmp_path: Path):
    training_data = """
stories:
- story: My story
  steps:
  - intent: greet
  - action: utter_greet

rules:
- rule: My rule
  steps:
  - intent: greet
  - action: utter_greet

intents:
- greet

nlu:
- intent: greet
  examples: |
    - hi
    - hello

responses:
 utter_greet:
 - text: Hi

language: en

polices:
- name: RulePolicy

pipeline:
  - name: KeywordIntentClassifier
"""
    _, response = await rasa_app.post(
        "/model/train",
        data=training_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == 200
    assert_trained_model(response.body, tmp_path)


async def test_train_with_invalid_yaml(rasa_app: SanicASGITestClient):
    invalid_yaml = """
rules:
rule my rule
"""

    _, response = await rasa_app.post(
        "/model/train",
        data=invalid_yaml,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )
    assert response.status == 400


@pytest.mark.parametrize(
    "headers, expected",
    [({}, False), ({"force_training": False}, False), ({"force_training": True}, True)],
)
def test_training_payload_from_yaml_force_training(headers: Dict, expected: bool):
    request = Mock()
    request.body = b""
    request.args = headers

    payload = rasa.server._training_payload_from_yaml(request)
    assert payload.get("force_training") == expected


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({}, rasa.shared.constants.DEFAULT_MODELS_PATH),
        ({"save_to_default_model_directory": False}, ANY),
        (
            {"save_to_default_model_directory": True},
            rasa.shared.constants.DEFAULT_MODELS_PATH,
        ),
    ],
)
def test_training_payload_from_yaml_save_to_default_model_directory(
    headers: Dict, expected: Text
):
    request = Mock()
    request.body = b""
    request.args = headers

    payload = rasa.server._training_payload_from_yaml(request)
    assert payload.get("output")
    assert payload.get("output") == expected


async def test_train_missing_config(rasa_app: SanicASGITestClient):
    payload = dict(domain="domain data", config=None)

    _, response = await rasa_app.post("/model/train", json=payload)
    assert response.status == 400


async def test_train_missing_training_data(rasa_app: SanicASGITestClient):
    payload = dict(domain="domain data", config="config data")

    _, response = await rasa_app.post("/model/train", json=payload)
    assert response.status == 400


async def test_train_internal_error(rasa_app: SanicASGITestClient):
    payload = dict(domain="domain data", config="config data", nlu="nlu data")

    _, response = await rasa_app.post("/model/train", json=payload)
    assert response.status == 500


async def test_evaluate_stories(
    rasa_app: SanicASGITestClient, default_stories_file: Text
):
    stories = rasa.shared.utils.io.read_file(default_stories_file)

    _, response = await rasa_app.post("/model/test/stories", data=stories)

    assert response.status == 200

    js = response.json()
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


async def test_evaluate_stories_not_ready_agent(
    rasa_app_nlu: SanicASGITestClient, default_stories_file: Text
):
    stories = rasa.shared.utils.io.read_file(default_stories_file)

    _, response = await rasa_app_nlu.post("/model/test/stories", data=stories)

    assert response.status == 409


async def test_evaluate_stories_end_to_end(
    rasa_app: SanicASGITestClient, end_to_end_story_file: Text
):
    stories = rasa.shared.utils.io.read_file(end_to_end_story_file)

    _, response = await rasa_app.post("/model/test/stories?e2e=true", data=stories)

    assert response.status == 200
    js = response.json()
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
    assert js["actions"] != []
    assert set(js["actions"][0].keys()) == {
        "action",
        "predicted",
        "confidence",
        "policy",
    }


async def test_evaluate_intent(rasa_app: SanicASGITestClient, default_nlu_data: Text):
    nlu_data = rasa.shared.utils.io.read_file(default_nlu_data)

    _, response = await rasa_app.post(
        "/model/test/intents",
        data=nlu_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == 200
    assert set(response.json().keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }


async def test_evaluate_intent_on_just_nlu_model(
    rasa_app_nlu: SanicASGITestClient, default_nlu_data: Text
):
    nlu_data = rasa.shared.utils.io.read_file(default_nlu_data)

    _, response = await rasa_app_nlu.post(
        "/model/test/intents",
        data=nlu_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == 200
    assert set(response.json().keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }


async def test_evaluate_intent_with_query_param(
    rasa_app: SanicASGITestClient, trained_nlu_model, default_nlu_data: Text
):
    _, response = await rasa_app.get("/status")
    previous_model_file = response.json()["model_file"]

    nlu_data = rasa.shared.utils.io.read_file(default_nlu_data)

    _, response = await rasa_app.post(
        f"/model/test/intents?model={trained_nlu_model}",
        data=nlu_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == 200
    assert set(response.json().keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }

    _, response = await rasa_app.get("/status")
    assert previous_model_file == response.json()["model_file"]


async def test_predict(rasa_app: SanicASGITestClient):
    data = {
        "Events": {
            "value": [
                {"event": "action", "name": "action_listen"},
                {
                    "event": "user",
                    "text": "hello",
                    "parse_data": {
                        "entities": [],
                        "intent": {"confidence": 0.57, INTENT_NAME_KEY: "greet"},
                        "text": "hello",
                    },
                },
            ]
        }
    }
    _, response = await rasa_app.post(
        "/model/predict",
        json=data,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    content = response.json()
    assert response.status == 200
    assert "scores" in content
    assert "tracker" in content
    assert "policy" in content


@freeze_time("2018-01-01")
async def test_requesting_non_existent_tracker(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.get("/conversations/madeupid/tracker")
    content = response.json()
    assert response.status == 200
    assert content["paused"] is False
    assert content["slots"] == {"name": None}
    assert content["sender_id"] == "madeupid"
    assert content["events"] == [
        {
            "event": "action",
            "name": "action_session_start",
            "policy": None,
            "confidence": 1,
            "timestamp": 1514764800,
        },
        {"event": "session_started", "timestamp": 1514764800},
        {
            "event": "action",
            INTENT_NAME_KEY: "action_listen",
            "policy": None,
            "confidence": None,
            "timestamp": 1514764800,
        },
    ]
    assert content["latest_message"] == {
        "text": None,
        "intent": {},
        "entities": [],
        "message_id": None,
        "metadata": {},
    }


@pytest.mark.parametrize("event", test_events)
async def test_pushing_event(rasa_app: SanicASGITestClient, event: Event):
    sender_id = str(uuid.uuid1())
    conversation = f"/conversations/{sender_id}"

    serialized_event = event.as_dict()
    # Remove timestamp so that a new one is assigned on the server
    serialized_event.pop("timestamp")

    time_before_adding_events = time.time()
    _, response = await rasa_app.post(
        f"{conversation}/tracker/events",
        json=serialized_event,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    assert response.json() is not None
    assert response.status == 200

    _, tracker_response = await rasa_app.get(f"/conversations/{sender_id}/tracker")
    tracker = tracker_response.json()
    assert tracker is not None

    assert len(tracker.get("events")) == 4

    deserialized_events = [Event.from_parameters(event) for event in tracker["events"]]

    # there is an initial session start sequence at the beginning of the tracker
    assert deserialized_events[:3] == session_start_sequence

    assert deserialized_events[3] == event
    assert deserialized_events[3].timestamp > time_before_adding_events


async def test_push_multiple_events(rasa_app: SanicASGITestClient):
    conversation_id = str(uuid.uuid1())
    conversation = f"/conversations/{conversation_id}"

    events = [e.as_dict() for e in test_events]
    _, response = await rasa_app.post(
        f"{conversation}/tracker/events",
        json=events,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    assert response.json() is not None
    assert response.status == 200

    _, tracker_response = await rasa_app.get(
        f"/conversations/{conversation_id}/tracker"
    )
    tracker = tracker_response.json()
    assert tracker is not None

    # there is an initial session start sequence at the beginning
    assert [
        Event.from_parameters(event) for event in tracker.get("events")
    ] == session_start_sequence + test_events


@pytest.mark.parametrize(
    "params", ["?execute_side_effects=true&output_channel=callback", ""]
)
async def test_pushing_event_while_executing_side_effects(
    rasa_server: Sanic, params: Text
):
    input_channel = CallbackInput(EndpointConfig("https://example.com/callback"))
    channel.register([input_channel], rasa_server, "/webhooks/")
    rasa_app = rasa_server.asgi_client
    sender_id = str(uuid.uuid1())
    conversation = f"/conversations/{sender_id}"

    serialized_event = test_events[1].as_dict()

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/callback",
            repeat=True,
            headers={"Content-Type": "application/json"},
        )
        await rasa_app.post(
            f"{conversation}/tracker/events{params}",
            json=serialized_event,
            headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
        )

        r = latest_request(mocked, "post", "https://example.com/callback")

        if not params:
            assert r is None
        else:
            message_received = json_of_latest_request(r)
            assert message_received.get("recipient_id") == sender_id
            assert message_received.get("text") == serialized_event.get("text")


async def test_post_conversation_id_with_slash(rasa_app: SanicASGITestClient):
    conversation_id = str(uuid.uuid1())
    id_len = len(conversation_id) // 2
    conversation_id = conversation_id[:id_len] + "/+-_\\=" + conversation_id[id_len:]
    conversation = f"/conversations/{conversation_id}"

    events = [e.as_dict() for e in test_events]
    _, response = await rasa_app.post(
        f"{conversation}/tracker/events",
        json=events,
        headers={"Content-Type": "application/json"},
    )
    assert response.json() is not None
    assert response.status == 200

    _, tracker_response = await rasa_app.get(
        f"/conversations/{conversation_id}/tracker"
    )
    tracker = tracker_response.json()
    assert tracker is not None

    # there is a session start sequence at the start
    assert [
        Event.from_parameters(event) for event in tracker.get("events")
    ] == session_start_sequence + test_events


async def test_put_tracker(rasa_app: SanicASGITestClient):
    data = [event.as_dict() for event in test_events]
    _, response = await rasa_app.put(
        "/conversations/pushtracker/tracker/events",
        json=data,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    content = response.json()
    assert response.status == 200
    assert len(content["events"]) == len(test_events)
    assert content["sender_id"] == "pushtracker"

    _, tracker_response = await rasa_app.get("/conversations/pushtracker/tracker")
    tracker = tracker_response.json()
    assert tracker is not None
    evts = tracker.get("events")
    assert events.deserialise_events(evts) == test_events


async def test_sorted_predict(rasa_app: SanicASGITestClient):
    await _create_tracker_for_sender(rasa_app, "sortedpredict")

    _, response = await rasa_app.post("/conversations/sortedpredict/predict")
    scores = response.json()["scores"]
    sorted_scores = sorted(scores, key=lambda k: (-k["score"], k["action"]))
    assert scores == sorted_scores


async def _create_tracker_for_sender(app: SanicASGITestClient, sender_id: Text) -> None:
    data = [event.as_dict() for event in test_events[:3]]
    _, response = await app.put(
        f"/conversations/{sender_id}/tracker/events",
        json=data,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == 200


async def test_get_tracker_with_jwt(rasa_secured_app: SanicASGITestClient):
    # token generated with secret "core" and algorithm HS256
    # on https://jwt.io/

    # {"user": {"username": "testadmin", "role": "admin"}}
    jwt_header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJ1c2VyIjp7InVzZXJuYW1lIjoidGVzdGFkbWluIiwic"
        "m9sZSI6ImFkbWluIn19.NAQr0kbtSrY7d28XTqRzawq2u"
        "QRre7IWTuIDrCn5AIw"
    }
    _, response = await rasa_secured_app.get(
        "/conversations/testadmin/tracker", headers=jwt_header
    )
    assert response.status == 200

    _, response = await rasa_secured_app.get(
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
    _, response = await rasa_secured_app.get(
        "/conversations/testadmin/tracker", headers=jwt_header
    )
    assert response.status == 403

    _, response = await rasa_secured_app.get(
        "/conversations/testuser/tracker", headers=jwt_header
    )
    assert response.status == 200


def test_list_routes(default_agent: Agent):
    app = rasa.server.create_app(default_agent, auth_token=None)

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
        "trigger_intent",
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


async def test_unload_model_error(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.get("/status")
    assert response.status == 200
    assert "model_file" in response.json() and response.json()["model_file"] is not None

    _, response = await rasa_app.delete("/model")
    assert response.status == 204


async def test_get_domain(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.get(
        "/domain", headers={"accept": rasa.server.JSON_CONTENT_TYPE}
    )

    content = response.json()

    assert response.status == 200
    assert "config" in content
    assert "intents" in content
    assert "entities" in content
    assert "slots" in content
    assert "responses" in content
    assert "actions" in content


async def test_get_domain_invalid_accept_header(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.get("/domain")

    assert response.status == 406


async def test_load_model(rasa_app: SanicASGITestClient, trained_core_model: Text):
    _, response = await rasa_app.get("/status")

    assert response.status == 200
    assert "fingerprint" in response.json()

    old_fingerprint = response.json()["fingerprint"]

    data = {"model_file": trained_core_model}
    _, response = await rasa_app.put("/model", json=data)

    assert response.status == 204

    _, response = await rasa_app.get("/status")

    assert response.status == 200
    assert "fingerprint" in response.json()

    assert old_fingerprint != response.json()["fingerprint"]


async def test_load_model_from_model_server(
    rasa_app: SanicASGITestClient, trained_core_model: Text
):
    _, response = await rasa_app.get("/status")

    assert response.status == 200
    assert "fingerprint" in response.json()

    old_fingerprint = response.json()["fingerprint"]

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
            _, response = await rasa_app.put("/model", json=data)

            assert response.status == 204

            _, response = await rasa_app.get("/status")

            assert response.status == 200
            assert "fingerprint" in response.json()

            assert old_fingerprint != response.json()["fingerprint"]

    import rasa.core.jobs

    rasa.core.jobs.__scheduler = None


async def test_load_model_invalid_request_body(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.put("/model")

    assert response.status == 400


async def test_load_model_invalid_configuration(rasa_app: SanicASGITestClient):
    data = {"model_file": "some-random-path"}
    _, response = await rasa_app.put("/model", json=data)

    assert response.status == 400


async def test_execute(rasa_app: SanicASGITestClient):
    await _create_tracker_for_sender(rasa_app, "test_execute")

    data = {INTENT_NAME_KEY: "utter_greet"}
    _, response = await rasa_app.post("/conversations/test_execute/execute", json=data)

    assert response.status == 200

    parsed_content = response.json()
    assert parsed_content["tracker"]
    assert parsed_content["messages"]


async def test_execute_with_missing_action_name(rasa_app: SanicASGITestClient):
    test_sender = "test_execute_with_missing_action_name"
    await _create_tracker_for_sender(rasa_app, test_sender)

    data = {"wrong-key": "utter_greet"}
    _, response = await rasa_app.post(
        f"/conversations/{test_sender}/execute", json=data
    )

    assert response.status == 400


async def test_execute_with_not_existing_action(rasa_app: SanicASGITestClient):
    test_sender = "test_execute_with_not_existing_action"
    await _create_tracker_for_sender(rasa_app, test_sender)

    data = {"name": "ka[pa[opi[opj[oj[oija"}
    _, response = await rasa_app.post(
        f"/conversations/{test_sender}/execute", json=data
    )

    assert response.status == 500


async def test_trigger_intent(rasa_app: SanicASGITestClient):
    data = {INTENT_NAME_KEY: "greet"}
    _, response = await rasa_app.post(
        "/conversations/test_trigger/trigger_intent", json=data
    )

    assert response.status == 200

    parsed_content = response.json()
    assert parsed_content["tracker"]
    assert parsed_content["messages"]


async def test_trigger_intent_with_missing_intent_name(rasa_app: SanicASGITestClient):
    test_sender = "test_trigger_intent_with_missing_action_name"

    data = {"wrong-key": "greet"}
    _, response = await rasa_app.post(
        f"/conversations/{test_sender}/trigger_intent", json=data
    )

    assert response.status == 400


async def test_trigger_intent_with_not_existing_intent(rasa_app: SanicASGITestClient):
    test_sender = "test_trigger_intent_with_not_existing_intent"
    await _create_tracker_for_sender(rasa_app, test_sender)

    data = {INTENT_NAME_KEY: "ka[pa[opi[opj[oj[oija"}
    _, response = await rasa_app.post(
        f"/conversations/{test_sender}/trigger_intent", json=data
    )

    assert response.status == 404


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
    input_channels: List[Text], output_channel_to_use: Text, expected_channel: Type
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


@pytest.mark.parametrize(
    "conversation_events,until_time,fetch_all_sessions,expected",
    # conversation with one session
    [
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("hi", {"name": "greet"}),
                ActionExecuted("utter_greet"),
            ],
            None,
            True,
            """version: "2.0"
stories:
- story: some-conversation-ID
  steps:
  - intent: greet
    user: |-
      hi
  - action: utter_greet""",
        ),
        # conversation with multiple sessions
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("hi", {"name": "greet"}),
                ActionExecuted("utter_greet"),
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("bye bye", {"name": "goodbye"}),
                ActionExecuted("utter_goodbye"),
            ],
            None,
            True,
            """version: "2.0"
stories:
- story: some-conversation-ID, story 1
  steps:
  - intent: greet
    user: |-
      hi
  - action: utter_greet
- story: some-conversation-ID, story 2
  steps:
  - intent: goodbye
    user: |-
      bye bye
  - action: utter_goodbye""",
        ),
        # conversation with multiple sessions, but setting `all_sessions=false`
        # means only the last one is returned
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("hi", {"name": "greet"}),
                ActionExecuted("utter_greet"),
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("bye bye", {"name": "goodbye"}),
                ActionExecuted("utter_goodbye"),
            ],
            None,
            False,
            """version: "2.0"
stories:
- story: some-conversation-ID
  steps:
  - intent: goodbye
    user: |-
      bye bye
  - action: utter_goodbye""",
        ),
        # the default for `all_sessions` is `false` - this test checks that
        # only the latest session is returned in that case
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("hi", {"name": "greet"}),
                ActionExecuted("utter_greet"),
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("bye bye", {"name": "goodbye"}),
                ActionExecuted("utter_goodbye"),
            ],
            None,
            None,
            """version: "2.0"
stories:
- story: some-conversation-ID
  steps:
  - intent: goodbye
    user: |-
      bye bye
  - action: utter_goodbye""",
        ),
        # `until` parameter means only the first session is returned
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME, timestamp=1),
                SessionStarted(timestamp=2),
                UserUttered("hi", {"name": "greet"}, timestamp=3),
                ActionExecuted("utter_greet", timestamp=4),
                ActionExecuted(ACTION_SESSION_START_NAME, timestamp=5),
                SessionStarted(timestamp=6),
                UserUttered("bye bye", {"name": "goodbye"}, timestamp=7),
                ActionExecuted("utter_goodbye", timestamp=8),
            ],
            4,
            True,
            """version: "2.0"
stories:
- story: some-conversation-ID
  steps:
  - intent: greet
    user: |-
      hi
  - action: utter_greet""",
        ),
        # empty conversation
        ([], None, True, 'version: "2.0"'),
    ],
)
async def test_get_story(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    conversation_events: List[Event],
    until_time: Optional[float],
    fetch_all_sessions: Optional[bool],
    expected: Text,
):
    conversation_id = "some-conversation-ID"

    tracker_store = InMemoryTrackerStore(Domain.empty())
    tracker = DialogueStateTracker.from_events(conversation_id, conversation_events)

    tracker_store.save(tracker)

    monkeypatch.setattr(rasa_app.app.agent, "tracker_store", tracker_store)

    url = f"/conversations/{conversation_id}/story?"

    query = {}

    if fetch_all_sessions is not None:
        query["all_sessions"] = fetch_all_sessions

    if until_time is not None:
        query["until"] = until_time

    _, response = await rasa_app.get(url + urllib.parse.urlencode(query))

    assert response.status == 200
    assert response.content.decode().strip() == expected


async def test_get_story_does_not_update_conversation_session(
    rasa_app: SanicASGITestClient, monkeypatch: MonkeyPatch
):
    conversation_id = "some-conversation-ID"

    # domain with short session expiration time of one second
    domain = Domain.empty()
    domain.session_config = SessionConfig(
        session_expiration_time=1 / 60, carry_over_slots=True
    )

    monkeypatch.setattr(rasa_app.app.agent, "domain", domain)

    # conversation contains one session that has expired
    now = time.time()
    conversation_events = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=now - 10),
        SessionStarted(timestamp=now - 9),
        UserUttered("hi", {"name": "greet"}, timestamp=now - 8),
        ActionExecuted("utter_greet", timestamp=now - 7),
    ]

    tracker = DialogueStateTracker.from_events(conversation_id, conversation_events)

    # the conversation session has expired
    assert rasa_app.app.agent.create_processor()._has_session_expired(tracker)

    tracker_store = InMemoryTrackerStore(domain)

    tracker_store.save(tracker)

    monkeypatch.setattr(rasa_app.app.agent, "tracker_store", tracker_store)

    _, response = await rasa_app.get(f"/conversations/{conversation_id}/story")

    assert response.status == 200

    # expected story is returned
    assert (
        response.content.decode().strip()
        == """version: "2.0"
stories:
- story: some-conversation-ID
  steps:
  - intent: greet
    user: |-
      hi
  - action: utter_greet"""
    )

    # the tracker has the same number of events as were initially added
    assert len(tracker.events) == len(conversation_events)

    # the last event is still the same as before
    assert tracker.events[-1].timestamp == conversation_events[-1].timestamp


@pytest.mark.parametrize(
    "initial_tracker_events,events_to_append,expected_events",
    [
        (
            # the tracker is initially empty, and no events are appended
            # so we'll just expect the session start sequence with an `action_listen`
            [],
            [],
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
        ),
        (
            # the tracker is initially empty, and a user utterance is appended
            # we expect a tracker with a session start sequence and a user utterance
            [],
            [UserUttered("/greet", {"name": "greet", "confidence": 1.0})],
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered("/greet", {"name": "greet", "confidence": 1.0}),
            ],
        ),
        (
            # the tracker is initially empty, and a session start sequence is appended
            # we'll just expect the session start sequence
            [],
            [ActionExecuted(ACTION_SESSION_START_NAME), SessionStarted()],
            [ActionExecuted(ACTION_SESSION_START_NAME), SessionStarted()],
        ),
        (
            # the tracker already contains some events - we can simply append events
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered("/greet", {"name": "greet", "confidence": 1.0}),
            ],
            [ActionExecuted("utter_greet")],
            [
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered("/greet", {"name": "greet", "confidence": 1.0}),
                ActionExecuted("utter_greet"),
            ],
        ),
    ],
)
async def test_update_conversation_with_events(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    initial_tracker_events: List[Event],
    events_to_append: List[Event],
    expected_events: List[Event],
):
    conversation_id = "some-conversation-ID"
    domain = Domain.empty()
    tracker_store = InMemoryTrackerStore(domain)
    monkeypatch.setattr(rasa_app.app.agent, "tracker_store", tracker_store)

    if initial_tracker_events:
        tracker = DialogueStateTracker.from_events(
            conversation_id, initial_tracker_events
        )
        tracker_store.save(tracker)

    fetched_tracker = await rasa.server.update_conversation_with_events(
        conversation_id, rasa_app.app.agent.create_processor(), domain, events_to_append
    )

    assert list(fetched_tracker.events) == expected_events
