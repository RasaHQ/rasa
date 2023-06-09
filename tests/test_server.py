import asyncio
import json
import os
import textwrap
import time
import urllib.parse
import uuid
import sys
from argparse import Namespace
from http import HTTPStatus
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, List, Text, Tuple, Type, Generator, NoReturn, Dict, Optional
from unittest.mock import Mock, ANY

from _pytest.tmpdir import TempPathFactory
import pytest
from _pytest.monkeypatch import MonkeyPatch
from aioresponses import aioresponses
from freezegun import freeze_time
from unittest.mock import MagicMock
from ruamel.yaml import StringIO
from sanic import Sanic
from sanic_testing.testing import SanicASGITestClient

import rasa
import rasa.constants
import rasa.core.jobs
from rasa.engine.storage.local_model_storage import LocalModelStorage
import rasa.nlu
import rasa.server
import rasa.shared.constants
import rasa.shared.utils.io
import rasa.utils.io
from rasa.core import utils
from rasa.core.agent import Agent, load_agent
from rasa.core.channels import (
    channel,
    CollectingOutputChannel,
    RestInput,
    SlackInput,
    CallbackInput,
)
from rasa.core.channels.slack import SlackBot
from rasa.core.tracker_store import InMemoryTrackerStore
import rasa.nlu.test
from rasa.nlu.test import CVEvaluationResult
from rasa.shared.core import events
from rasa.shared.core.constants import (
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_LISTEN_NAME,
    REQUESTED_SLOT,
    SESSION_START_METADATA_SLOT,
)
from rasa.shared.core.domain import Domain, SessionConfig
from rasa.shared.core.events import (
    Event,
    Restarted,
    UserUttered,
    SlotSet,
    BotUttered,
    ActionExecuted,
    SessionStarted,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    INTENT_NAME_KEY,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    PREDICTED_CONFIDENCE_KEY,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.model_training import TrainingResult
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import (
    AsyncMock,
    with_assistant_id,
    with_assistant_ids,
    with_model_id,
    with_model_ids,
)
from tests.nlu.utilities import ResponseTest
from tests.utilities import json_of_latest_request, latest_request

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
def rasa_non_trained_app(rasa_non_trained_server: Sanic) -> SanicASGITestClient:
    return rasa_non_trained_server.asgi_client


@pytest.fixture
def rasa_app_nlu(rasa_nlu_server: Sanic) -> SanicASGITestClient:
    return rasa_nlu_server.asgi_client


@pytest.fixture
def rasa_app_core(rasa_core_server: Sanic) -> SanicASGITestClient:
    return rasa_core_server.asgi_client


@pytest.fixture
def rasa_secured_app(rasa_server_secured: Sanic) -> SanicASGITestClient:
    return rasa_server_secured.asgi_client


@pytest.fixture
def rasa_secured_app_asymmetric(
    rasa_server_secured_asymmetric: Sanic,
) -> SanicASGITestClient:
    return rasa_server_secured_asymmetric.asgi_client


@pytest.fixture
def rasa_non_trained_secured_app(
    rasa_non_trained_server_secured: Sanic,
) -> SanicASGITestClient:
    return rasa_non_trained_server_secured.asgi_client


@pytest.fixture()
async def tear_down_scheduler() -> Generator[None, None, None]:
    yield None
    rasa.core.jobs.__scheduler = None


async def test_root(rasa_non_trained_app: SanicASGITestClient):
    _, response = await rasa_non_trained_app.get("/")
    assert response.status == HTTPStatus.OK
    assert response.text.startswith("Hello from Rasa:")


async def test_root_without_enable_api(rasa_app_without_api: SanicASGITestClient):
    _, response = await rasa_app_without_api.get("/")
    assert response.status == HTTPStatus.OK
    assert response.text.startswith("Hello from Rasa:")


async def test_root_secured(rasa_non_trained_secured_app: SanicASGITestClient):
    _, response = await rasa_non_trained_secured_app.get("/")
    assert response.status == HTTPStatus.OK
    assert response.text.startswith("Hello from Rasa:")


async def test_version(rasa_non_trained_app: SanicASGITestClient):
    _, response = await rasa_non_trained_app.get("/version")
    content = response.json
    assert response.status == HTTPStatus.OK
    assert content.get("version") == rasa.__version__
    assert (
        content.get("minimum_compatible_version")
        == rasa.constants.MINIMUM_COMPATIBLE_VERSION
    )


async def test_status(rasa_app: SanicASGITestClient, trained_rasa_model: Text):
    _, response = await rasa_app.get("/status")
    model_file = response.json["model_file"]
    assert response.status == HTTPStatus.OK
    assert "model_id" in response.json
    assert model_file == Path(trained_rasa_model).name


async def test_status_nlu_only(
    rasa_app_nlu: SanicASGITestClient, trained_nlu_model: Text
):
    _, response = await rasa_app_nlu.get("/status")
    model_file = response.json["model_file"]
    assert response.status == HTTPStatus.OK
    assert "model_id" in response.json
    assert "model_file" in response.json
    assert model_file == Path(trained_nlu_model).name


async def test_status_secured(rasa_secured_app: SanicASGITestClient):
    _, response = await rasa_secured_app.get("/status")
    assert response.status == HTTPStatus.UNAUTHORIZED


async def test_status_not_ready_agent(rasa_app: SanicASGITestClient):
    rasa_app.sanic_app.ctx.agent = None
    _, response = await rasa_app.get("/status")
    assert response.status == HTTPStatus.CONFLICT


@pytest.fixture
def shared_statuses() -> DictProxy:
    return Manager().dict()


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
    rjs = response.json
    assert response.status == HTTPStatus.OK
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
    assert response.status == HTTPStatus.OK


async def test_parse_without_nlu_model(rasa_app_core: SanicASGITestClient):
    _, response = await rasa_app_core.post("/model/parse", json={"text": "hello"})
    assert response.status == HTTPStatus.OK

    rjs = response.json
    assert all(prop in rjs for prop in ["entities", "intent", "text"])


async def test_parse_on_invalid_emulation_mode(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.post(
        "/model/parse?emulation_mode=ANYTHING", json={"text": "hello"}
    )
    assert response.status == HTTPStatus.BAD_REQUEST


async def test_train_nlu_success(
    rasa_app: SanicASGITestClient,
    stack_config_path: Text,
    nlu_data_path: Text,
    domain_path: Text,
    tmp_path_factory: TempPathFactory,
):
    domain_data = rasa.shared.utils.io.read_yaml_file(domain_path)
    config_data = rasa.shared.utils.io.read_yaml_file(stack_config_path)
    nlu_data = rasa.shared.utils.io.read_yaml_file(nlu_data_path)

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
    assert response.status == HTTPStatus.OK

    # save model to temporary file
    model_path = str(Path(tmp_path_factory.mktemp("model_dir")) / "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    storage_path = tmp_path_factory.mktemp("storage_path")
    model_storage, model_metadata = LocalModelStorage.from_model_archive(
        storage_path, model_path
    )
    assert model_metadata.model_id


async def test_train_core_success_with(
    rasa_app: SanicASGITestClient,
    stack_config_path: Text,
    stories_path: Text,
    domain_path: Text,
    tmp_path_factory: TempPathFactory,
):
    payload = f"""
{Path(domain_path).read_text()}
{Path(stack_config_path).read_text()}
{Path(stories_path).read_text()}
    """

    _, response = await rasa_app.post(
        "/model/train",
        data=payload,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )
    assert response.status == HTTPStatus.OK

    # save model to temporary file
    model_path = str(Path(tmp_path_factory.mktemp("model_dir")) / "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response.body)

    storage_path = tmp_path_factory.mktemp("storage_path")
    model_storage, model_metadata = LocalModelStorage.from_model_archive(
        storage_path, model_path
    )
    assert model_metadata.model_id


async def test_train_with_retrieval_events_success(
    rasa_app: SanicASGITestClient,
    stack_config_path: Text,
    tmp_path_factory: TempPathFactory,
):
    payload = {}

    tmp_path = tmp_path_factory.mktemp("tmp")

    for file in [
        "data/test_domains/default_retrieval_intents.yml",
        stack_config_path,
        "data/test_yaml_stories/stories_retrieval_intents.yml",
        "data/test_responses/default.yml",
        "data/test/stories_default_retrieval_intents.yml",
    ]:
        # Read in as dictionaries to avoid that keys, which are specified in
        # multiple files (such as 'version'), clash.
        content = rasa.shared.utils.io.read_yaml_file(file)
        payload.update(content)

        concatenated_payload_file = tmp_path / "concatenated.yml"
        rasa.shared.utils.io.write_yaml(payload, concatenated_payload_file)

        payload_as_yaml = concatenated_payload_file.read_text()

    # it usually takes a bit longer on windows so we're going to double the timeout
    timeout = 60 * 10 if sys.platform == "win32" else 60 * 5

    _, response = await rasa_app.post(
        "/model/train",
        data=payload_as_yaml,
        timeout=timeout,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )
    assert response.status == HTTPStatus.OK

    assert_trained_model(response.body, tmp_path_factory)


def assert_trained_model(
    response_body: bytes, tmp_path_factory: TempPathFactory
) -> None:
    # save model to temporary file

    model_path = str(Path(tmp_path_factory.mktemp("model_dir")) / "model.tar.gz")
    with open(model_path, "wb") as f:
        f.write(response_body)

    storage_path = tmp_path_factory.mktemp("storage_path")
    model_storage, model_metadata = LocalModelStorage.from_model_archive(
        storage_path, model_path
    )
    assert model_metadata.model_id


async def test_train_with_yaml(
    rasa_app: SanicASGITestClient, tmp_path_factory: TempPathFactory
):
    training_data = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

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

recipe: default.v1
language: en
assistant_id: placeholder_default

policies:
- name: RulePolicy

pipeline:
  - name: KeywordIntentClassifier
"""
    _, response = await rasa_app.post(
        "/model/train",
        data=training_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    assert_trained_model(response.body, tmp_path_factory)


@pytest.mark.parametrize(
    "params", [{}, {"augmentation": 20, "num_threads": 2, "force_training": True}]
)
async def test_train_with_yaml_with_params(
    monkeypatch: MonkeyPatch,
    rasa_non_trained_app: SanicASGITestClient,
    tmp_path: Path,
    params: Dict,
):
    fake_model = Path(tmp_path) / "fake_model.tar.gz"
    fake_model.touch()
    fake_model_path = str(fake_model)
    mock_train = Mock(return_value=TrainingResult(model=fake_model_path))
    monkeypatch.setattr(rasa.model_training, "train", mock_train)

    training_data = """
stories: []
rules: []
intents: []
nlu: []
responses: {}
recipe: default.v1
language: en
policies: []
pipeline: []
"""
    _, response = await rasa_non_trained_app.post(
        "/model/train",
        data=training_data,
        params=params,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    assert mock_train.call_count == 1
    args, kwargs = mock_train.call_args_list[0]
    assert kwargs["core_additional_arguments"]["augmentation_factor"] == params.get(
        "augmentation", 50
    )
    assert kwargs["nlu_additional_arguments"]["num_threads"] == params.get(
        "num_threads", 1
    )
    assert kwargs["force_training"] == params.get("force_training", False)


async def test_train_with_invalid_yaml(rasa_non_trained_app: SanicASGITestClient):
    invalid_yaml = """
rules:
rule my rule
"""

    _, response = await rasa_non_trained_app.post(
        "/model/train",
        data=invalid_yaml,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )
    assert response.status == HTTPStatus.BAD_REQUEST


@pytest.mark.parametrize(
    "headers, expected",
    [({}, False), ({"force_training": False}, False), ({"force_training": True}, True)],
)
def test_training_payload_from_yaml_force_training(
    headers: Dict, expected: bool, tmp_path: Path
):
    request = Mock()
    request.body = b""
    request.args = headers

    payload = rasa.server._training_payload_from_yaml(request, tmp_path)
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
    headers: Dict, expected: Text, tmp_path: Path
):
    request = Mock()
    request.body = b""
    request.args = headers

    payload = rasa.server._training_payload_from_yaml(request, tmp_path)
    assert payload.get("output")
    assert payload.get("output") == expected


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
def test_nlu_training_payload_from_json(headers: Dict, expected: Text, tmp_path: Path):
    request = Mock()
    request.json = {"rasa_nlu_data": {"common_examples": []}}
    request.args = headers

    payload = rasa.server._nlu_training_payload_from_json(request, tmp_path)
    assert payload.get("output")
    assert payload.get("output") == expected


async def test_evaluate_stories(rasa_app: SanicASGITestClient, stories_path: Text):
    stories = rasa.shared.utils.io.read_file(stories_path)

    _, response = await rasa_app.post(
        "/model/test/stories",
        data=stories,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK

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


async def test_evaluate_stories_not_ready_agent(
    rasa_non_trained_app: SanicASGITestClient, stories_path: Text
):
    stories = rasa.shared.utils.io.read_file(stories_path)

    _, response = await rasa_non_trained_app.post("/model/test/stories", data=stories)

    assert response.status == HTTPStatus.CONFLICT


async def test_evaluate_stories_end_to_end(
    rasa_app: SanicASGITestClient, end_to_end_story_path: Text
):
    stories = rasa.shared.utils.io.read_file(end_to_end_story_path)

    _, response = await rasa_app.post(
        "/model/test/stories?e2e=true",
        data=stories,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
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
    assert js["actions"] != []
    assert set(js["actions"][0].keys()) == {
        "action",
        "predicted",
        "confidence",
        "policy",
    }


async def test_add_message(rasa_app: SanicASGITestClient):

    conversation_id = "test_add_message_test_id"

    _, response = await rasa_app.get(f"/conversations/{conversation_id}/tracker")
    previous_num_events = len(response.json["events"])

    unique_text = f"test_add_message_text_{time.time()}"
    unique_slot_value = f"test_add_message_entity_{time.time()}"
    data = {
        "text": unique_text,
        "sender": "user",  # must be "user"
        "parse_data": {
            "text": unique_text,  # this is what is used for "latest_message"
            "intent": {PREDICTED_CONFIDENCE_KEY: 0.57, INTENT_NAME_KEY: "greet"},
            "entities": [
                {
                    ENTITY_ATTRIBUTE_TYPE: "name",
                    ENTITY_ATTRIBUTE_VALUE: unique_slot_value,
                }
            ],
        },
    }
    _, response = await rasa_app.post(
        f"/conversations/{conversation_id}/messages",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
        json=data,
    )
    assert response.json["latest_message"]["text"] == unique_text

    _, response = await rasa_app.get(f"/conversations/{conversation_id}/tracker")
    updated_events = response.json["events"]
    assert len(updated_events) == previous_num_events + 2
    assert updated_events[-2]["text"] == unique_text
    assert updated_events[-1]["event"] == "slot"
    assert updated_events[-1]["value"] == unique_slot_value


async def test_evaluate_intent(rasa_app: SanicASGITestClient, nlu_data_path: Text):
    nlu_data = rasa.shared.utils.io.read_file(nlu_data_path)

    _, response = await rasa_app.post(
        "/model/test/intents",
        data=nlu_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    assert set(response.json.keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }


async def test_evaluate_invalid_intent_model_file(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.post(
        "/model/test/intents?model=invalid.tar.gz",
        json={},
        headers={"Content-type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR


async def test_evaluate_intent_without_body(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.post(
        "/model/test/intents", headers={"Content-type": rasa.server.YAML_CONTENT_TYPE}
    )

    assert response.status == HTTPStatus.BAD_REQUEST


async def test_evaluate_intent_on_just_nlu_model(
    rasa_app_nlu: SanicASGITestClient, nlu_data_path: Text
):
    nlu_data = rasa.shared.utils.io.read_file(nlu_data_path)

    _, response = await rasa_app_nlu.post(
        "/model/test/intents",
        data=nlu_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    assert set(response.json.keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }


async def test_evaluate_intent_with_model_param(
    rasa_app: SanicASGITestClient, trained_nlu_model: Text, nlu_data_path: Text
):
    _, response = await rasa_app.get("/status")
    previous_model_file = response.json["model_file"]

    nlu_data = rasa.shared.utils.io.read_file(nlu_data_path)

    _, response = await rasa_app.post(
        f"/model/test/intents?model={trained_nlu_model}",
        data=nlu_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    assert set(response.json.keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }

    _, response = await rasa_app.get("/status")
    assert previous_model_file == response.json["model_file"]


async def test_evaluate_intent_with_model_server(
    rasa_app: SanicASGITestClient,
    trained_rasa_model: Text,
    nlu_data_path: Text,
    tear_down_scheduler: None,
):
    production_model_server_url = (
        "https://example.com/webhooks/actions?model=production"
    )
    test_model_server_url = "https://example.com/webhooks/actions?model=test"

    nlu_data = rasa.shared.utils.io.read_file(nlu_data_path)

    with aioresponses() as mocked:
        # Mock retrieving the production model from the model server
        mocked.get(
            production_model_server_url,
            body=Path(trained_rasa_model).read_bytes(),
            headers={"ETag": "production", "filename": "prod_model.tar.gz"},
        )
        # Mock retrieving the test model from the model server
        mocked.get(
            test_model_server_url,
            body=Path(trained_rasa_model).read_bytes(),
            headers={"ETag": "test", "filename": "test_model.tar.gz"},
        )

        agent_with_model_server = await load_agent(
            model_server=EndpointConfig(production_model_server_url)
        )
        rasa_app.sanic_app.ctx.agent = agent_with_model_server

        _, response = await rasa_app.post(
            f"/model/test/intents?model={test_model_server_url}",
            data=nlu_data,
            headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
        )

    assert response.status == HTTPStatus.OK
    assert set(response.json.keys()) == {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }

    production_model_server = rasa_app.sanic_app.ctx.agent.model_server
    # Assert that the model server URL for the test didn't override the production
    # model server URL
    assert production_model_server.url == production_model_server_url
    # Assert the tests didn't break pulling the models
    assert production_model_server.kwargs.get("wait_time_between_pulls") != 0


async def test_cross_validation(
    rasa_non_trained_app: SanicASGITestClient,
    nlu_data_path: Text,
    stack_config_path: Text,
):
    nlu_data = Path(nlu_data_path).read_text()
    config = Path(stack_config_path).read_text()
    payload = f"{nlu_data}\n{config}"

    _, response = await rasa_non_trained_app.post(
        "/model/test/intents",
        data=payload,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
        params={"cross_validation_folds": 3},
    )

    assert response.status == HTTPStatus.OK
    response_body = response.json
    for required_key in {
        "intent_evaluation",
        "entity_evaluation",
        "response_selection_evaluation",
    }:
        assert required_key in response_body

        details = response_body[required_key]
        assert all(
            key in details for key in ["precision", "f1_score", "report", "errors"]
        )


async def test_cross_validation_with_callback_success(
    rasa_non_trained_app: SanicASGITestClient,
    nlu_data_path: Text,
    monkeypatch: MonkeyPatch,
    stack_config_path: Text,
):
    nlu_data = Path(nlu_data_path).read_text()
    config = Path(stack_config_path).read_text()
    payload = f"{nlu_data}\n{config}"

    callback_url = "https://example.com/webhooks/actions"
    with aioresponses() as mocked:
        mocked.post(callback_url, payload={})

        mocked_cross_validation = AsyncMock(
            return_value=(
                CVEvaluationResult({}, {}, {}),
                CVEvaluationResult({}, {}, {}),
                CVEvaluationResult({}, {}, {}),
            )
        )
        monkeypatch.setattr(
            rasa.nlu.test,
            rasa.nlu.test.cross_validate.__name__,
            mocked_cross_validation,
        )

        _, response = await rasa_non_trained_app.post(
            "/model/test/intents",
            data=payload,
            headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
            params={"cross_validation_folds": 3, "callback_url": callback_url},
        )

        assert response.status == HTTPStatus.NO_CONTENT

        # Sleep to give event loop time to process things in the background
        await asyncio.sleep(1)

        mocked_cross_validation.assert_called_once()

        last_request = latest_request(mocked, "POST", callback_url)
        assert last_request

        content = last_request[0].kwargs["data"]
        response_body = json.loads(content)
        for required_key in {
            "intent_evaluation",
            "entity_evaluation",
            "response_selection_evaluation",
        }:
            assert required_key in response_body

            details = response_body[required_key]
            assert all(
                key in details for key in ["precision", "f1_score", "report", "errors"]
            )


@pytest.mark.flaky
async def test_cross_validation_with_callback_error(
    rasa_non_trained_app: SanicASGITestClient,
    nlu_data_path: Text,
    monkeypatch: MonkeyPatch,
    stack_config_path: Text,
):
    nlu_data = Path(nlu_data_path).read_text()
    config = Path(stack_config_path).read_text()
    payload = f"{nlu_data}\n{config}"

    monkeypatch.setattr(
        rasa.nlu.test,
        rasa.nlu.test.cross_validate.__name__,
        Mock(side_effect=ValueError()),
    )

    callback_url = "https://example.com/webhooks/actions"
    with aioresponses() as mocked:
        mocked.post(callback_url, payload={})

        _, response = await rasa_non_trained_app.post(
            "/model/test/intents",
            data=payload,
            headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
            params={"cross_validation_folds": 3, "callback_url": callback_url},
        )

        assert response.status == HTTPStatus.NO_CONTENT

        await asyncio.sleep(1)

        last_request = latest_request(mocked, "POST", callback_url)
        assert last_request

        content = last_request[0].kwargs["json"]
        assert content["code"] == HTTPStatus.INTERNAL_SERVER_ERROR


async def test_callback_unexpected_error(
    rasa_non_trained_app: SanicASGITestClient,
    nlu_data_path: Text,
    monkeypatch: MonkeyPatch,
    stack_config_path: Text,
):
    nlu_data = Path(nlu_data_path).read_text()
    config = Path(stack_config_path).read_text()
    payload = f"{nlu_data}\n{config}"

    async def raiseUnexpectedError() -> NoReturn:
        raise ValueError()

    monkeypatch.setattr(
        rasa.server,
        rasa.server._training_payload_from_yaml.__name__,
        Mock(side_effect=ValueError()),
    )

    callback_url = "https://example.com/webhooks/actions"
    with aioresponses() as mocked:
        mocked.post(callback_url, payload={})

        _, response = await rasa_non_trained_app.post(
            "/model/test/intents",
            data=payload,
            headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
            params={"cross_validation_folds": 3, "callback_url": callback_url},
        )

        assert response.status == HTTPStatus.NO_CONTENT

        await asyncio.sleep(1)

        last_request = latest_request(mocked, "POST", callback_url)
        assert last_request

        content = last_request[0].kwargs["json"]
        assert content["code"] == HTTPStatus.INTERNAL_SERVER_ERROR


async def test_predict(rasa_app: SanicASGITestClient):
    data = [
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

    _, response = await rasa_app.post(
        "/model/predict",
        json=data,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    content = response.json
    assert response.status == HTTPStatus.OK
    assert "scores" in content
    assert "tracker" in content
    assert "policy" in content


async def test_predict_invalid_entities_format(rasa_app: SanicASGITestClient):
    data = [
        {"event": "action", "name": "action_listen"},
        {
            "event": "user",
            "text": "hello",
            "parse_data": {
                "entities": {},
                "intent": {"confidence": 0.57, INTENT_NAME_KEY: "greet"},
                "text": "hello",
            },
        },
    ]

    _, response = await rasa_app.post(
        "/model/predict",
        json=data,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    assert response.status == HTTPStatus.BAD_REQUEST


async def test_predict_empty_request_body(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.post(
        "/model/predict", headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE}
    )
    assert response.status == HTTPStatus.BAD_REQUEST


async def test_append_events_empty_request_body(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.post(
        "/conversations/testid/tracker/events",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    assert response.status == HTTPStatus.BAD_REQUEST


async def test_replace_events_empty_request_body(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.put(
        "/conversations/testid/tracker/events",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    assert response.status == HTTPStatus.BAD_REQUEST


@freeze_time("2018-01-01")
async def test_requesting_non_existent_tracker(rasa_app: SanicASGITestClient):
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    assistant_id = rasa_app.sanic_app.ctx.agent.processor.model_metadata.assistant_id
    _, response = await rasa_app.get("/conversations/madeupid/tracker")
    content = response.json
    assert response.status == HTTPStatus.OK
    assert content["paused"] is False
    assert content["slots"] == {
        "name": None,
        REQUESTED_SLOT: None,
        SESSION_START_METADATA_SLOT: None,
    }
    assert content["sender_id"] == "madeupid"
    assert content["events"] == [
        {
            "event": "action",
            "name": "action_session_start",
            "policy": None,
            "confidence": 1,
            "timestamp": 1514764800,
            "action_text": None,
            "hide_rule_turn": False,
            "metadata": {"assistant_id": assistant_id, "model_id": model_id},
        },
        {
            "event": "session_started",
            "timestamp": 1514764800,
            "metadata": {"assistant_id": assistant_id, "model_id": model_id},
        },
        {
            "event": "action",
            INTENT_NAME_KEY: "action_listen",
            "policy": None,
            "confidence": None,
            "timestamp": 1514764800,
            "action_text": None,
            "hide_rule_turn": False,
            "metadata": {"assistant_id": assistant_id, "model_id": model_id},
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
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    assistant_id = rasa_app.sanic_app.ctx.agent.processor.model_metadata.assistant_id
    sender_id = str(uuid.uuid1())
    conversation = f"/conversations/{sender_id}"

    serialized_event = event.as_dict()
    # Remove timestamp so that a new one is assigned on the server
    serialized_event.pop("timestamp")

    time_before_adding_events = time.time()
    # Wait a bit so that the server-generated timestamp is strictly greater
    # than time_before_adding_events
    time.sleep(0.01)
    _, response = await rasa_app.post(
        f"{conversation}/tracker/events",
        json=serialized_event,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    assert response.json is not None
    assert response.status == HTTPStatus.OK

    _, tracker_response = await rasa_app.get(f"/conversations/{sender_id}/tracker")
    tracker = tracker_response.json
    assert tracker is not None

    assert len(tracker.get("events")) == 4

    deserialized_events = [Event.from_parameters(event) for event in tracker["events"]]

    # there is an initial session start sequence at the beginning of the tracker

    assert deserialized_events[:3] == with_assistant_ids(
        with_model_ids(session_start_sequence, model_id), assistant_id
    )

    assert deserialized_events[3] == with_assistant_id(
        with_model_id(event, model_id), assistant_id
    )
    assert deserialized_events[3].timestamp > time_before_adding_events


async def test_pushing_event_with_existing_model_id(rasa_app: SanicASGITestClient):
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    assistant_id = rasa_app.sanic_app.ctx.agent.processor.model_metadata.assistant_id
    sender_id = str(uuid.uuid1())
    conversation = f"/conversations/{sender_id}"

    existing_model_id = "some_old_id"
    assert existing_model_id != model_id
    event = with_assistant_id(
        with_model_id(BotUttered("hello!"), existing_model_id), assistant_id
    )
    serialized_event = event.as_dict()

    # Wait a bit so that the server-generated timestamp is strictly greater
    # than time_before_adding_events
    _, response = await rasa_app.post(
        f"{conversation}/tracker/events",
        json=serialized_event,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    _, tracker_response = await rasa_app.get(f"/conversations/{sender_id}/tracker")
    tracker = tracker_response.json

    deserialized_events = [Event.from_parameters(event) for event in tracker["events"]]

    # there is an initial session start sequence at the beginning of the tracker
    received_event = deserialized_events[3]
    assert received_event == with_assistant_id(
        with_model_id(event, existing_model_id), assistant_id
    )


async def test_push_multiple_events(rasa_app: SanicASGITestClient):
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    assistant_id = rasa_app.sanic_app.ctx.agent.processor.model_metadata.assistant_id
    conversation_id = str(uuid.uuid1())
    conversation = f"/conversations/{conversation_id}"

    events = [e.as_dict() for e in test_events]
    _, response = await rasa_app.post(
        f"{conversation}/tracker/events",
        json=events,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    assert response.json is not None
    assert response.status == HTTPStatus.OK

    _, tracker_response = await rasa_app.get(
        f"/conversations/{conversation_id}/tracker"
    )
    tracker = tracker_response.json
    assert tracker is not None

    # there is an initial session start sequence at the beginning
    assert [
        Event.from_parameters(event) for event in tracker.get("events")
    ] == with_assistant_ids(
        with_model_ids(session_start_sequence + test_events, model_id), assistant_id
    )


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
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    assistant_id = rasa_app.sanic_app.ctx.agent.processor.model_metadata.assistant_id
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
    assert response.json is not None
    assert response.status == HTTPStatus.OK

    _, tracker_response = await rasa_app.get(
        f"/conversations/{conversation_id}/tracker"
    )
    tracker = tracker_response.json
    assert tracker is not None

    # there is a session start sequence at the start
    assert [
        Event.from_parameters(event) for event in tracker.get("events")
    ] == with_assistant_ids(
        with_model_ids(session_start_sequence + test_events, model_id), assistant_id
    )


async def test_put_tracker(rasa_app: SanicASGITestClient):
    data = [event.as_dict() for event in test_events]
    _, response = await rasa_app.put(
        "/conversations/pushtracker/tracker/events",
        json=data,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )
    content = response.json
    assert response.status == HTTPStatus.OK
    assert len(content["events"]) == len(test_events)
    assert content["sender_id"] == "pushtracker"

    _, tracker_response = await rasa_app.get("/conversations/pushtracker/tracker")
    tracker = tracker_response.json
    assert tracker is not None
    evts = tracker.get("events")
    assert events.deserialise_events(evts) == test_events


async def test_predict_without_conversation_id(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.post("/conversations/non_existent_id/predict")

    assert response.status == HTTPStatus.NOT_FOUND
    assert response.json["message"] == "Conversation ID not found."


async def test_sorted_predict(rasa_app: SanicASGITestClient):
    await _create_tracker_for_sender(rasa_app, "sortedpredict")

    _, response = await rasa_app.post("/conversations/sortedpredict/predict")
    scores = response.json["scores"]
    sorted_scores = sorted(scores, key=lambda k: (-k["score"], k["action"]))
    assert scores == sorted_scores


async def _create_tracker_for_sender(app: SanicASGITestClient, sender_id: Text) -> None:
    data = [event.as_dict() for event in test_events[:3]]
    _, response = await app.put(
        f"/conversations/{sender_id}/tracker/events",
        json=data,
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK


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
    assert response.status == HTTPStatus.OK

    _, response = await rasa_secured_app.get(
        "/conversations/testuser/tracker", headers=jwt_header
    )
    assert response.status == HTTPStatus.OK

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
    assert response.status == HTTPStatus.FORBIDDEN

    _, response = await rasa_secured_app.get(
        "/conversations/testuser/tracker", headers=jwt_header
    )
    assert response.status == HTTPStatus.OK


async def test_get_tracker_with_asymmetric_jwt(
    rasa_secured_app_asymmetric: SanicASGITestClient,
    encoded_jwt: Text,
) -> None:
    jwt_header = {"Authorization": f"Bearer {encoded_jwt}"}
    _, response = await rasa_secured_app_asymmetric.get(
        "/conversations/myuser/tracker", headers=jwt_header
    )
    assert response.status == HTTPStatus.OK

    _, response = await rasa_secured_app_asymmetric.get(
        "/conversations/testuser/tracker", headers=jwt_header
    )
    assert response.status == HTTPStatus.OK


def test_list_routes(empty_agent: Agent):
    app = rasa.server.create_app(empty_agent, auth_token=None)

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
    assert response.status == HTTPStatus.OK
    assert "model_file" in response.json and response.json["model_file"] is not None

    _, response = await rasa_app.delete("/model")
    assert response.status == HTTPStatus.NO_CONTENT


async def test_get_domain(rasa_app: SanicASGITestClient, domain_path: Text):
    _, response = await rasa_app.get(
        "/domain", headers={"accept": rasa.server.JSON_CONTENT_TYPE}
    )

    content = response.json

    assert response.status == HTTPStatus.OK
    # assert only keys in `domain_path` fixture
    original_domain_dict = Domain.load(domain_path).as_dict()
    for key in original_domain_dict.keys():
        assert key in content


async def test_get_domain_invalid_accept_header(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.get("/domain")

    assert response.status == HTTPStatus.NOT_ACCEPTABLE


async def test_load_model(rasa_app: SanicASGITestClient, trained_core_model: Text):
    _, response = await rasa_app.get("/status")

    assert response.status == HTTPStatus.OK
    assert "model_id" in response.json

    old_model_id = response.json["model_id"]

    data = {"model_file": trained_core_model}
    _, response = await rasa_app.put("/model", json=data)

    assert response.status == HTTPStatus.NO_CONTENT

    _, response = await rasa_app.get("/status")

    assert response.status == HTTPStatus.OK
    assert "model_id" in response.json

    assert old_model_id != response.json["model_id"]


async def test_load_model_from_model_server(
    rasa_app: SanicASGITestClient, trained_core_model: Text, tear_down_scheduler: None
):
    _, response = await rasa_app.get("/status")

    assert response.status == HTTPStatus.OK
    assert "model_id" in response.json

    old_model_id = response.json["model_id"]

    endpoint = EndpointConfig("https://example.com/model/trained_core_model")
    with open(trained_core_model, "rb") as f:
        with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
            headers = {}
            fs = os.fstat(f.fileno())
            headers["Content-Length"] = str(fs[6])
            mocked.get(
                "https://example.com/model/trained_core_model",
                content_type="application/x-tar",
                headers={
                    "filename": "some_model_name.tar.gz",
                    "ETag": "new_fingerprint",
                },
                body=f.read(),
            )
            data = {"model_server": {"url": endpoint.url}}
            _, response = await rasa_app.put("/model", json=data)

            assert response.status == HTTPStatus.NO_CONTENT

            _, response = await rasa_app.get("/status")

            assert response.status == HTTPStatus.OK
            assert "model_id" in response.json

            assert old_model_id != response.json["model_id"]


async def test_load_model_invalid_request_body(
    rasa_non_trained_app: SanicASGITestClient,
):
    _, response = await rasa_non_trained_app.put("/model")

    assert response.status == HTTPStatus.BAD_REQUEST


async def test_load_model_invalid_configuration(
    rasa_non_trained_app: SanicASGITestClient,
):
    data = {"model_file": "some-random-path"}
    _, response = await rasa_non_trained_app.put("/model", json=data)

    assert response.status == HTTPStatus.BAD_REQUEST


async def test_execute(rasa_app: SanicASGITestClient):
    await _create_tracker_for_sender(rasa_app, "test_execute")

    data = {INTENT_NAME_KEY: "utter_greet"}
    _, response = await rasa_app.post("/conversations/test_execute/execute", json=data)

    assert response.status == HTTPStatus.OK

    parsed_content = response.json
    assert parsed_content["tracker"]
    assert parsed_content["messages"]


async def test_execute_without_conversation_id(rasa_app: SanicASGITestClient):
    data = {INTENT_NAME_KEY: "utter_greet"}
    _, response = await rasa_app.post(
        "/conversations/non_existent_id/execute", json=data
    )

    assert response.status == HTTPStatus.NOT_FOUND
    assert response.json["message"] == "Conversation ID not found."


async def test_execute_with_missing_action_name(rasa_app: SanicASGITestClient):
    test_sender = "test_execute_with_missing_action_name"
    await _create_tracker_for_sender(rasa_app, test_sender)

    data = {"wrong-key": "utter_greet"}
    _, response = await rasa_app.post(
        f"/conversations/{test_sender}/execute", json=data
    )

    assert response.status == HTTPStatus.BAD_REQUEST


async def test_execute_with_not_existing_action(rasa_app: SanicASGITestClient):
    test_sender = "test_execute_with_not_existing_action"
    await _create_tracker_for_sender(rasa_app, test_sender)

    data = {"name": "ka[pa[opi[opj[oj[oija"}
    _, response = await rasa_app.post(
        f"/conversations/{test_sender}/execute", json=data
    )

    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR


async def test_trigger_intent(rasa_app: SanicASGITestClient):
    data = {INTENT_NAME_KEY: "greet"}
    _, response = await rasa_app.post(
        "/conversations/test_trigger/trigger_intent", json=data
    )

    assert response.status == HTTPStatus.OK

    parsed_content = response.json
    assert parsed_content["tracker"]
    assert parsed_content["messages"]


async def test_trigger_intent_with_entity(rasa_app: SanicASGITestClient):
    entity_name = "name"
    entity_value = "Sara"
    data = {INTENT_NAME_KEY: "greet", "entities": {entity_name: entity_value}}
    _, response = await rasa_app.post(
        "/conversations/test_trigger/trigger_intent", json=data
    )

    assert response.status == HTTPStatus.OK

    parsed_content = response.json
    last_slot_set_event = [
        event
        for event in parsed_content["tracker"]["events"]
        if event["event"] == "slot"
    ][-1]

    assert parsed_content["tracker"]
    assert parsed_content["messages"]
    assert last_slot_set_event["name"] == entity_name
    assert last_slot_set_event["value"] == entity_value


async def test_trigger_intent_with_missing_intent_name(rasa_app: SanicASGITestClient):
    test_sender = "test_trigger_intent_with_missing_action_name"

    data = {"wrong-key": "greet"}
    _, response = await rasa_app.post(
        f"/conversations/{test_sender}/trigger_intent", json=data
    )

    assert response.status == HTTPStatus.BAD_REQUEST


async def test_trigger_intent_with_not_existing_intent(rasa_app: SanicASGITestClient):
    test_sender = "test_trigger_intent_with_not_existing_intent"
    await _create_tracker_for_sender(rasa_app, test_sender)

    data = {INTENT_NAME_KEY: "ka[pa[opi[opj[oj[oija"}
    _, response = await rasa_app.post(
        f"/conversations/{test_sender}/trigger_intent", json=data
    )

    assert response.status == HTTPStatus.NOT_FOUND


@pytest.mark.parametrize(
    "input_channels, output_channel_to_use, expected_channel",
    [
        (None, "slack", CollectingOutputChannel),
        ([], None, CollectingOutputChannel),
        ([RestInput()], "slack", CollectingOutputChannel),
        ([RestInput()], "rest", CollectingOutputChannel),
        (
            [RestInput(), SlackInput("test", slack_signing_secret="foobar")],
            "slack",
            SlackBot,
        ),
    ],
)
def test_get_output_channel(
    input_channels: List[Text], output_channel_to_use: Text, expected_channel: Type
):
    request = MagicMock()
    app = MagicMock(ctx=Namespace())
    app.ctx.input_channels = input_channels
    request.app = app
    request.args = {"output_channel": output_channel_to_use}

    actual = rasa.server._get_output_channel(request, None)

    assert isinstance(actual, expected_channel)


@pytest.mark.parametrize(
    "input_channels, expected_channel",
    [
        ([], CollectingOutputChannel),
        ([RestInput()], CollectingOutputChannel),
        ([RestInput(), SlackInput("test", slack_signing_secret="foobar")], SlackBot),
    ],
)
def test_get_latest_output_channel(input_channels: List[Text], expected_channel: Type):
    request = MagicMock()
    app = MagicMock(ctx=Namespace())
    app.ctx.input_channels = input_channels
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
        ctx = Namespace()
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
            f"""version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
            f"""version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
            f"""version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
            f"""version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
            f"""version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
stories:
- story: some-conversation-ID
  steps:
  - intent: greet
    user: |-
      hi
  - action: utter_greet""",
        ),
        # empty conversation
        ([], None, True, f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"'),
        # Conversation with slot
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                UserUttered("hi", {"name": "greet"}),
                ActionExecuted("utter_greet"),
                SlotSet(REQUESTED_SLOT, "some value"),
            ],
            None,
            True,
            f"""version: "{rasa.shared.constants.LATEST_TRAINING_DATA_FORMAT_VERSION}"
stories:
- story: some-conversation-ID
  steps:
  - intent: greet
    user: |-
      hi
  - action: utter_greet
  - slot_was_set:
    - requested_slot: some value""",
        ),
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

    await tracker_store.save(tracker)

    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )

    url = f"/conversations/{conversation_id}/story?"

    query = {}

    if fetch_all_sessions is not None:
        query["all_sessions"] = fetch_all_sessions

    if until_time is not None:
        query["until"] = until_time

    _, response = await rasa_app.get(url + urllib.parse.urlencode(query))

    assert response.status == HTTPStatus.OK
    assert response.content.decode().strip() == expected


async def test_get_story_without_conversation_id(
    rasa_app: SanicASGITestClient, monkeypatch: MonkeyPatch
):
    conversation_id = "some-conversation-ID"
    url = f"/conversations/{conversation_id}/story"

    _, response = await rasa_app.get(url)

    assert response.status == HTTPStatus.NOT_FOUND
    assert response.json["message"] == "Conversation ID not found."


async def test_get_story_does_not_update_conversation_session(
    rasa_app: SanicASGITestClient, monkeypatch: MonkeyPatch
):
    conversation_id = "some-conversation-ID"

    # domain with short session expiration time of one second
    domain = Domain.empty()
    domain.session_config = SessionConfig(
        session_expiration_time=1 / 60, carry_over_slots=True
    )

    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent.processor, "domain", domain)

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
    assert rasa_app.sanic_app.ctx.agent.processor._has_session_expired(tracker)

    tracker_store = InMemoryTrackerStore(domain)

    await tracker_store.save(tracker)

    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )

    _, response = await rasa_app.get(f"/conversations/{conversation_id}/story")

    assert response.status == HTTPStatus.OK

    # expected story is returned
    assert (
        response.content.decode().strip()
        == f"""version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
    agent = rasa_app.sanic_app.ctx.agent
    tracker_store = agent.tracker_store
    domain = agent.domain
    model_id = agent.model_id
    assistant_id = agent.processor.model_metadata.assistant_id

    if initial_tracker_events:
        tracker = await agent.processor.get_tracker(conversation_id)
        tracker.update_with_events(initial_tracker_events, domain)
        await tracker_store.save(tracker)

    fetched_tracker = await rasa.server.update_conversation_with_events(
        conversation_id, agent.processor, domain, events_to_append
    )
    assert list(fetched_tracker.events) == with_assistant_ids(
        with_model_ids(expected_events, model_id), assistant_id
    )


async def test_append_events_does_not_repeat_session_start(
    rasa_app: SanicASGITestClient,
):
    session_start_events = [
        {
            "event": "action",
            "timestamp": 1644577572.9639301,
            "metadata": {
                "assistant_id": "unique_stack_assistant_test_name",
                "model_id": "f90a69066e4a438aa6edfbed5b529919",
            },
            "name": "action_session_start",
            "policy": None,
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "session_started",
            "timestamp": 1644577572.963996,
            "metadata": {
                "assistant_id": "unique_stack_assistant_test_name",
                "model_id": "f90a69066e4a438aa6edfbed5b529919",
            },
        },
        {
            "event": "action",
            "timestamp": 1644577572.964009,
            "metadata": {
                "assistant_id": "unique_stack_assistant_test_name",
                "model_id": "f90a69066e4a438aa6edfbed5b529919",
            },
            "name": "action_listen",
            "policy": None,
            "confidence": None,
            "action_text": None,
            "hide_rule_turn": False,
        },
    ]
    _, response = await rasa_app.post(
        "/conversations/testid/tracker/events", json=session_start_events
    )

    assert response.json["events"] == session_start_events


async def _create_tracker_for_query_params(
    rasa_app: SanicASGITestClient,
    model_id: Text,
) -> Tuple[Text, List[Event]]:
    sender_id = uuid.uuid4().hex

    events_to_store: List[Event] = (
        session_start_sequence
        + [
            UserUttered(
                "hi",
                parse_data={
                    "intent": {"name": "greet"},
                    "metadata": {"model_id": model_id},
                },
            ),
            ActionExecuted("utter_greet"),
            BotUttered("hey there"),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                "/restart",
                parse_data={
                    "intent": {"name": "restart"},
                    "metadata": {"model_id": model_id},
                },
            ),
            ActionExecuted(ACTION_RESTART_NAME),
            Restarted(),
        ]
        + session_start_sequence
        + [
            UserUttered(
                "hi again",
                parse_data={
                    "intent": {"name": "greet"},
                    "metadata": {"model_id": model_id},
                },
            ),
        ]
    )
    serialized_events_to_store = [event.as_dict() for event in events_to_store]

    _, response = await rasa_app.post(
        f"/conversations/{sender_id}/tracker/events", json=serialized_events_to_store
    )
    assert response.status == 200

    return sender_id, events_to_store


async def test_get_tracker_with_query_param_include_events_all(
    rasa_app: SanicASGITestClient,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    assistant_id = rasa_app.sanic_app.ctx.agent.processor.model_metadata.assistant_id
    sender_id, events_to_store = await _create_tracker_for_query_params(
        rasa_app, model_id
    )

    _, response = await rasa_app.get(
        f"/conversations/{sender_id}/tracker?include_events=ALL"
    )
    assert response.status == 200

    tracker = response.json
    assert tracker["sender_id"] == sender_id

    serialized_actual_events = tracker["events"]

    expected_events = with_assistant_ids(
        with_model_ids(events_to_store, model_id), assistant_id
    )
    serialized_expected_events = [event.as_dict() for event in expected_events]

    assert serialized_actual_events == serialized_expected_events


async def test_get_tracker_with_query_param_include_events_after_restart(
    rasa_app: SanicASGITestClient,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    assistant_id = rasa_app.sanic_app.ctx.agent.processor.model_metadata.assistant_id
    sender_id, events_to_store = await _create_tracker_for_query_params(
        rasa_app, model_id
    )

    _, response = await rasa_app.get(
        f"/conversations/{sender_id}/tracker?include_events=AFTER_RESTART"
    )
    assert response.status == 200

    tracker = response.json
    assert tracker["sender_id"] == sender_id

    serialized_actual_events = tracker["events"]

    restarted_event = [
        event for event in events_to_store if isinstance(event, Restarted)
    ][0]
    truncated_events = events_to_store[events_to_store.index(restarted_event) + 1 :]
    expected_events = with_assistant_ids(
        with_model_ids(truncated_events, model_id), assistant_id
    )
    serialized_expected_events = [e.as_dict() for e in expected_events]

    assert serialized_actual_events == serialized_expected_events


async def test_get_tracker_with_query_param_include_events_applied(
    rasa_app: SanicASGITestClient,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    assistant_id = rasa_app.sanic_app.ctx.agent.processor.model_metadata.assistant_id
    sender_id, events_to_store = await _create_tracker_for_query_params(
        rasa_app, model_id
    )

    _, response = await rasa_app.get(
        f"/conversations/{sender_id}/tracker?include_events=APPLIED"
    )
    assert response.status == 200

    tracker = response.json
    assert tracker["sender_id"] == sender_id

    serialized_actual_events = tracker["events"]

    restarted_event = [
        event for event in events_to_store if isinstance(event, Restarted)
    ][0]
    truncated_events = events_to_store[events_to_store.index(restarted_event) + 1 :]
    session_started = [
        event for event in truncated_events if isinstance(event, SessionStarted)
    ][0]
    truncated_events = truncated_events[truncated_events.index(session_started) + 1 :]

    expected_events = with_assistant_ids(
        with_model_ids(truncated_events, model_id), assistant_id
    )
    serialized_expected_events = [e.as_dict() for e in expected_events]

    assert serialized_actual_events == serialized_expected_events


async def test_get_tracker_with_query_param_include_events_none(
    rasa_app: SanicASGITestClient,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    sender_id, events_to_store = await _create_tracker_for_query_params(
        rasa_app, model_id
    )

    _, response = await rasa_app.get(
        f"/conversations/{sender_id}/tracker?include_events=NONE"
    )
    assert response.status == 200

    tracker = response.json
    assert tracker["sender_id"] == sender_id

    serialized_actual_events = tracker["events"]
    assert serialized_actual_events is None


async def test_retrieve_story_with_query_param_all_sessions_true(
    rasa_app: SanicASGITestClient,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    sender_id, events_to_store = await _create_tracker_for_query_params(
        rasa_app, model_id
    )

    _, response = await rasa_app.get(
        f"/conversations/{sender_id}/story?all_sessions=true"
    )
    assert response.status == 200

    story_content = response.body.decode("utf-8")

    expected_first_story = textwrap.dedent(
        f"""
    - story: {sender_id}, story 1
      steps:
      - intent: greet
        user: |-
          hi
      - action: utter_greet
      - intent: restart
        user: |-
          /restart
      - action: action_restart"""
    )
    assert expected_first_story in story_content

    expected_second_story = textwrap.dedent(
        f"""
    - story: {sender_id}, story 2
      steps:
      - intent: greet
        user: |-
          hi again"""
    )
    assert expected_second_story in story_content


async def test_retrieve_story_with_query_param_all_sessions_false(
    rasa_app: SanicASGITestClient,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    sender_id, events_to_store = await _create_tracker_for_query_params(
        rasa_app, model_id
    )

    _, response = await rasa_app.get(
        f"/conversations/{sender_id}/story?all_sessions=false"
    )
    assert response.status == 200

    story_content = response.body.decode("utf-8")
    expected_story = textwrap.dedent(
        f"""
    - story: {sender_id}
      steps:
      - intent: greet
        user: |-
          hi again"""
    )
    assert expected_story in story_content


async def test_retrieve_tracker_with_customized_action_session_start(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    sender_id = str(uuid.uuid1())

    async def mock_run_action_session_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[Event]:
        _events: List[Event] = [
            SessionStarted(),
            BotUttered("Hey there!", {"name": "utter_greet"}),
            ActionExecuted(ACTION_LISTEN_NAME),
        ]

        return _events

    monkeypatch.setattr(
        "rasa.core.actions.action.ActionSessionStart.run", mock_run_action_session_start
    )

    _, response = await rasa_app.get(
        f"/conversations/{sender_id}/tracker",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == 200

    tracker = response.json
    assert tracker is not None

    tracker_events = tracker.get("events")
    assert len(tracker_events) == 4

    assert tracker_events[0].get("event") == "action"
    assert tracker_events[0].get("name") == "action_session_start"

    assert tracker_events[1].get("event") == "session_started"

    assert tracker_events[2].get("event") == "bot"
    assert tracker_events[2].get("text") == "Hey there!"

    assert tracker_events[3].get("event") == "action"
    assert tracker_events[3].get("name") == "action_listen"
