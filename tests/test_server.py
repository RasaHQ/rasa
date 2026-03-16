# file deepcode ignore HardcodedNonCryptoSecret/test: Secrets are all just examples for tests. # noqa: E501

import asyncio
import json
import logging
import os
import socket
import sys
import textwrap
import threading
import time
import urllib.parse
import uuid
from argparse import Namespace
from http import HTTPStatus
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Dict, Generator, List, NoReturn, Optional, Text, Tuple, Type
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
import requests
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
from a2a.types import Task, TaskState, TaskStatus
from aioresponses import aioresponses
from pytest import LogCaptureFixture
from ruamel.yaml import StringIO
from sanic import Sanic
from sanic_testing.testing import SanicASGITestClient

import rasa
import rasa.agents.protocol.a2a.a2a_agent as a2a_mod
import rasa.constants
import rasa.core.jobs
import rasa.nlu
import rasa.nlu.test
import rasa.server
import rasa.shared.constants
import rasa.shared.utils.io
import rasa.utils.io
from rasa.agents.core.cancellation import CancellationToken
from rasa.agents.core.types import AgentStatus
from rasa.agents.protocol.a2a.a2a_agent import A2AAgent
from rasa.agents.schemas import AgentInput
from rasa.core import utils
from rasa.core.agent import Agent, load_agent
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentInfo,
    ProtocolConfig,
)
from rasa.core.channels import (
    CallbackInput,
    CollectingOutputChannel,
    RestInput,
    SlackInput,
    channel,
)
from rasa.core.channels.slack import SlackBot
from rasa.core.lock_store import InMemoryLockStore
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_stores.sql_tracker_store import SQLTrackerStore
from rasa.core.tracker_stores.tracker_store import InMemoryTrackerStore, TrackerStore
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.model_training import TrainingResult
from rasa.nlu.test import CVEvaluationResult
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core import events
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    DEFAULT_SLOT_NAMES,
    REQUESTED_SLOT,
)
from rasa.shared.core.domain import Domain, SessionConfig
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    Restarted,
    SessionEnded,
    SessionStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
from rasa.shared.utils.yaml import read_yaml_file, write_yaml
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import (
    USERNAME,
    with_assistant_id,
    with_assistant_ids,
    with_model_id,
    with_model_ids,
    with_model_name,
    with_model_names,
    with_session_id,
    with_session_ids,
)
from tests.core.conftest import MockedMongoTrackerStore
from tests.core.tracker_stores.conftest import create_multiple_trackers_with_user_id
from tests.core.tracker_stores.test_redis_tracker_store import MockedRedisTrackerStore
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
def rasa_app_without_api_and_without_inspector(
    rasa_server_without_api_and_without_inspector: Sanic,
) -> SanicASGITestClient:
    return rasa_server_without_api_and_without_inspector.asgi_client


@pytest.fixture
def rasa_app_without_api_and_with_inspector(
    rasa_server_without_api_and_with_inspector: Sanic,
) -> SanicASGITestClient:
    return rasa_server_without_api_and_with_inspector.asgi_client


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
    assert "Hello from Rasa:" in response.text


async def test_root_with_enabled_inspector(
    rasa_app_without_api_and_with_inspector: SanicASGITestClient,
):
    _, response = await rasa_app_without_api_and_with_inspector.get("/")
    assert response.status == HTTPStatus.OK
    assert "Hello from Rasa:" in response.text
    assert (
        '<a href="./webhooks/inspector/inspect.html">Go to the inspector</a>'
        in response.text
    )


async def test_root_without_enabled_inspector(
    rasa_app_without_api_and_without_inspector: SanicASGITestClient,
):
    _, response = await rasa_app_without_api_and_without_inspector.get("/")
    assert response.status == HTTPStatus.OK
    assert "Hello from Rasa:" in response.text
    assert (
        '<a href="./webhooks/inspector/inspect.html">Go to the inspector</a>'
        not in response.text
    )


async def test_root_without_enable_api(rasa_app_without_api: SanicASGITestClient):
    _, response = await rasa_app_without_api.get("/")
    assert response.status == HTTPStatus.OK
    assert "Hello from Rasa:" in response.text


async def test_root_secured(rasa_non_trained_secured_app: SanicASGITestClient):
    _, response = await rasa_non_trained_secured_app.get("/")
    assert response.status == HTTPStatus.OK
    assert "Hello from Rasa:" in response.text


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


async def test_train_status_tracks_active_training_jobs(
    rasa_app: SanicASGITestClient,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    """Test that the status endpoint correctly tracks active training jobs.

    This test verifies that:
    1. num_active_training_jobs is 0 when no training is running
    2. num_active_training_jobs increments when training starts
    3. The status endpoint is accessible during training (not blocked)
    4. num_active_training_jobs decrements when training completes
    """
    fake_model = Path(tmp_path) / "fake_model.tar.gz"
    fake_model.touch()
    fake_model_path = str(fake_model)

    # Use threading.Event since training runs in a thread pool (due to @run_in_thread)
    # The training function runs in a separate thread with its own event loop
    training_complete = threading.Event()

    async def mocked_training_function(*_, **__) -> TrainingResult:
        """Mock training function that simulates a training process."""
        # Wait for the test to check status during training
        # Poll the event with async sleep to avoid blocking
        max_wait = 30.0
        elapsed = 0.0
        while not training_complete.is_set() and elapsed < max_wait:
            await asyncio.sleep(0.1)
            elapsed += 0.1
        return TrainingResult(model=fake_model_path)

    monkeypatch.setattr(rasa.model_training, "train", mocked_training_function)

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

    # Check initial status - should be 0 active training jobs
    _, initial_status = await rasa_app.get("/status")
    assert initial_status.status == HTTPStatus.OK
    assert initial_status.json["num_active_training_jobs"] == 0

    # Start training in the background
    training_task = asyncio.create_task(
        rasa_app.post(
            "/model/train",
            data=training_data,
            headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
        )
    )

    # Poll the status endpoint until training starts (counter > 0)
    # This verifies the status endpoint is not blocked by training
    max_attempts = 50
    for attempt in range(max_attempts):
        await asyncio.sleep(0.1)  # Small delay between checks
        _, status_check = await rasa_app.get("/status")
        assert status_check.status == HTTPStatus.OK
        num_jobs = status_check.json["num_active_training_jobs"]
        if num_jobs == 1:
            # Training has started, verify status is accessible
            assert num_jobs == 1
            break
    else:
        pytest.fail(
            f"Training did not start within {max_attempts * 0.1}s. "
            f"Final num_active_training_jobs: "
            f"{status_check.json['num_active_training_jobs']}"
        )

    # Allow training to complete
    training_complete.set()

    # Wait for training to finish
    _, training_response = await training_task
    assert training_response.status == HTTPStatus.OK

    # Check final status - should be back to 0 active training jobs
    _, final_status = await rasa_app.get("/status")
    assert final_status.status == HTTPStatus.OK
    assert final_status.json["num_active_training_jobs"] == 0


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
    domain_data = read_yaml_file(domain_path)
    config_data = read_yaml_file(stack_config_path)
    nlu_data = read_yaml_file(nlu_data_path)

    # combine all data into our payload
    payload = {
        key: val for d in [domain_data, config_data, nlu_data] for key, val in d.items()
    }

    data = StringIO()
    write_yaml(payload, data)

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
        content = read_yaml_file(file)
        payload.update(content)

        concatenated_payload_file = tmp_path / "concatenated.yml"
        write_yaml(payload, concatenated_payload_file)

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


@patch("langchain_community.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_llm_client")
@patch("rasa.shared.utils.health_check.health_check.try_instantiate_embedder")
async def test_train_CALM_bot_with_yaml_success(
    mock_try_instantiate_llm_client: Mock,
    mock_try_instantiate_embedder: Mock,
    mock_flow_search_create_embedder: Mock,
    mock_from_documents: Mock,
    rasa_app: SanicASGITestClient,
    tmp_path_factory: TempPathFactory,
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    mock_try_instantiate_llm_client.return_value = Mock()
    mock_flow_search_create_embedder.return_value = Mock()
    mock_from_documents.return_value = Mock()
    mock_try_instantiate_embedder.return_value = Mock()

    training_data = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

pipeline:
  - name: SingleStepLLMCommandGenerator
    llm:
      model: "gpt-4"
      provider: "openai"

policies:
    - name: "FlowPolicy"

nlu: []

intents: []

entities: []

rules: []

stories: []

responses:
  utter_no_contacts:
   - text: "You have no contacts in your list."

slots:
    contacts_list:
      type: text
      mappings:
       - type: custom
         action: list_contacts

actions:
 - list_contacts

flows:
  list_contacts:
    name: "list your contacts"
    description: "show your contact list"
    steps:
     - action: list_contacts
     - action: utter_no_contacts
"""
    _, response = await rasa_app.post(
        "/model/train",
        data=training_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    assert_trained_model(response.body, tmp_path_factory)


async def test_train_CALM_bot_with_yaml_bad_flows_request(
    rasa_app: SanicASGITestClient, tmp_path_factory: TempPathFactory
):
    training_data = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

pipeline:
  - name: SingleStepLLMCommandGenerator
    llm:
        "model": "gpt-4"
        "provider": "openai"

policies:
    - "name": "FlowPolicy"

nlu: []

intents: []

entities: []

rules: []

stories: []

responses:
  utter_no_contacts:
   - text: "You have no contacts in your list."

slots:
    contacts_list:
      type: text
      mappings:
       - type: custom
         action: list_contacts

actions:
 - list_contacts

flows:
  - list_contacts:
    name: "list your contacts"
    description: "show your contact list"
    steps:
     - action: list_contacts
     - action: utter_no_contacts
"""
    _, response = await rasa_app.post(
        "/model/train",
        data=training_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.BAD_REQUEST
    assert "Found a list but expected a dictionary of flows." in response.body.decode()


async def test_train_CALM_bot_with_yaml_bad_request_rasa_exception(
    rasa_app: SanicASGITestClient,
    tmp_path_factory: TempPathFactory,
    monkeypatch: MonkeyPatch,
):
    training_data = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

pipeline:
  - name: SingleStepLLMCommandGenerator
    llm:
        "model": "gpt-4"
        "provider": "openai"

policies:
    - "name": "FlowPolicy"

nlu: []

intents: []

entities: []

rules: []

stories: []

responses:
  utter_no_contacts:
   - text: "You have no contacts in your list."

slots:
    contacts_list:
      type: text
      mappings:
       - type: custom
         action: list_contacts

actions:
 - list_contacts

flows:
  list_contacts:
    name: "list your contacts"
    description: "show your contact list"
    steps:
     - action: list_contacts
     - action: utter_no_contacts
"""
    monkeypatch.setattr(
        rasa.shared.core.flows.yaml_flows_io.YAMLFlowsReader,
        "read_from_string",
        Mock(side_effect=RasaException("Failed to read YAML.")),
    )
    _, response = await rasa_app.post(
        "/model/train",
        data=training_data,
        headers={"Content-type": rasa.server.YAML_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.BAD_REQUEST
    assert (
        "The request body does not contain valid YAML. Error: Failed to read YAML."
        in response.body.decode()
    )


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
    mock_train = AsyncMock(return_value=TrainingResult(model=fake_model_path))
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

        await asyncio.sleep(3)

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


async def test_requesting_non_existent_tracker(rasa_app: SanicASGITestClient):
    _, response = await rasa_app.get("/conversations/madeupid/tracker")
    content = response.json
    assert response.status == HTTPStatus.OK
    assert content["paused"] is False
    assert content["slots"] == {
        "language": "en",
        "name": None,
        **{slot: None for slot in DEFAULT_SLOT_NAMES},
    }
    assert content["sender_id"] == "madeupid"
    assert len(content["events"]) == 3
    assert content["events"][0]["event"] == "action"
    assert content["events"][0]["name"] == "action_session_start"

    assert content["events"][1]["event"] == "session_started"
    assert content["events"][2]["event"] == "action"
    assert content["events"][2]["name"] == "action_listen"

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
    model_name = rasa_app.sanic_app.ctx.agent.processor.model_filename
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
    session_id = tracker.get("current_session_id")

    # there is an initial session start sequence at the beginning of the tracker

    assert deserialized_events[:3] == with_session_ids(
        with_assistant_ids(
            with_model_names(
                with_model_ids(session_start_sequence, model_id), model_name
            ),
            assistant_id,
        ),
        session_id,
    )

    assert deserialized_events[3] == with_session_id(
        with_assistant_id(
            with_model_name(with_model_id(event, model_id), model_name),
            assistant_id,
        ),
        session_id,
    )
    assert deserialized_events[3].timestamp > time_before_adding_events


async def test_pushing_event_with_existing_model_id(rasa_app: SanicASGITestClient):
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    model_name = rasa_app.sanic_app.ctx.agent.processor.model_filename
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
    session_id = tracker["current_session_id"]

    deserialized_events = [Event.from_parameters(event) for event in tracker["events"]]

    # there is an initial session start sequence at the beginning of the tracker
    received_event = deserialized_events[3]
    assert received_event == with_session_id(
        with_model_name(
            with_assistant_id(with_model_id(event, existing_model_id), assistant_id),
            model_name,
        ),
        session_id,
    )


async def test_push_multiple_events(rasa_app: SanicASGITestClient):
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    model_name = rasa_app.sanic_app.ctx.agent.processor.model_filename
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
    session_id = tracker.get("current_session_id")

    # there is an initial session start sequence at the beginning
    assert [
        Event.from_parameters(event) for event in tracker.get("events")
    ] == with_session_ids(
        with_assistant_ids(
            with_model_names(
                with_model_ids(session_start_sequence + test_events, model_id),
                model_name,
            ),
            assistant_id,
        ),
        session_id,
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
    model_name = rasa_app.sanic_app.ctx.agent.processor.model_filename
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
    session_id = tracker.get("current_session_id")
    assert [
        Event.from_parameters(event) for event in tracker.get("events")
    ] == with_session_ids(
        with_assistant_ids(
            with_model_names(
                with_model_ids(session_start_sequence + test_events, model_id),
                model_name,
            ),
            assistant_id,
        ),
        session_id,
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
        "delete_tracker",
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
        "license",
        "load_model",
        "unload_model",
        "get_domain",
        "get_flows",
        "get_data",
        "get_sub_agents",
        "get_trackers_by_user_id",
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
        (
            [RestInput(), SlackInput("test", slack_signing_secret="foobar")],
            SlackBot,
        ),
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


async def test_get_story_with_new_conversation_id(
    rasa_app: SanicASGITestClient, monkeypatch: MonkeyPatch
):
    conversation_id = "some-conversation-ID-42"
    url = f"/conversations/{conversation_id}/story"

    _, response = await rasa_app.get(url)

    expected = """version: "3.1"
stories:
- story: some-conversation-ID-42
  steps: []"""

    assert response.status == HTTPStatus.OK
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
    model_name = agent.processor.model_filename
    assistant_id = agent.processor.model_metadata.assistant_id

    if initial_tracker_events:
        tracker = await agent.processor.get_tracker(conversation_id)
        tracker.update_with_events(initial_tracker_events)
        await tracker_store.save(tracker)

    fetched_tracker = await rasa.server.update_conversation_with_events(
        conversation_id, agent.processor, domain, events_to_append
    )
    session_id = fetched_tracker.current_session_id
    assert list(fetched_tracker.events) == with_session_ids(
        with_assistant_ids(
            with_model_names(with_model_ids(expected_events, model_id), model_name),
            assistant_id,
        ),
        session_id,
    )


async def test_update_conversation_with_events_ignores_terminated_tracker(
    rasa_app: SanicASGITestClient,
    caplog: LogCaptureFixture,
):
    """Test that events are ignored when trying to update a terminated conversation."""
    conversation_id = uuid.uuid4().hex
    agent = rasa_app.sanic_app.ctx.agent
    tracker_store = agent.tracker_store
    domain = agent.domain
    model_id = agent.model_id
    model_name = agent.processor.model_filename
    assistant_id = agent.processor.model_metadata.assistant_id

    # Create a terminated tracker
    initial_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("/greet", {"name": "greet", "confidence": 1.0}),
        SessionEnded(),
    ]
    tracker = await agent.processor.get_tracker(conversation_id)
    tracker.update_with_events(initial_events)
    await tracker_store.save(tracker)

    events_to_append = [
        ActionExecuted("utter_greet"),
        UserUttered("/goodbye", {"name": "goodbye", "confidence": 1.0}),
    ]

    with caplog.at_level(logging.WARNING):
        fetched_tracker = await rasa.server.update_conversation_with_events(
            conversation_id, agent.processor, domain, events_to_append
        )

    message = (
        f"Attempting to add {len(events_to_append)} event(s) to terminated "
        f"conversation '{conversation_id}'. Events will be ignored."
    )
    assert message in caplog.text

    # Events should NOT be added after SessionEnded
    assert list(fetched_tracker.events) == with_assistant_ids(
        with_model_names(with_model_ids(initial_events, model_id), model_name),
        assistant_id,
    )


@pytest.mark.parametrize(
    "query_string",
    ["", "?execute_side_effects=true"],
)
async def test_append_session_ended_cancels_timer(
    rasa_app: SanicASGITestClient,
    query_string: str,
):
    """Appending SessionEnded via the API cancels any active session timer.

    The timer must be cancelled unconditionally regardless of whether
    execute_side_effects is passed, so that timer store resources are freed
    immediately upon session termination.
    """
    conversation_id = uuid.uuid4().hex
    agent = rasa_app.sanic_app.ctx.agent
    timer_manager = agent.processor.timer_manager

    # Schedule a timer to simulate an active session
    await timer_manager.schedule_timer(
        sender_id=conversation_id,
        session_id="test-session-id",
        timeout_seconds=300.0,
        callback=AsyncMock(),
    )
    assert await timer_manager.get_timer(conversation_id) is not None

    _, response = await rasa_app.post(
        f"/conversations/{conversation_id}/tracker/events{query_string}",
        json=[{"event": "session_ended"}],
    )

    assert response.status == HTTPStatus.OK
    assert await timer_manager.get_timer(conversation_id) is None


async def test_append_events_does_not_repeat_session_start(
    rasa_app: SanicASGITestClient,
    mock_session_id: str,
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

    resp_events = response.json["events"]
    model_name = rasa_app.sanic_app.ctx.agent.processor.model_filename
    for e in session_start_events:
        e.setdefault("metadata", {})["session_id"] = mock_session_id
        e["metadata"]["model_name"] = model_name

    assert resp_events == session_start_events


async def test_append_session_ended_cancels_background_tasks(
    rasa_app: SanicASGITestClient,
):
    """Appending SessionEnded via API cancels active background tasks."""
    agent = rasa_app.sanic_app.ctx.agent
    with patch.object(
        agent, "cancel_background_tasks", return_value=True
    ) as mock_cancel:
        _, response = await rasa_app.post(
            "/conversations/cancel-test/tracker/events",
            json=[{"event": "session_ended"}],
        )
        assert response.status == HTTPStatus.OK
        mock_cancel.assert_called_once_with("cancel-test")


async def test_append_conversation_inactive_cancels_background_tasks(
    rasa_app: SanicASGITestClient,
):
    """Appending ConversationInactive via API cancels active background tasks."""
    agent = rasa_app.sanic_app.ctx.agent
    with patch.object(
        agent, "cancel_background_tasks", return_value=True
    ) as mock_cancel:
        _, response = await rasa_app.post(
            "/conversations/cancel-test-2/tracker/events",
            json=[{"event": "inactive"}],
        )
        assert response.status == HTTPStatus.OK
        mock_cancel.assert_called_once_with("cancel-test-2")


async def test_append_regular_event_does_not_cancel_background_tasks(
    rasa_app: SanicASGITestClient,
):
    """Appending a regular event (e.g. SlotSet) does not cancel background tasks."""
    agent = rasa_app.sanic_app.ctx.agent
    with patch.object(agent, "cancel_background_tasks") as mock_cancel:
        _, response = await rasa_app.post(
            "/conversations/cancel-test-3/tracker/events",
            json=[{"event": "slot", "name": "test_slot", "value": "test_value"}],
        )
        assert response.status == HTTPStatus.OK
        mock_cancel.assert_not_called()


def test_e2e_append_session_ended_stops_a2a_polling():
    """POST /tracker/events with SessionEnded interrupts active A2A polling.

    Wires the full ``append_events`` HTTP endpoint (via ``create_app``)
    through a real ``Agent.cancel_background_tasks`` → real processor token
    registry → real ``CancellationToken`` → real A2A polling loop.

    Verifies:

    1. The HTTP endpoint detects the terminal event and cancels background tasks.
    2. A2A polling (max_wait=120s) exits promptly with ``CANCELLED``.
    3. ``SessionEnded`` is persisted on the tracker.
    """
    # -- Processor with real token registry + tracker methods --
    domain = Domain.empty()
    tracker_store = InMemoryTrackerStore(domain)
    lock_store = InMemoryLockStore()

    processor = MagicMock(spec=MessageProcessor)
    processor._active_cancellation_tokens = {}
    processor.register_cancellation_token = (
        MessageProcessor.register_cancellation_token.__get__(processor)
    )
    processor.cancel_background_tasks = (
        MessageProcessor.cancel_background_tasks.__get__(processor)
    )
    processor.get_tracker = MessageProcessor.get_tracker.__get__(processor)
    processor.fetch_tracker_with_initial_session = (
        MessageProcessor.fetch_tracker_with_initial_session.__get__(processor)
    )
    processor.tracker_store = tracker_store
    processor.lock_store = lock_store
    processor.domain = domain
    processor.model_metadata = MagicMock(model_id="test", assistant_id="test")
    processor.model_filename = "test_model"
    processor.timer_manager = None
    processor._handle_session_timer_events = AsyncMock()

    # -- Agent with real cancel_background_tasks --
    agent = Agent.__new__(Agent)
    agent.processor = processor
    agent.tracker_store = tracker_store
    agent.lock_store = lock_store
    agent.domain = domain

    # -- Create a Sanic app using create_app --
    app = rasa.server.create_app(agent=agent)

    sender_id = "api-session-ended-e2e"

    # Pre-populate tracker with a valid session (synchronous via new loop)
    loop = asyncio.new_event_loop()

    async def _setup_tracker():
        tracker = await tracker_store.get_or_create_tracker(sender_id)
        tracker.update(SessionStarted())
        tracker.update(UserUttered("hello"))
        tracker.model_id = "test"
        tracker.model_name = "test_model"
        tracker.assistant_id = "test"
        await tracker_store.save(tracker)

    loop.run_until_complete(_setup_tracker())

    # -- A2A agent that polls forever --
    non_terminal_task = Task(
        context_id="ctx",
        id="ctx-001",
        status=TaskStatus(state=TaskState.working),
    )

    async def _forever_working_stream():
        yield (non_terminal_task, None)

    mock_client = MagicMock()
    mock_client.send_message.side_effect = lambda *a, **kw: _forever_working_stream()
    mock_client.get_task = AsyncMock(return_value=non_terminal_task)

    with patch("rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client") as mock_init:
        mock_init.return_value = mock_client
        a2a_agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_agent",
                    description="Test",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(agent_card="some/path"),
            )
        )
        with patch(
            "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
        ) as mock_load:
            mock_card = MagicMock()
            mock_card.url = "http://example.com"
            mock_load.return_value = mock_card
            loop.run_until_complete(a2a_agent.connect())

    original_max_wait = getattr(a2a_mod, "A2A_TASK_POLLING_MAX_WAIT", 60)

    token = CancellationToken()
    processor.register_cancellation_token(sender_id, token)

    polling_result: dict = {}

    def _run_polling():
        try:
            a2a_mod.A2A_TASK_POLLING_MAX_WAIT = 120
            start = time.monotonic()
            output = loop.run_until_complete(
                a2a_agent.run(
                    AgentInput(
                        id="ctx",
                        metadata={},
                        user_message="Test",
                        slots=[],
                        conversation_history="",
                        events=[],
                    ),
                    cancellation_token=token,
                )
            )
            polling_result["output"] = output
            polling_result["elapsed"] = time.monotonic() - start
        finally:
            a2a_mod.A2A_TASK_POLLING_MAX_WAIT = original_max_wait

    polling_thread = threading.Thread(target=_run_polling)
    polling_thread.start()

    # Give polling time to start, then POST SessionEnded through the endpoint
    time.sleep(0.3)

    _, res = app.test_client.post(
        f"/conversations/{sender_id}/tracker/events",
        json=[{"event": "session_ended"}],
    )
    assert res.status_code == HTTPStatus.OK

    polling_thread.join(timeout=10)
    assert not polling_thread.is_alive(), "Polling thread should have finished"

    loop.close()

    output = polling_result["output"]
    elapsed = polling_result["elapsed"]

    assert output.status == AgentStatus.CANCELLED
    assert (output.metadata or {}).get("cancellation_reason") == "Polling cancelled"
    assert elapsed < 5.0, f"Polling should have exited promptly, took {elapsed:.2f}s"
    assert token.is_cancelled is True


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
    mock_session_id: str,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    model_name = rasa_app.sanic_app.ctx.agent.processor.model_filename
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

    expected_events = with_session_ids(
        with_assistant_ids(
            with_model_names(with_model_ids(events_to_store, model_id), model_name),
            assistant_id,
        ),
        mock_session_id,
    )

    serialized_expected_events = [event.as_dict() for event in expected_events]

    assert serialized_actual_events == serialized_expected_events


async def test_get_tracker_with_query_param_include_events_after_restart(
    rasa_app: SanicASGITestClient,
    mock_session_id: str,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    model_name = rasa_app.sanic_app.ctx.agent.processor.model_filename
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

    restarted_event = [  # noqa: RUF015
        event for event in events_to_store if isinstance(event, Restarted)
    ][0]
    truncated_events = events_to_store[events_to_store.index(restarted_event) + 1 :]
    expected_events = with_session_ids(
        with_assistant_ids(
            with_model_names(with_model_ids(truncated_events, model_id), model_name),
            assistant_id,
        ),
        mock_session_id,
    )
    serialized_expected_events = [e.as_dict() for e in expected_events]

    assert serialized_actual_events == serialized_expected_events


async def test_get_tracker_with_query_param_include_events_applied(
    rasa_app: SanicASGITestClient,
    mock_session_id: str,
) -> None:
    model_id = rasa_app.sanic_app.ctx.agent.model_id
    model_name = rasa_app.sanic_app.ctx.agent.processor.model_filename
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

    restarted_event = [  # noqa: RUF015
        event for event in events_to_store if isinstance(event, Restarted)
    ][0]
    truncated_events = events_to_store[events_to_store.index(restarted_event) + 1 :]
    session_started = [  # noqa: RUF015
        event for event in truncated_events if isinstance(event, SessionStarted)
    ][0]
    truncated_events = truncated_events[truncated_events.index(session_started) + 1 :]

    expected_events = with_session_ids(
        with_assistant_ids(
            with_model_names(with_model_ids(truncated_events, model_id), model_name),
            assistant_id,
        ),
        mock_session_id,
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


async def test_delete_tracker(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    conversation_id = "test_id"
    tracker_store = InMemoryTrackerStore(Domain.empty())
    tracker = DialogueStateTracker.from_events(conversation_id, [])

    await tracker_store.save(tracker)

    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )

    with caplog.at_level(logging.INFO):
        _, response = await rasa_app.delete(f"/conversations/{conversation_id}/tracker")

        assert response.status == HTTPStatus.NO_CONTENT
        assert f"Tracker for conversation '{conversation_id}' deleted." in caplog.text


async def test_delete_no_tracker(
    rasa_app: SanicASGITestClient,
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        _, response = await rasa_app.delete("/conversations/non_existent_id}/tracker")

        assert response.status == HTTPStatus.NOT_FOUND
        assert "Conversation ID not found." in caplog.text


async def test_delete_tracker_server_error(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    """Test that delete tracker endpoint returns 500 when an unexpected error occurs."""
    # Given
    conversation_id = "test_id"
    tracker_store = InMemoryTrackerStore(Domain.empty())
    tracker = DialogueStateTracker.from_events(conversation_id, [])

    await tracker_store.save(tracker)

    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )

    # Simulate a server error by raising an exception in the delete method
    error = "Database connection failed"
    mock_delete = AsyncMock(side_effect=RuntimeError(error))
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.tracker_store, "delete", mock_delete
    )

    # When
    with caplog.at_level(logging.DEBUG):
        _, response = await rasa_app.delete(f"/conversations/{conversation_id}/tracker")

    # Then
    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json["reason"] == "ConversationError"
    assert "An unexpected error occurred" in response.json["message"]
    assert error in response.json["message"]
    assert error in caplog.text


@pytest.fixture
def server_host() -> str:
    return "localhost"


@pytest.fixture
def server_port() -> int:
    """Get a free port for the test server to avoid conflicts in parallel runs."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture
def start_server(rasa_server_with_flows, server_host, server_port):
    def start_rasa_server():
        rasa_server_with_flows.run(
            host=server_host,
            port=server_port,
            single_process=True,
            debug=True,
            register_sys_signals=False,
        )

    thread = threading.Thread(target=start_rasa_server, daemon=True)
    thread.start()

    # TODO: Remove sleep (currently using because `after_server_start` didn't work)
    time.sleep(0.1)


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on windows")
def test_retrieve_flows(
    start_server,
    server_host,
    server_port,
):
    response = requests.get(
        f"http://{server_host}:{server_port}/flows",
        params={"token": "rasa"},
    )
    assert response.status_code == HTTPStatus.OK
    flows = response.json()
    assert flows
    required_fields = {"id", "description", "steps", "file_path"}
    assert all(required_fields.issubset(flow.keys()) for flow in flows)
    step_types = {"action", "collect", "link", "call", "set_slots", "noop"}
    steps = []
    for flow in flows:
        steps.extend(flow["steps"])
    assert all(any(key in step_types for key in step.keys()) for step in steps)


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on windows")
def test_retrieve_flows_with_invalid_authentication(
    start_server,
    server_host,
    server_port,
):
    response = requests.get(
        f"http://{server_host}:{server_port}/flows",
        params={"token": "invalid"},
    )
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    jsonResponse = response.json()
    assert jsonResponse["version"]
    assert jsonResponse["status"] == "failure"
    assert jsonResponse["reason"] == "NotAuthenticated"
    # Message assertion fails as actual message is just "User is not authenticated."
    # assert jsonResponse["message"] == "User is not authenticated to access resource."
    assert "User is not authenticated. " in jsonResponse["message"]
    assert jsonResponse["code"] == HTTPStatus.UNAUTHORIZED


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on windows")
def test_retrieve_bot_data(start_server, server_host, server_port, domain_path):
    response = requests.get(
        f"http://{server_host}:{server_port}/data",
        params={"token": "rasa"},
    )
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    flows = data.get("flows").values()
    assert flows
    required_fields = {"description", "steps", "file_path"}
    assert all(required_fields.issubset(flow.keys()) for flow in flows)
    step_types = {"action", "collect", "link", "call", "set_slots", "noop"}
    steps = []
    for flow in flows:
        steps.extend(flow["steps"])
    assert all(any(key in step_types for key in step.keys()) for step in steps)

    domain = data.get("domain")
    assert domain
    original_domain_dict = Domain.load(domain_path).as_dict()
    for key in original_domain_dict.keys():
        assert key in domain


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on windows")
def test_retrieve_bot_data_with_invalid_authentication(
    start_server,
    server_host,
    server_port,
):
    response = requests.get(
        f"http://{server_host}:{server_port}/data",
        params={"token": "invalid"},
    )
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    jsonResponse = response.json()
    assert jsonResponse["version"]
    assert jsonResponse["status"] == "failure"
    assert jsonResponse["reason"] == "NotAuthenticated"
    # Message assertion fails as actual message is just "User is not authenticated."
    # assert jsonResponse["message"] == "User is not authenticated to access resource."
    assert "User is not authenticated." in jsonResponse["message"]
    assert jsonResponse["code"] == HTTPStatus.UNAUTHORIZED


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_get_trackers_by_user_id_success(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    tracker_store_type: Type[TrackerStore],
    domain: Domain,
    tmp_path: Path,
) -> None:
    """Test successful retrieval of trackers by user_id."""
    user_id = "test_user_123"
    if tracker_store_type == SQLTrackerStore:
        tracker_store = SQLTrackerStore(domain, db=str(tmp_path / "rasa.db"))
    else:
        tracker_store = tracker_store_type(domain)
    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )
    num_conversations = 2
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, num_conversations
    )

    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    assert "conversations" in response.json
    conversations = response.json["conversations"]
    assert len(conversations) == 2
    conversation1 = conversations[0]
    tracker1 = saved_trackers[0]
    tracker2 = saved_trackers[1]
    assert conversation1["sender_id"] == tracker1.sender_id
    assert conversation1[rasa.constants.USER_ID] == user_id
    assert (
        conversation1["events"] == tracker1.current_state(EventVerbosity.ALL)["events"]
    )
    conversation2 = conversations[1]
    assert (
        conversation2["events"] == tracker2.current_state(EventVerbosity.ALL)["events"]
    )
    assert conversation2["sender_id"] == tracker2.sender_id
    assert conversation2[rasa.constants.USER_ID] == user_id


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_get_trackers_by_user_id_with_pagination(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    tracker_store_type: Type[TrackerStore],
    domain: Domain,
    tmp_path: Path,
) -> None:
    """Test pagination support for trackers by user_id."""
    user_id = "test_user_pagination"
    if tracker_store_type == SQLTrackerStore:
        tracker_store = SQLTrackerStore(domain, db=str(tmp_path / "rasa.db"))
    else:
        tracker_store = tracker_store_type(domain)

    num_conversations = 5
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, num_conversations
    )

    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )

    # Test limit parameter
    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers?limit=2",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    retrieved_conversations = response.json["conversations"]
    assert len(retrieved_conversations) == 2
    assert retrieved_conversations[0]["sender_id"] == saved_trackers[0].sender_id
    assert retrieved_conversations[1]["sender_id"] == saved_trackers[1].sender_id
    assert response.json["limit"] == 2

    # Test offset parameter
    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers?limit=2&offset=2",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    retrieved_conversations2 = response.json["conversations"]
    assert response.status == HTTPStatus.OK
    assert len(retrieved_conversations2) == 2
    assert retrieved_conversations2[0]["sender_id"] == saved_trackers[2].sender_id
    assert retrieved_conversations2[1]["sender_id"] == saved_trackers[3].sender_id
    assert response.json["limit"] == 2
    assert response.json["offset"] == 2


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_get_trackers_by_user_id_empty_result(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    tracker_store_type: Type[TrackerStore],
    domain: Domain,
    tmp_path: Path,
) -> None:
    """Test retrieval when user has no trackers."""
    user_id = "user_with_no_trackers"
    if tracker_store_type == SQLTrackerStore:
        tracker_store = SQLTrackerStore(domain, db=str(tmp_path / "rasa.db"))
    else:
        tracker_store = tracker_store_type(domain)
    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )

    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    assert "conversations" in response.json
    assert len(response.json["conversations"]) == 0


@pytest.mark.parametrize("user_id", ["", " ", "%20", "null"])
async def test_get_trackers_by_user_id_invalid_user_id(
    rasa_app: SanicASGITestClient,
    user_id: str,
) -> None:
    """Test error handling for invalid user_id parameter."""
    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.BAD_REQUEST
    assert response.json["reason"] == "BadRequest"
    assert "user_id cannot be empty" in response.json["message"]


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_get_trackers_by_user_id_invalid_limit(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    tracker_store_type: Type[TrackerStore],
    domain: Domain,
) -> None:
    """Test error handling for invalid limit parameter."""
    user_id = "test_user"
    if tracker_store_type == SQLTrackerStore:
        tracker_store = SQLTrackerStore(domain, host="sqlite:///:memory:")
    else:
        tracker_store = tracker_store_type(domain)
    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)

    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers?limit=-1",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.BAD_REQUEST
    assert response.json["reason"] == "BadRequest"
    assert (
        "Invalid limit parameter. Limit must be positive." == response.json["message"]
    )


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_get_trackers_by_user_id_invalid_offset(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    tracker_store_type: Type[TrackerStore],
    domain: Domain,
) -> None:
    """Test error handling for invalid offset parameter."""
    user_id = "test_user"
    tracker_store = tracker_store_type(domain)
    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)

    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers?offset=-1",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.BAD_REQUEST
    assert response.json["reason"] == "BadRequest"
    assert (
        "Invalid offset parameter. Offset must be positive." in response.json["message"]
    )


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_get_trackers_by_user_id_event_verbosity(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    tracker_store_type: Type[TrackerStore],
    domain: Domain,
    tmp_path: Path,
) -> None:
    """Test that event verbosity parameter works correctly."""
    user_id = "test_user_verbosity"
    if tracker_store_type == SQLTrackerStore:
        tracker_store = SQLTrackerStore(domain, db=str(tmp_path / "rasa.db"))
    else:
        tracker_store = tracker_store_type(domain)
    tracker = DialogueStateTracker.from_events(
        "conversation_1",
        [
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("hello", {"name": "greet"}),
        ],
        user_id=user_id,
    )
    await tracker_store.save(tracker)

    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )

    # Test with NONE verbosity
    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers?include_events=NONE",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    conversations = response.json["conversations"]
    assert len(conversations) == 1
    # With NONE verbosity, events should not be included
    assert (
        conversations[0].get("events") is None
        or len(conversations[0].get("events", [])) == 0
    )


async def test_get_trackers_by_user_id_not_implemented(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    domain: Domain,
) -> None:
    """Test error handling when tracker store doesn't implement method."""
    user_id = "test_user"
    tracker_store = InMemoryTrackerStore(domain)

    # Mock the method to raise NotImplementedError
    async def mock_get_trackers_by_user_id(*args, **kwargs):
        raise NotImplementedError()

    monkeypatch.setattr(
        tracker_store, "get_trackers_by_user_id", mock_get_trackers_by_user_id
    )
    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)

    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.NOT_IMPLEMENTED
    assert response.json["reason"] == "NotImplemented"
    assert "does not support querying by user_id" in response.json["message"]


async def test_get_trackers_by_user_id_server_error(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    """Test error handling when an unexpected error occurs."""
    user_id = "test_user"
    tracker_store = InMemoryTrackerStore(Domain.empty())

    # Mock the method to raise an unexpected error
    error_message = "Database connection failed"

    async def mock_get_trackers_by_user_id(*args, **kwargs):
        raise RuntimeError(error_message)

    monkeypatch.setattr(
        tracker_store, "get_trackers_by_user_id", mock_get_trackers_by_user_id
    )
    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)

    with caplog.at_level(logging.DEBUG):
        _, response = await rasa_app.get(
            f"/users/{user_id}/trackers",
            headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
        )

    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json["reason"] == "TrackerRetrievalError"
    assert "unexpected error occurred" in response.json["message"]
    assert error_message in response.json["message"]


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_get_trackers_by_user_id_with_different_users(
    rasa_app: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    tracker_store_type: Type[TrackerStore],
    domain: Domain,
    tmp_path: Path,
) -> None:
    """Test that endpoint returns only trackers for the specified user."""
    user_id_1 = "user_1"
    user_id_2 = "user_2"
    if tracker_store_type == SQLTrackerStore:
        tracker_store = SQLTrackerStore(domain, db=str(tmp_path / "rasa.db"))
    else:
        tracker_store = tracker_store_type(domain)
    # Create trackers for user_1
    tracker1 = DialogueStateTracker.from_events(
        "conv_1_user_1",
        [SessionStarted(), ActionExecuted(ACTION_LISTEN_NAME)],
        user_id=user_id_1,
    )
    tracker2 = DialogueStateTracker.from_events(
        "conv_2_user_1",
        [SessionStarted(), ActionExecuted(ACTION_LISTEN_NAME)],
        user_id=user_id_1,
    )

    # Create trackers for user_2
    tracker3 = DialogueStateTracker.from_events(
        "conv_1_user_2",
        [SessionStarted(), ActionExecuted(ACTION_LISTEN_NAME)],
        user_id=user_id_2,
    )

    await tracker_store.save(tracker1)
    await tracker_store.save(tracker2)
    await tracker_store.save(tracker3)

    monkeypatch.setattr(rasa_app.sanic_app.ctx.agent, "tracker_store", tracker_store)
    monkeypatch.setattr(
        rasa_app.sanic_app.ctx.agent.processor, "tracker_store", tracker_store
    )

    # Get trackers for user_1
    _, response = await rasa_app.get(
        f"/users/{user_id_1}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    conversations = response.json["conversations"]
    assert len(conversations) == 2
    assert conversations[0]["sender_id"] == tracker1.sender_id
    assert (
        conversations[0]["events"]
        == tracker1.current_state(EventVerbosity.ALL)["events"]
    )
    assert conversations[1]["sender_id"] == tracker2.sender_id
    assert (
        conversations[1]["events"]
        == tracker2.current_state(EventVerbosity.ALL)["events"]
    )

    # Get trackers for user_2
    _, response = await rasa_app.get(
        f"/users/{user_id_2}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.OK
    conversations = response.json["conversations"]
    assert len(conversations) == 1
    assert conversations[0]["sender_id"] == tracker3.sender_id
    assert (
        conversations[0]["events"]
        == tracker3.current_state(EventVerbosity.ALL)["events"]
    )


async def test_get_trackers_by_user_id_authentication_required(
    monkeypatch: MonkeyPatch, empty_agent: Agent
) -> None:
    """Integration test to verify authentication is required."""
    app = rasa.server.create_app(agent=empty_agent, auth_token="rasa")
    rasa_app = SanicASGITestClient(app)

    user_id = f"test_user_{uuid.uuid4().hex}"

    # Test without token
    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.UNAUTHORIZED

    # Test with invalid token
    _, response = await rasa_app.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
        params={"token": "invalid_token"},
    )

    assert response.status == HTTPStatus.UNAUTHORIZED


async def test_get_trackers_by_user_id_jwt_token_required(
    monkeypatch: MonkeyPatch,
    rasa_secured_app_asymmetric: SanicASGITestClient,
) -> None:
    """Integration test to verify authentication is required."""
    user_id = f"test_user_{uuid.uuid4().hex}"

    # Test without token
    _, response = await rasa_secured_app_asymmetric.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
    )

    assert response.status == HTTPStatus.UNAUTHORIZED

    # Test with invalid token
    invalid_jwt = {"Authorization": "Bearer invalid_token"}
    _, response = await rasa_secured_app_asymmetric.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE, **invalid_jwt},
    )

    assert response.status == HTTPStatus.UNAUTHORIZED


async def test_get_trackers_by_user_id_with_valid_jwt(
    rasa_secured_app_asymmetric: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    encoded_jwt_user: str,
) -> None:
    """Test user trackers endpoint succeeds with valid JWT token."""
    tracker_store = InMemoryTrackerStore(Domain.empty())
    monkeypatch.setattr(
        rasa_secured_app_asymmetric.sanic_app.ctx.agent, "tracker_store", tracker_store
    )
    monkeypatch.setattr(
        rasa_secured_app_asymmetric.sanic_app.ctx.agent.processor,
        "tracker_store",
        tracker_store,
    )

    # Save trackers for the user
    tracker1 = DialogueStateTracker.from_events(
        "conversation_1",
        [
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("hello"),
        ],
        user_id=USERNAME,
    )
    tracker2 = DialogueStateTracker.from_events(
        "conversation_2",
        [
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("goodbye"),
        ],
        user_id=USERNAME,
    )
    await tracker_store.save(tracker1)
    await tracker_store.save(tracker2)

    jwt_header = {"Authorization": f"Bearer {encoded_jwt_user}"}
    _, response = await rasa_secured_app_asymmetric.get(
        f"/users/{USERNAME}/trackers", headers=jwt_header
    )

    assert response.status == HTTPStatus.OK
    assert "conversations" in response.json
    conversations = response.json["conversations"]
    assert len(conversations) == 2
    assert conversations[0]["sender_id"] == tracker1.sender_id
    assert conversations[0][rasa.constants.USER_ID] == USERNAME
    assert conversations[1]["sender_id"] == tracker2.sender_id
    assert conversations[1][rasa.constants.USER_ID] == USERNAME


async def test_get_trackers_by_user_id_auth_invalid_jwt_payload(
    rasa_secured_app_asymmetric: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
    encoded_jwt_user: str,
) -> None:
    tracker_store = InMemoryTrackerStore(Domain.empty())
    monkeypatch.setattr(
        rasa_secured_app_asymmetric.sanic_app.ctx.agent, "tracker_store", tracker_store
    )
    monkeypatch.setattr(
        rasa_secured_app_asymmetric.sanic_app.ctx.agent.processor,
        "tracker_store",
        tracker_store,
    )

    # Save trackers for the user
    tracker1 = DialogueStateTracker.from_events(
        "conversation_1",
        [
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("hello"),
        ],
        user_id=USERNAME,
    )
    tracker2 = DialogueStateTracker.from_events(
        "conversation_2",
        [
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("goodbye"),
        ],
        user_id=USERNAME,
    )
    await tracker_store.save(tracker1)
    await tracker_store.save(tracker2)

    jwt_header = {"Authorization": f"Bearer {encoded_jwt_user}"}
    _, response = await rasa_secured_app_asymmetric.get(
        "/users/forbidden_user/trackers", headers=jwt_header
    )

    assert response.status == HTTPStatus.FORBIDDEN
    assert response.json["message"] == "User has insufficient permissions."


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_endpoint_performance_with_large_dataset(
    tracker_store_type: Type[TrackerStore],
    rasa_server_with_flows: Sanic,
    domain: Domain,
    tmp_path: Path,
) -> None:
    """Performance test for /users/{user_id}/trackers endpoint with 1000+ conversations."""  # noqa: E501
    user_id = f"perf_endpoint_test_{uuid.uuid4().hex}"
    if tracker_store_type == SQLTrackerStore:
        tracker_store = SQLTrackerStore(domain, db=str(tmp_path / "rasa.db"))
    else:
        tracker_store = tracker_store_type(domain)

    num_conversations = 1000
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, num_conversations
    )

    rasa_server_with_flows.ctx.agent.tracker_store = tracker_store
    client = SanicASGITestClient(rasa_server_with_flows)

    # Test endpoint performance
    endpoint_start = time.time()
    _, response = await client.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
        params={"token": "rasa"},
    )
    endpoint_time = time.time() - endpoint_start

    assert response.status == HTTPStatus.OK
    conversations = response.json["conversations"]
    assert len(conversations) == num_conversations

    # Performance assertion
    max_time = 15.0
    assert (
        endpoint_time < max_time
    ), f"Endpoint took {endpoint_time:.2f}s, expected < {max_time}s"

    # Test endpoint performance with pagination
    paginated_endpoint_start = time.time()
    _, response = await client.get(
        f"/users/{user_id}/trackers",
        headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
        params={"token": "rasa", "limit": "50"},
    )
    paginated_endpoint_time = time.time() - paginated_endpoint_start

    assert response.status == HTTPStatus.OK
    conversations = response.json["conversations"]
    assert len(conversations) == 50

    # Paginated endpoint should be much faster
    assert (
        paginated_endpoint_time < 5.0
    ), f"Paginated endpoint took {paginated_endpoint_time:.2f}s, expected < 5.0s"


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        InMemoryTrackerStore,
        SQLTrackerStore,
        MockedMongoTrackerStore,
        MockedRedisTrackerStore,
    ],
)
async def test_pagination_offset_performance(
    tracker_store_type: Type[TrackerStore],
    rasa_server_with_flows: Sanic,
    domain: Domain,
    tmp_path: Path,
) -> None:
    """Test that pagination with different offsets performs consistently."""
    if tracker_store_type == SQLTrackerStore:
        tracker_store = SQLTrackerStore(domain, db=str(tmp_path / "rasa.db"))
    else:
        tracker_store = tracker_store_type(domain)
    user_id = f"pagination_perf_test_{uuid.uuid4().hex}"

    num_conversations = 1000
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, num_conversations
    )

    rasa_server_with_flows.ctx.agent.tracker_store = tracker_store
    client = SanicASGITestClient(rasa_server_with_flows)

    # Test pagination at different offsets
    offsets = [10, 250, 500, 750]
    limit = 50

    for offset in offsets:
        start = time.time()
        _, response = await client.get(
            f"/users/{user_id}/trackers",
            headers={"Content-Type": rasa.server.JSON_CONTENT_TYPE},
            params={"token": "rasa", "limit": "50", "offset": str(offset)},
        )
        duration = time.time() - start

        expected_count = min(limit, num_conversations - offset)
        retrieved_conversations = response.json["conversations"]
        assert len(retrieved_conversations) == expected_count

        # Performance should be consistent regardless of offset
        assert (
            duration < 2.0
        ), f"Pagination at offset {offset} too slow: {duration:.2f}s"
