import asyncio
from http import HTTPStatus
import json
from pathlib import Path
from typing import Any, Dict, Text, Callable, Optional
from unittest.mock import patch
import uuid

from aioresponses import aioresponses
import pytest
from _pytest.monkeypatch import MonkeyPatch
from pytest_sanic.utils import TestClient
from sanic import Sanic, response
from sanic.request import Request
from sanic.response import ResponseStream

import rasa.core
from rasa.core.exceptions import AgentNotReady
from rasa.core.utils import AvailableEndpoints
from rasa.exceptions import ModelNotFound
from rasa.nlu.persistor import Persistor
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    DefinePrevUserUtteredFeaturization,
    SessionStarted,
    UserUttered,
)
from rasa.shared.nlu.constants import INTENT_NAME_KEY
import rasa.shared.utils.common
import rasa.utils.io
from rasa.core import jobs
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.channel import UserMessage
from rasa.shared.core.domain import Domain
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import with_assistant_ids, with_model_ids


def model_server_app(model_path: Text, model_hash: Text = "somehash") -> Sanic:
    app = Sanic("test_agent")
    app.ctx.number_of_model_requests = 0

    @app.route("/model", methods=["GET"])
    async def model(request: Request) -> ResponseStream:
        """Simple HTTP model server responding with a trained model."""

        if model_hash == request.headers.get("If-None-Match"):
            return response.text("", 204)

        app.ctx.number_of_model_requests += 1

        return await response.file_stream(
            location=model_path,
            headers={"ETag": model_hash, "filename": Path(model_path).name},
            mime_type="application/gzip",
        )

    return app


@pytest.fixture()
def model_server(
    loop: asyncio.AbstractEventLoop, sanic_client: Callable, trained_rasa_model: Text
) -> TestClient:
    app = model_server_app(trained_rasa_model, model_hash="somehash")
    return loop.run_until_complete(sanic_client(app))


async def test_agent_train(default_agent: Agent):
    domain = Domain.load("data/test_domains/default_with_slots.yml")

    assert default_agent.domain.action_names_or_texts == domain.action_names_or_texts
    assert default_agent.domain.intents == domain.intents
    assert default_agent.domain.entities == domain.entities
    assert default_agent.domain.responses == domain.responses
    assert [s.name for s in default_agent.domain.slots] == [
        s.name for s in domain.slots
    ]

    assert default_agent.processor
    assert default_agent.processor.graph_runner


@pytest.mark.parametrize(
    "text_message_data, expected",
    [
        (
            '/greet{"name":"Rasa"}',
            {
                "text": '/greet{"name":"Rasa"}',
                "intent": {"name": "greet", "confidence": 1.0},
                "intent_ranking": [{"name": "greet", "confidence": 1.0}],
                "entities": [
                    {
                        "entity": "name",
                        "start": 6,
                        "end": 21,
                        "value": "Rasa",
                        "extractor": "RegexMessageHandler",
                    }
                ],
            },
        )
    ],
)
async def test_agent_parse_message(
    default_agent: Agent, text_message_data: Text, expected: Dict[Text, Any]
):
    result = await default_agent.parse_message(text_message_data)
    assert result == expected


async def test_agent_handle_text(default_agent: Agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    result = await default_agent.handle_text(text, sender_id="test_agent_handle_text")
    assert result == [
        {"recipient_id": "test_agent_handle_text", "text": "hey there Rasa!"}
    ]


async def test_default_agent_handle_message(default_agent: Agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    message = UserMessage(text, sender_id="test_agent_handle_message")
    result = await default_agent.handle_message(message)
    assert result == [
        {"recipient_id": "test_agent_handle_message", "text": "hey there Rasa!"}
    ]


async def test_agent_wrong_use_of_load():
    training_data_file = "data/test_moodbot/data/stories.yml"

    with pytest.raises(ModelNotFound):
        # try to load a model file from a data path, which is nonsense and
        # should fail properly
        Agent.load(training_data_file)


async def test_agent_with_model_server_in_thread(
    model_server: TestClient, domain: Domain
):
    model_endpoint_config = EndpointConfig.from_dict(
        {"url": model_server.make_url("/model"), "wait_time_between_pulls": 2}
    )

    agent = Agent()
    agent = await rasa.core.agent.load_from_server(
        agent, model_server=model_endpoint_config
    )

    await asyncio.sleep(5)

    assert agent.fingerprint == "somehash"
    assert agent.domain.as_dict() == domain.as_dict()
    assert agent.processor.graph_runner

    assert model_server.app.ctx.number_of_model_requests == 1
    jobs.kill_scheduler()


async def test_wait_time_between_pulls_without_interval(
    model_server: TestClient, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(
        "rasa.core.agent._schedule_model_pulling", lambda *args: 1 / 0
    )  # will raise an exception

    model_endpoint_config = EndpointConfig.from_dict(
        {"url": model_server.make_url("/model"), "wait_time_between_pulls": None}
    )

    agent = Agent()
    # should not call _schedule_model_pulling, if it does, this will raise
    await rasa.core.agent.load_from_server(agent, model_server=model_endpoint_config)


async def test_load_agent(trained_rasa_model: Text):
    agent = await load_agent(model_path=trained_rasa_model)

    assert agent.tracker_store is not None
    assert agent.lock_store is not None
    assert agent.processor is not None
    assert agent.processor.graph_runner is not None


async def test_load_agent_on_not_existing_path():
    agent = await load_agent(model_path="some-random-path")

    assert agent
    assert agent.processor is None


async def test_load_from_remote_storage(trained_nlu_model: Text):
    class FakePersistor(Persistor):
        def _persist_tar(self, filekey: Text, tarname: Text) -> None:
            pass

        def _retrieve_tar(self, filename: Text) -> Text:
            pass

        def retrieve(self, model_name: Text, target_path: Text) -> None:
            self._copy(model_name, target_path)

    with patch("rasa.nlu.persistor.get_persistor", new=lambda _: FakePersistor()):
        agent = await load_agent(
            remote_storage="some-random-remote", model_path=trained_nlu_model
        )

    assert agent is not None
    assert agent.is_ready()


@pytest.mark.parametrize(
    "model_path",
    [
        "non-existing-path",
        "data/test_domains/default_with_slots.yml",
        "not-existing-model.tar.gz",
        None,
    ],
)
async def test_agent_load_on_invalid_model_path(model_path: Optional[Text]):
    with pytest.raises(ModelNotFound):
        Agent.load(model_path)


async def test_agent_handle_message_full_model(default_agent: Agent):
    model_id = default_agent.model_id
    assistant_id = default_agent.processor.model_metadata.assistant_id
    sender_id = uuid.uuid4().hex
    message = UserMessage("hello", sender_id=sender_id)
    await default_agent.handle_message(message)
    tracker = await default_agent.tracker_store.get_or_create_tracker(sender_id)
    events = with_model_ids(
        [
            ActionExecuted(action_name="action_session_start"),
            SessionStarted(),
            ActionExecuted(action_name="action_listen"),
            UserUttered(text="hello", intent={"name": "greet"}),
            DefinePrevUserUtteredFeaturization(False),
            ActionExecuted(action_name="utter_greet"),
            BotUttered(
                "hey there None!",
                {
                    "elements": None,
                    "quick_replies": None,
                    "buttons": None,
                    "attachment": None,
                    "image": None,
                    "custom": None,
                },
                {"utter_action": "utter_greet"},
            ),
            ActionExecuted(action_name="action_listen"),
        ],
        model_id,
    )
    expected_events = with_assistant_ids(events, assistant_id)
    assert len(tracker.events) == len(expected_events)
    for e1, e2 in zip(tracker.events, expected_events):
        assert e1 == e2


async def test_agent_handle_message_only_nlu(trained_nlu_model: Text):
    agent = await load_agent(model_path=trained_nlu_model)
    model_id = agent.model_id
    assistant_id = agent.processor.model_metadata.assistant_id
    sender_id = uuid.uuid4().hex
    message = UserMessage("hello", sender_id=sender_id)
    await agent.handle_message(message)
    tracker = await agent.tracker_store.get_or_create_tracker(sender_id)
    events = with_model_ids(
        [
            ActionExecuted(action_name="action_session_start"),
            SessionStarted(),
            ActionExecuted(action_name="action_listen"),
            UserUttered(text="hello", intent={"name": "greet"}),
        ],
        model_id,
    )
    expected_events = with_assistant_ids(events, assistant_id)
    assert len(tracker.events) == len(expected_events)
    for e1, e2 in zip(tracker.events, expected_events):
        assert e1 == e2


async def test_agent_handle_message_only_core(trained_core_model: Text):
    agent = await load_agent(model_path=trained_core_model)
    model_id = agent.model_id
    assistant_id = agent.processor.model_metadata.assistant_id
    sender_id = uuid.uuid4().hex
    message = UserMessage("/greet", sender_id=sender_id)
    await agent.handle_message(message)
    tracker = await agent.tracker_store.get_or_create_tracker(sender_id)
    events = with_model_ids(
        [
            ActionExecuted(action_name="action_session_start"),
            SessionStarted(),
            ActionExecuted(action_name="action_listen"),
            UserUttered(text="/greet", intent={"name": "greet"}),
            DefinePrevUserUtteredFeaturization(False),
            ActionExecuted(action_name="utter_greet"),
            BotUttered(
                "hey there None!",
                {
                    "elements": None,
                    "quick_replies": None,
                    "buttons": None,
                    "attachment": None,
                    "image": None,
                    "custom": None,
                },
                {"utter_action": "utter_greet"},
            ),
            ActionExecuted(action_name="action_listen"),
        ],
        model_id,
    )
    expected_events = with_assistant_ids(events, assistant_id)
    assert len(tracker.events) == len(expected_events)
    for e1, e2 in zip(tracker.events, expected_events):
        assert e1 == e2


async def test_agent_update_model(trained_core_model: Text, trained_nlu_model: Text):
    agent1 = await load_agent(model_path=trained_core_model)
    agent2 = await load_agent(model_path=trained_core_model)

    assert (
        agent1.processor.model_metadata.predict_schema
        == agent2.processor.model_metadata.predict_schema
    )

    agent2.load_model(trained_nlu_model)
    assert not (
        agent1.processor.model_metadata.predict_schema
        == agent2.processor.model_metadata.predict_schema
    )


async def test_parse_with_http_interpreter(trained_default_agent_model: Text):
    endpoints = AvailableEndpoints(nlu=EndpointConfig("https://interpreter.com"))
    agent = await load_agent(
        model_path=trained_default_agent_model, endpoints=endpoints
    )
    response_body = {
        "intent": {INTENT_NAME_KEY: "some_intent", "confidence": 1.0},
        "entities": [],
        "text": "lunch?",
    }

    with aioresponses() as mocked:
        mocked.post(
            "https://interpreter.com/model/parse",
            repeat=True,
            status=HTTPStatus.OK,
            body=json.dumps(response_body),
        )

        # mock the parse function with the one defined for this test
        result = await agent.parse_message("lunch?")
        assert result == response_body


@pytest.mark.parametrize(
    "method_name",
    [
        "parse_message",
        "predict_next_for_sender_id",
        "predict_next_with_tracker",
        "log_message",
        "execute_action",
        "trigger_intent",
        "handle_text",
    ],
)
def test_agent_checks_if_ready(method_name):
    not_ready_agent = Agent()
    with pytest.raises(AgentNotReady):
        getattr(not_ready_agent, method_name)()
