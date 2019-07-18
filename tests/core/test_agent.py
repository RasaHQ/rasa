import asyncio
from typing import Text

import pytest
from async_generator import async_generator, yield_
from sanic import Sanic, response

import rasa.utils.io
import rasa.core
from rasa.core import jobs, utils
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.channel import UserMessage
from rasa.core.interpreter import INTENT_MESSAGE_PREFIX
from rasa.core.policies.memoization import AugmentedMemoizationPolicy
from rasa.utils.endpoints import EndpointConfig
from tests.core.conftest import DEFAULT_DOMAIN_PATH


@pytest.fixture(scope="session")
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)
    yield loop
    loop.close()


def model_server_app(model_path: Text, model_hash: Text = "somehash"):
    app = Sanic(__name__)
    app.number_of_model_requests = 0

    @app.route("/model", methods=["GET"])
    async def model(request):
        """Simple HTTP model server responding with a trained model."""

        if model_hash == request.headers.get("If-None-Match"):
            return response.text("", 204)

        app.number_of_model_requests += 1

        return await response.file_stream(
            location=model_path,
            headers={"ETag": model_hash, "filename": model_path},
            mime_type="application/gzip",
        )

    return app


@pytest.fixture
@async_generator
async def model_server(test_server, trained_moodbot_path):
    server = await test_server(
        model_server_app(trained_moodbot_path, model_hash="somehash")
    )
    await yield_(server)  # python 3.5 compatibility
    await server.close()


async def test_agent_train(tmpdir, default_domain):
    training_data_file = "examples/moodbot/data/stories.md"
    agent = Agent(
        "examples/moodbot/domain.yml", policies=[AugmentedMemoizationPolicy()]
    )

    training_data = await agent.load_data(training_data_file)
    agent.train(training_data)
    agent.persist(tmpdir.strpath)

    loaded = Agent.load(tmpdir.strpath)

    # test domain
    assert loaded.domain.action_names == agent.domain.action_names
    assert loaded.domain.intents == agent.domain.intents
    assert loaded.domain.entities == agent.domain.entities
    assert loaded.domain.templates == agent.domain.templates
    assert [s.name for s in loaded.domain.slots] == [s.name for s in agent.domain.slots]

    # test policies
    assert isinstance(loaded.policy_ensemble, type(agent.policy_ensemble))
    assert [type(p) for p in loaded.policy_ensemble.policies] == [
        type(p) for p in agent.policy_ensemble.policies
    ]


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
                    {"entity": "name", "start": 6, "end": 21, "value": "Rasa"}
                ],
            },
        ),
        (
            "text",
            {
                "text": "/text",
                "intent": {"name": "text", "confidence": 1.0},
                "intent_ranking": [{"name": "text", "confidence": 1.0}],
                "entities": [],
            },
        ),
    ],
)
async def test_agent_parse_message_using_nlu_interpreter(
    default_agent, text_message_data, expected
):
    result = await default_agent.parse_message_using_nlu_interpreter(text_message_data)
    assert result == expected


async def test_agent_handle_text(default_agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    result = await default_agent.handle_text(text, sender_id="test_agent_handle_text")
    assert result == [
        {"recipient_id": "test_agent_handle_text", "text": "hey there Rasa!"}
    ]


async def test_agent_handle_message(default_agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    message = UserMessage(text, sender_id="test_agent_handle_message")
    result = await default_agent.handle_message(message)
    assert result == [
        {"recipient_id": "test_agent_handle_message", "text": "hey there Rasa!"}
    ]


def test_agent_wrong_use_of_load(tmpdir, default_domain):
    training_data_file = "examples/moodbot/data/stories.md"
    agent = Agent(
        "examples/moodbot/domain.yml", policies=[AugmentedMemoizationPolicy()]
    )

    with pytest.raises(ValueError):
        # try to load a model file from a data path, which is nonsense and
        # should fail properly
        agent.load(training_data_file)


async def test_agent_with_model_server_in_thread(
    model_server, moodbot_domain, moodbot_metadata
):
    model_endpoint_config = EndpointConfig.from_dict(
        {"url": model_server.make_url("/model"), "wait_time_between_pulls": 2}
    )

    agent = Agent()
    agent = await rasa.core.agent.load_from_server(
        agent, model_server=model_endpoint_config
    )

    await asyncio.sleep(3)

    assert agent.fingerprint == "somehash"

    assert agent.domain.as_dict() == moodbot_domain.as_dict()

    agent_policies = {
        utils.module_path_from_instance(p) for p in agent.policy_ensemble.policies
    }
    moodbot_policies = set(moodbot_metadata["policy_names"])
    assert agent_policies == moodbot_policies
    assert model_server.app.number_of_model_requests == 1
    jobs.kill_scheduler()


async def test_wait_time_between_pulls_without_interval(model_server, monkeypatch):

    monkeypatch.setattr(
        "rasa.core.agent.schedule_model_pulling", lambda *args: 1 / 0
    )  # will raise an exception

    model_endpoint_config = EndpointConfig.from_dict(
        {"url": model_server.make_url("/model"), "wait_time_between_pulls": None}
    )

    agent = Agent()
    # schould not call schedule_model_pulling, if it does, this will raise
    await rasa.core.agent.load_from_server(agent, model_server=model_endpoint_config)
    jobs.kill_scheduler()


async def test_load_agent(trained_model):
    agent = await load_agent(model_path=trained_model)

    assert agent.tracker_store is not None
    assert agent.interpreter is not None
    assert agent.model_directory is not None


async def test_load_agent_on_not_existing_path():
    agent = await load_agent(model_path="some-random-path")

    assert agent is None


@pytest.mark.parametrize(
    "model_path",
    ["non-existing-path", DEFAULT_DOMAIN_PATH, "not-existing-model.tar.gz", None],
)
async def test_agent_load_on_invalid_model_path(model_path):
    with pytest.raises(ValueError):
        Agent.load(model_path)
