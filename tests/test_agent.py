import asyncio
import io

import pytest
from aioresponses import aioresponses
from sanic import Sanic, response
from typing import Text

import rasa_core
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.interpreter import INTENT_MESSAGE_PREFIX
from rasa_core.policies.memoization import AugmentedMemoizationPolicy
from rasa_core.utils import EndpointConfig


@pytest.fixture(scope="session")
def loop():
    from pytest_sanic.plugin import loop as sanic_loop
    return next(sanic_loop())


async def test_agent_train(tmpdir, default_domain):
    training_data_file = 'examples/moodbot/data/stories.md'
    agent = Agent("examples/moodbot/domain.yml",
                  policies=[AugmentedMemoizationPolicy()])

    training_data = await agent.load_data(training_data_file)
    agent.train(training_data)
    agent.persist(tmpdir.strpath)

    loaded = Agent.load(tmpdir.strpath)

    # test domain
    assert loaded.domain.action_names == agent.domain.action_names
    assert loaded.domain.intents == agent.domain.intents
    assert loaded.domain.entities == agent.domain.entities
    assert loaded.domain.templates == agent.domain.templates
    assert [s.name for s in loaded.domain.slots] == \
           [s.name for s in agent.domain.slots]

    # test policies
    assert type(loaded.policy_ensemble) is type(
        agent.policy_ensemble)  # nopep8
    assert [type(p) for p in loaded.policy_ensemble.policies] == \
           [type(p) for p in agent.policy_ensemble.policies]


async def test_agent_handle_message(default_agent):
    message = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    result = await default_agent.handle_message(
        message,
        sender_id="test_agent_handle_message")
    assert result == [{'recipient_id': 'test_agent_handle_message',
                       'text': 'hey there Rasa!'}]


def test_agent_wrong_use_of_load(tmpdir, default_domain):
    training_data_file = 'examples/moodbot/data/stories.md'
    agent = Agent("examples/moodbot/domain.yml",
                  policies=[AugmentedMemoizationPolicy()])

    with pytest.raises(ValueError):
        # try to load a model file from a data path, which is nonsense and
        # should fail properly
        agent.load(training_data_file)


async def test_agent_with_model_server(tmpdir, zipped_moodbot_model,
                                       moodbot_domain, moodbot_metadata):
    fingerprint = 'somehash'
    model_endpoint_config = EndpointConfig.from_dict(
        {"url": 'http://server.com/model/default_core@latest',
         "wait_time_between_pulls": None}
    )

    # mock a response that returns a zipped model
    with io.open(zipped_moodbot_model, 'rb') as f:
        body = f.read()
    with aioresponses() as mocked:
        mocked.get(model_endpoint_config.url,
                   headers={"ETag": fingerprint,
                            "Content-Type": 'application/zip'},
                   body=body)
        agent = Agent()
        agent = await rasa_core.agent.load_from_server(
            agent, model_server=model_endpoint_config)

    assert agent.fingerprint == fingerprint

    assert agent.domain.as_dict() == moodbot_domain.as_dict()

    agent_policies = {utils.module_path_from_instance(p)
                      for p in agent.policy_ensemble.policies}
    moodbot_policies = set(moodbot_metadata["policy_names"])
    assert agent_policies == moodbot_policies


def model_server_app(model_path: Text, model_hash: Text = "somehash"):
    app = Sanic(__name__)

    @app.route("/model", methods=['GET'])
    async def model(request):
        """Simple HTTP NLG generator, checks that the incoming request
        is format according to the spec."""

        if model_hash == request.headers.get("If-None-Match"):
            return response.text("", 204)

        return await response.file_stream(
            location=model_path,
            headers={'ETag': model_hash,
                     'filename': model_path},
            mime_type='application/zip')

    return app


async def test_agent_with_model_server_in_thread(test_server, tmpdir,
                                                 zipped_moodbot_model,
                                                 moodbot_domain,
                                                 moodbot_metadata):
    fingerprint = 'somehash'
    server = await test_server(model_server_app(zipped_moodbot_model,
                                                model_hash=fingerprint))

    model_endpoint_config = EndpointConfig.from_dict({
        "url": server.make_url('/model'),
        "wait_time_between_pulls": 1
    })

    agent = Agent()
    agent = await rasa_core.agent.load_from_server(
        agent, model_server=model_endpoint_config)

    await asyncio.sleep(10)

    assert agent.fingerprint == fingerprint

    assert agent.domain.as_dict() == moodbot_domain.as_dict()

    agent_policies = {utils.module_path_from_instance(p)
                      for p in agent.policy_ensemble.policies}
    moodbot_policies = set(moodbot_metadata["policy_names"])
    assert agent_policies == moodbot_policies
    await server.close()


def test_wait_time_between_pulls_from_file(monkeypatch):
    monkeypatch.setattr("rasa_core.agent.schedule_model_pulling",
                        lambda *args: True)
    monkeypatch.setattr("rasa_core.agent._update_model_from_server",
                        lambda *args: 1 / 0)  # raises an exception

    model_endpoint_config = utils. \
        read_endpoint_config("data/test_endpoints/model_endpoint.yml", "model")

    agent = Agent()
    rasa_core.agent.load_from_server(agent, model_server=model_endpoint_config)


def test_wait_time_between_pulls_str(monkeypatch):
    monkeypatch.setattr("rasa_core.agent.schedule_model_pulling",
                        lambda *args: True)
    monkeypatch.setattr("rasa_core.agent._update_model_from_server",
                        lambda *args: 1 / 0)  # raises an exception

    model_endpoint_config = EndpointConfig.from_dict(
        {"url": 'http://server.com/model/default_core@latest',
         "wait_time_between_pulls": "10"}
    )

    agent = Agent()
    rasa_core.agent.load_from_server(agent, model_server=model_endpoint_config)


def test_wait_time_between_pulls_with_not_number(monkeypatch):
    monkeypatch.setattr("rasa_core.agent.schedule_model_pulling",
                        lambda *args: 1 / 0)
    monkeypatch.setattr("rasa_core.agent._update_model_from_server",
                        lambda *args: True)

    model_endpoint_config = EndpointConfig.from_dict(
        {"url": 'http://server.com/model/default_core@latest',
         "wait_time_between_pulls": "None"}
    )

    agent = Agent()
    rasa_core.agent.load_from_server(agent, model_server=model_endpoint_config)
