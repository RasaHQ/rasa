import asyncio
from typing import Text

import pytest
from sanic import Sanic, response

import rasa.core
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.mapping_policy import MappingPolicy
import rasa.utils.io
from rasa.core import jobs, utils
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.channel import UserMessage
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.interpreter import INTENT_MESSAGE_PREFIX
from rasa.core.policies.ensemble import PolicyEnsemble, SimplePolicyEnsemble
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from rasa.utils.endpoints import EndpointConfig
from tests.core.conftest import DEFAULT_DOMAIN_PATH_WITH_SLOTS


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


@pytest.fixture()
def model_server(loop, sanic_client, trained_moodbot_path):
    app = model_server_app(trained_moodbot_path, model_hash="somehash")
    return loop.run_until_complete(sanic_client(app))


async def test_training_data_is_reproducible(tmpdir, default_domain):
    training_data_file = "examples/moodbot/data/stories.md"
    agent = Agent(
        "examples/moodbot/domain.yml", policies=[AugmentedMemoizationPolicy()]
    )

    training_data = await agent.load_data(training_data_file)
    # make another copy of training data
    same_training_data = await agent.load_data(training_data_file)

    # test if both datasets are identical (including in the same order)
    for i, x in enumerate(training_data):
        assert str(x.as_dialogue()) == str(same_training_data[i].as_dialogue())


async def test_agent_train(trained_moodbot_path: Text):
    moodbot_domain = Domain.load("examples/moodbot/domain.yml")
    loaded = Agent.load(trained_moodbot_path)

    # test domain
    assert loaded.domain.action_names == moodbot_domain.action_names
    assert loaded.domain.intents == moodbot_domain.intents
    assert loaded.domain.entities == moodbot_domain.entities
    assert loaded.domain.templates == moodbot_domain.templates
    assert [s.name for s in loaded.domain.slots] == [
        s.name for s in moodbot_domain.slots
    ]

    # test policies
    assert isinstance(loaded.policy_ensemble, SimplePolicyEnsemble)
    assert [type(p) for p in loaded.policy_ensemble.policies] == [
        TEDPolicy,
        MemoizationPolicy,
        MappingPolicy,
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


async def test_agent_handle_text(default_agent: Agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    result = await default_agent.handle_text(text, sender_id="test_agent_handle_text")
    assert result == [
        {"recipient_id": "test_agent_handle_text", "text": "hey there Rasa!"}
    ]


async def test_agent_handle_message(default_agent: Agent):
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

    await asyncio.sleep(5)

    assert agent.fingerprint == "somehash"
    assert hash(agent.domain) == hash(moodbot_domain)

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


async def test_load_agent(trained_rasa_model: str):
    agent = await load_agent(model_path=trained_rasa_model)

    assert agent.tracker_store is not None
    assert agent.interpreter is not None
    assert agent.model_directory is not None


@pytest.mark.parametrize(
    "domain, policy_config",
    [({"forms": ["restaurant_form"]}, {"policies": [{"name": "MemoizationPolicy"}]})],
)
def test_form_without_form_policy(domain, policy_config):
    with pytest.raises(InvalidDomain) as execinfo:
        Agent(
            domain=Domain.from_dict(domain),
            policies=PolicyEnsemble.from_dict(policy_config),
        )
    assert "haven't added the FormPolicy" in str(execinfo.value)


@pytest.mark.parametrize(
    "domain, policy_config",
    [
        (
            {
                "intents": [{"affirm": {"triggers": "utter_ask_num_people"}}],
                "actions": ["utter_ask_num_people"],
            },
            {"policies": [{"name": "MemoizationPolicy"}]},
        )
    ],
)
def test_trigger_without_mapping_policy(domain, policy_config):
    with pytest.raises(InvalidDomain) as execinfo:
        Agent(
            domain=Domain.from_dict(domain),
            policies=PolicyEnsemble.from_dict(policy_config),
        )
    assert "haven't added the MappingPolicy" in str(execinfo.value)


@pytest.mark.parametrize(
    "domain, policy_config",
    [({"intents": ["affirm"]}, {"policies": [{"name": "TwoStageFallbackPolicy"}]})],
)
def test_two_stage_fallback_without_deny_suggestion(domain, policy_config):
    with pytest.raises(InvalidDomain) as execinfo:
        Agent(
            domain=Domain.from_dict(domain),
            policies=PolicyEnsemble.from_dict(policy_config),
        )
    assert "The intent 'out_of_scope' must be present" in str(execinfo.value)


async def test_agent_update_model_none_domain(trained_rasa_model: Text):
    agent = await load_agent(model_path=trained_rasa_model)
    agent.update_model(
        None, None, agent.fingerprint, agent.interpreter, agent.model_directory
    )

    assert agent.domain is not None
    sender_id = "test_sender_id"
    message = UserMessage("hello", sender_id=sender_id)
    await agent.handle_message(message)
    tracker = agent.tracker_store.get_or_create_tracker(sender_id)

    # UserUttered event was added to tracker, with correct intent data
    assert tracker.events[3].intent["name"] == "greet"


async def test_load_agent_on_not_existing_path():
    agent = await load_agent(model_path="some-random-path")

    assert agent is None


@pytest.mark.parametrize(
    "model_path",
    [
        "non-existing-path",
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        "not-existing-model.tar.gz",
        None,
    ],
)
async def test_agent_load_on_invalid_model_path(model_path):
    with pytest.raises(ValueError):
        Agent.load(model_path)
