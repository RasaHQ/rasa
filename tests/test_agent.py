import io

import pytest
import responses

import rasa_core
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.interpreter import INTENT_MESSAGE_PREFIX
from rasa_core.policies.memoization import AugmentedMemoizationPolicy
from rasa_core.utils import EndpointConfig


def test_agent_train(tmpdir, default_domain):
    training_data_file = 'examples/moodbot/data/stories.md'
    agent = Agent("examples/moodbot/domain.yml",
                  policies=[AugmentedMemoizationPolicy()])

    training_data = agent.load_data(training_data_file)
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


def test_agent_handle_message(default_agent):
    message = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    result = default_agent.handle_message(message,
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


@responses.activate
def test_agent_with_model_server(tmpdir, zipped_moodbot_model,
                                 moodbot_domain, moodbot_metadata):
    fingerprint = 'somehash'
    model_endpoint_config = EndpointConfig.from_dict(
        {"url": 'http://server.com/model/default_core@latest',
         "wait_time_between_pulls": None}
    )

    # mock a response that returns a zipped model
    with io.open(zipped_moodbot_model, 'rb') as f:
        responses.add(responses.GET,
                      model_endpoint_config.url,
                      headers={"ETag": fingerprint},
                      body=f.read(),
                      content_type='application/zip',
                      stream=True)
    agent = rasa_core.agent.load_from_server(
        model_server=model_endpoint_config)
    assert agent.fingerprint == fingerprint

    assert agent.domain.as_dict() == moodbot_domain.as_dict()

    agent_policies = {utils.module_path_from_instance(p)
                      for p in agent.policy_ensemble.policies}
    moodbot_policies = set(moodbot_metadata["policy_names"])
    assert agent_policies == moodbot_policies


def test_wait_time_between_pulls_from_file(monkeypatch):
    from future.utils import raise_

    monkeypatch.setattr("rasa_core.agent.start_model_pulling_in_worker",
                        lambda *args: True)
    monkeypatch.setattr("rasa_core.agent._update_model_from_server",
                        lambda *args: raise_(Exception()))

    model_endpoint_config = utils. \
        read_endpoint_config("data/test_endpoints/model_endpoint.yml", "model")

    rasa_core.agent. \
        load_from_server(model_server=model_endpoint_config)


def test_wait_time_between_pulls_str(monkeypatch):
    from future.utils import raise_

    monkeypatch.setattr("rasa_core.agent.start_model_pulling_in_worker",
                        lambda *args: True)
    monkeypatch.setattr("rasa_core.agent._update_model_from_server",
                        lambda *args: raise_(Exception()))

    model_endpoint_config = EndpointConfig.from_dict(
        {"url": 'http://server.com/model/default_core@latest',
         "wait_time_between_pulls": "10"}
    )

    rasa_core.agent. \
        load_from_server(model_server=model_endpoint_config)


def test_wait_time_between_pulls_with_not_number(monkeypatch):
    from future.utils import raise_

    monkeypatch.setattr("rasa_core.agent.start_model_pulling_in_worker",
                        lambda *args: raise_(Exception()))
    monkeypatch.setattr("rasa_core.agent._update_model_from_server",
                        lambda *args: True)

    model_endpoint_config = EndpointConfig.from_dict(
        {"url": 'http://server.com/model/default_core@latest',
         "wait_time_between_pulls": "None"}
    )

    rasa_core.agent. \
        load_from_server(model_server=model_endpoint_config)
