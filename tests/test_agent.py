from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

import rasa_core
from rasa_core.agent import Agent
from rasa_core.interpreter import RegexInterpreter, INTENT_MESSAGE_PREFIX
from rasa_core.policies.memoization import AugmentedMemoizationPolicy
from rasa_core.tracker_store import InMemoryTrackerStore


def test_agent_train(tmpdir, default_domain):
    training_data_file = 'examples/moodbot/data/stories.md'
    agent = Agent("examples/moodbot/domain.yml",
                  policies=[AugmentedMemoizationPolicy()])

    training_data = agent.load_data(training_data_file)
    agent.train(training_data)
    agent.persist(tmpdir.strpath)

    loaded = Agent.load(tmpdir.strpath)

    # test domain
    assert [a.name() for a in loaded.domain.actions] == \
           [a.name() for a in agent.domain.actions]
    assert loaded.domain.intents == agent.domain.intents
    assert loaded.domain.entities == agent.domain.entities
    assert loaded.domain.templates == agent.domain.templates
    assert [s.name for s in loaded.domain.slots] == \
           [s.name for s in agent.domain.slots]

    # test policies
    assert type(loaded.policy_ensemble) is type(agent.policy_ensemble)  # nopep8
    assert [type(p) for p in loaded.policy_ensemble.policies] == \
           [type(p) for p in agent.policy_ensemble.policies]


def test_agent_handle_message(default_agent):
    message = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    result = default_agent.handle_message(message,
                                          sender_id="test_agent_handle_message")
    assert result == [{'recipient_id': 'test_agent_handle_message',
                       'text': 'hey there Rasa!'}]


# TODO: DEPRECATED - remove in version 0.10
def test_deprecated_use_of_train(tmpdir, default_domain):
    training_data_file = 'examples/moodbot/data/stories.md'
    agent = Agent("examples/moodbot/domain.yml",
                  policies=[AugmentedMemoizationPolicy()])

    agent.train(training_data_file)
    agent.persist(tmpdir.strpath)

    loaded = Agent.load(tmpdir.strpath)

    # test domain
    assert [a.name() for a in loaded.domain.actions] == \
           [a.name() for a in agent.domain.actions]
    assert loaded.domain.intents == agent.domain.intents
    assert loaded.domain.entities == agent.domain.entities
    assert loaded.domain.templates == agent.domain.templates
    assert [s.name for s in loaded.domain.slots] == \
           [s.name for s in agent.domain.slots]

    # test policies
    assert type(loaded.policy_ensemble) is type(agent.policy_ensemble)  # nopep8
    assert [type(p) for p in loaded.policy_ensemble.policies] == \
           [type(p) for p in agent.policy_ensemble.policies]


def test_agent_wrong_use_of_load(tmpdir, default_domain):
    training_data_file = 'examples/moodbot/data/stories.md'
    agent = Agent("examples/moodbot/domain.yml",
                  policies=[AugmentedMemoizationPolicy()])

    with pytest.raises(ValueError):
        # try to load a model file from a data path, which is nonsense and
        # should fail properly
        agent.load(training_data_file)
