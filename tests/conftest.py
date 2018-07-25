from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import matplotlib
import pytest

from rasa_core import train
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleOutputChannel
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Dispatcher
from rasa_core.domain import TemplateDomain
from rasa_core.interpreter import RegexInterpreter
from rasa_core.nlg import TemplatedNaturalLanguageGenerator
from rasa_core.policies.ensemble import SimplePolicyEnsemble
from rasa_core.policies.memoization import \
    MemoizationPolicy, AugmentedMemoizationPolicy
from rasa_core.processor import MessageProcessor
from rasa_core.slots import Slot
from rasa_core.tracker_store import InMemoryTrackerStore
from rasa_core.trackers import DialogueStateTracker

matplotlib.use('Agg')

logging.basicConfig(level="DEBUG")

DEFAULT_DOMAIN_PATH = "data/test_domains/default_with_slots.yml"

DEFAULT_STORIES_FILE = "data/test_stories/stories_defaultdomain.md"

DEFAULT_ENDPOINTS_FILE = "data/example_endpoints.yml"


class CustomSlot(Slot):
    def as_feature(self):
        return [0.5]


@pytest.fixture(scope="session")
def default_domain():
    return TemplateDomain.load(DEFAULT_DOMAIN_PATH)


@pytest.fixture(scope="session")
def default_agent(default_domain):
    agent = Agent(default_domain,
                  policies=[MemoizationPolicy()],
                  interpreter=RegexInterpreter(),
                  tracker_store=InMemoryTrackerStore(default_domain))
    training_data = agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    return agent


@pytest.fixture(scope="session")
def default_agent_path(default_agent, tmpdir_factory):
    path = tmpdir_factory.mktemp("agent").strpath
    default_agent.persist(path)
    return path


@pytest.fixture
def default_dispatcher_cmd(default_nlg):
    bot = ConsoleOutputChannel()
    return Dispatcher("my-sender", bot, default_nlg)


@pytest.fixture
def default_dispatcher_collecting(default_nlg):
    bot = CollectingOutputChannel()
    return Dispatcher("my-sender", bot, default_nlg)


@pytest.fixture
def default_processor(default_domain, default_nlg):
    agent = Agent(default_domain,
                  SimplePolicyEnsemble([AugmentedMemoizationPolicy()]),
                  interpreter=RegexInterpreter())

    training_data = agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    tracker_store = InMemoryTrackerStore(default_domain)
    return MessageProcessor(agent.interpreter,
                            agent.policy_ensemble,
                            default_domain,
                            tracker_store,
                            default_nlg)


@pytest.fixture(scope="session")
def trained_moodbot_path():
    model_path = "examples/moodbot/models/dialogue"
    train.train_dialogue_model(
            domain_file="examples/moodbot/domain.yml",
            stories_file="examples/moodbot/data/stories.md",
            output_path=model_path,
            use_online_learning=False,
            nlu_model_path=None,
            max_history=None,
            kwargs=None
    )
    return model_path


@pytest.fixture
def default_nlg(default_domain):
    return TemplatedNaturalLanguageGenerator(default_domain.templates)


@pytest.fixture
def default_tracker(default_domain):
    import uuid
    uid = str(uuid.uuid1())
    return DialogueStateTracker(uid, default_domain.slots)
