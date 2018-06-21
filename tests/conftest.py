from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import matplotlib
import os
import pytest

from rasa_core import train
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleOutputChannel
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Dispatcher
from rasa_core.domain import TemplateDomain
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.ensemble import SimplePolicyEnsemble
from rasa_core.policies.memoization import \
    MemoizationPolicy, AugmentedMemoizationPolicy
from rasa_core.processor import MessageProcessor
from rasa_core.slots import Slot
from rasa_core.tracker_store import InMemoryTrackerStore
from rasa_core.utils import zip_folder

matplotlib.use('Agg')

logging.basicConfig(level="DEBUG")

DEFAULT_DOMAIN_PATH = "data/test_domains/default_with_slots.yml"

DEFAULT_STORIES_FILE = "data/test_stories/stories_defaultdomain.md"

MOODBOT_MODEL_PATH = "examples/moodbot/models/dialogue"


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


@pytest.fixture
def default_dispatcher_cmd(default_domain):
    bot = ConsoleOutputChannel()
    return Dispatcher("my-sender", bot, default_domain)


@pytest.fixture
def default_dispatcher_collecting(default_domain):
    bot = CollectingOutputChannel()
    return Dispatcher("my-sender", bot, default_domain)


@pytest.fixture
def default_processor(default_domain):
    agent = Agent(default_domain,
                  SimplePolicyEnsemble([AugmentedMemoizationPolicy()]),
                  interpreter=RegexInterpreter())

    training_data = agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    tracker_store = InMemoryTrackerStore(default_domain)
    return MessageProcessor(agent.interpreter,
                            agent.policy_ensemble,
                            default_domain,
                            tracker_store)


@pytest.fixture(scope="session")
def trained_moodbot_path():
    train.train_dialogue_model(
        domain_file="examples/moodbot/domain.yml",
        stories_file="examples/moodbot/data/stories.md",
        output_path=MOODBOT_MODEL_PATH,
        use_online_learning=False,
        nlu_model_path=None,
        max_history=None,
        kwargs=None
    )

    return MOODBOT_MODEL_PATH


@pytest.fixture(scope="session")
def zipped_moodbot_model():
    # train moodbot if necessary
    policy_file = os.path.join(MOODBOT_MODEL_PATH, 'policy_metadata.json')
    if not os.path.isfile(policy_file):
        trained_moodbot_path()

    zip_path = zip_folder(MOODBOT_MODEL_PATH)

    return zip_path
