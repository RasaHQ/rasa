import logging
import os
from typing import Text

import matplotlib
import pytest

import rasa.utils.io
from rasa.core import train
from rasa.core.agent import Agent
from rasa.core.channels import CollectingOutputChannel, RestInput, channel
from rasa.core.domain import Domain
from rasa.core.interpreter import RegexInterpreter
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.policies.ensemble import PolicyEnsemble, SimplePolicyEnsemble
from rasa.core.policies.memoization import (
    AugmentedMemoizationPolicy,
    MemoizationPolicy,
    Policy,
)
from rasa.core.processor import MessageProcessor
from rasa.core.slots import Slot
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.core.trackers import DialogueStateTracker
from rasa.core.utils import zip_folder
from rasa.train import train_async

matplotlib.use("Agg")

logging.basicConfig(level="DEBUG")

DEFAULT_DOMAIN_PATH = "data/test_domains/default_with_slots.yml"

DEFAULT_STORIES_FILE = "data/test_stories/stories_defaultdomain.md"

DEFAULT_STACK_CONFIG = "data/test_config/stack_config.yml"

DEFAULT_NLU_DATA = "examples/moodbot/data/nlu.md"

END_TO_END_STORY_FILE = "data/test_evaluations/end_to_end_story.md"

E2E_STORY_FILE_UNKNOWN_ENTITY = "data/test_evaluations/story_unknown_entity.md"

MOODBOT_MODEL_PATH = "examples/moodbot/models/"

DEFAULT_ENDPOINTS_FILE = "data/test_endpoints/example_endpoints.yml"

TEST_DIALOGUES = [
    "data/test_dialogues/default.json",
    "data/test_dialogues/formbot.json",
    "data/test_dialogues/moodbot.json",
    "data/test_dialogues/restaurantbot.json",
]

EXAMPLE_DOMAINS = [
    DEFAULT_DOMAIN_PATH,
    "examples/formbot/domain.yml",
    "examples/moodbot/domain.yml",
    "examples/restaurantbot/domain.yml",
]


class CustomSlot(Slot):
    def as_feature(self):
        return [0.5]


# noinspection PyAbstractClass,PyUnusedLocal,PyMissingConstructor
class ExamplePolicy(Policy):
    def __init__(self, example_arg):
        pass


@pytest.fixture
def loop():
    from pytest_sanic.plugin import loop as sanic_loop

    return rasa.utils.io.enable_async_loop_debugging(next(sanic_loop()))


@pytest.fixture(scope="session")
def default_domain_path():
    return DEFAULT_DOMAIN_PATH


@pytest.fixture(scope="session")
def default_stories_file():
    return DEFAULT_STORIES_FILE


@pytest.fixture(scope="session")
def default_stack_config():
    return DEFAULT_STACK_CONFIG


@pytest.fixture(scope="session")
def default_nlu_data():
    return DEFAULT_NLU_DATA


@pytest.fixture(scope="session")
def default_domain():
    return Domain.load(DEFAULT_DOMAIN_PATH)


@pytest.fixture(scope="session")
async def default_agent(default_domain):
    agent = Agent(
        default_domain,
        policies=[MemoizationPolicy()],
        interpreter=RegexInterpreter(),
        tracker_store=InMemoryTrackerStore(default_domain),
    )
    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    return agent


@pytest.fixture(scope="session")
def default_agent_path(default_agent, tmpdir_factory):
    path = tmpdir_factory.mktemp("agent").strpath
    default_agent.persist(path)
    return path


@pytest.fixture
def default_channel():
    return CollectingOutputChannel()


@pytest.fixture
async def default_processor(default_domain, default_nlg):
    agent = Agent(
        default_domain,
        SimplePolicyEnsemble([AugmentedMemoizationPolicy()]),
        interpreter=RegexInterpreter(),
    )

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    tracker_store = InMemoryTrackerStore(default_domain)
    return MessageProcessor(
        agent.interpreter,
        agent.policy_ensemble,
        default_domain,
        tracker_store,
        default_nlg,
    )


@pytest.fixture(scope="session")
async def trained_moodbot_path():
    await train(
        domain_file="examples/moodbot/domain.yml",
        stories_file="examples/moodbot/data/stories.md",
        output_path=MOODBOT_MODEL_PATH,
        interpreter=RegexInterpreter(),
        policy_config="rasa/cli/default_config.yml",
        kwargs=None,
    )

    return MOODBOT_MODEL_PATH


@pytest.fixture(scope="session")
async def zipped_moodbot_model():
    # train moodbot if necessary
    policy_file = os.path.join(MOODBOT_MODEL_PATH, "core", "metadata.json")
    if not os.path.isfile(policy_file):
        await trained_moodbot_path()

    zip_path = zip_folder(os.path.dirname(MOODBOT_MODEL_PATH))

    return zip_path


@pytest.fixture(scope="session")
def moodbot_domain():
    domain_path = os.path.join(MOODBOT_MODEL_PATH, "core", "domain.yml")
    return Domain.load(domain_path)


@pytest.fixture(scope="session")
def moodbot_metadata():
    return PolicyEnsemble.load_metadata(os.path.join(MOODBOT_MODEL_PATH, "core"))


@pytest.fixture()
async def trained_stack_model(
    default_domain_path, default_stack_config, default_nlu_data, default_stories_file
):

    trained_stack_model_path = await train_async(
        domain=default_domain_path,
        config=default_stack_config,
        training_files=[default_nlu_data, default_stories_file],
    )

    return trained_stack_model_path


@pytest.fixture
async def prepared_agent(tmpdir_factory) -> Agent:
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent(
        "data/test_domains/default.yml",
        policies=[AugmentedMemoizationPolicy(max_history=3)],
    )

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    agent.persist(model_path)
    return agent


@pytest.fixture
def default_nlg(default_domain):
    return TemplatedNaturalLanguageGenerator(default_domain.templates)


@pytest.fixture
def default_tracker(default_domain):
    return DialogueStateTracker("my-sender", default_domain.slots)


@pytest.fixture(scope="session")
def project() -> Text:
    import tempfile
    from rasa.cli.scaffold import _create_initial_project

    directory = tempfile.mkdtemp()
    _create_initial_project(directory)

    return directory


def train_model(project: Text, filename: Text = "test.tar.gz"):
    from rasa.constants import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_DATA_PATH,
        DEFAULT_DOMAIN_PATH,
        DEFAULT_MODELS_PATH,
    )
    import rasa.train

    output = os.path.join(project, DEFAULT_MODELS_PATH, filename)
    domain = os.path.join(project, DEFAULT_DOMAIN_PATH)
    config = os.path.join(project, DEFAULT_CONFIG_PATH)
    training_files = os.path.join(project, DEFAULT_DATA_PATH)

    rasa.train(domain, config, training_files, output)

    return output


@pytest.fixture(scope="session")
def trained_model(project) -> Text:
    return train_model(project)
