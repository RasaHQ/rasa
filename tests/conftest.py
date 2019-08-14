import os
import shutil
from typing import Text

import pytest
import logging

from rasa.core.domain import Domain
from rasa.constants import (
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODELS_PATH,
)
from rasa.core.interpreter import RegexInterpreter
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.core.run import _create_app_without_api
from rasa import server
from rasa.core import config
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.channel import RestInput
from rasa.core.channels import channel
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.model import get_model
from rasa.train import train_async


# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.

pytest_plugins = ["pytester"]


DEFAULT_DOMAIN_PATH_WITH_SLOTS = "data/test_domains/default_with_slots.yml"
DEFAULT_DOMAIN_PATH_WITH_MAPPING = "data/test_domains/default_with_mapping.yml"

END_TO_END_STORY_FILE = "data/test_evaluations/end_to_end_story.md"
END_TO_END_STORY_FILE_UNKNOWN_ENTITY = "data/test_evaluations/story_unknown_entity.md"

MOODBOT_MODEL_PATH = "examples/moodbot/models/"


@pytest.fixture(autouse=True)
def set_log_level_debug(caplog):
    # Set the post-test log level to DEBUG for failing tests.  For all tests
    # (failing and successful), the live log level can be additionally set in
    # `setup.cfg`. It should be set to WARNING.
    caplog.set_level(logging.DEBUG)


@pytest.fixture(scope="session")
def project() -> Text:
    import tempfile
    from rasa.cli.scaffold import create_initial_project

    directory = tempfile.mkdtemp()
    create_initial_project(directory)

    return directory


###############
# DEFAULT FILES
###############


@pytest.fixture(scope="session")
def default_domain_path_with_slots():
    return DEFAULT_DOMAIN_PATH_WITH_SLOTS


@pytest.fixture(scope="session")
def default_domain_path_with_mapping():
    return DEFAULT_DOMAIN_PATH_WITH_MAPPING


@pytest.fixture(scope="session")
def default_domain_path(project):
    return os.path.join(project, DEFAULT_DOMAIN_PATH)


@pytest.fixture(scope="session")
def default_stories_file(project):
    return os.path.join(project, DEFAULT_DATA_PATH, "stories.md")


@pytest.fixture(scope="session")
def default_nlu_file(project):
    return os.path.join(project, DEFAULT_DATA_PATH, "nlu.md")


@pytest.fixture(scope="session")
def default_config_path(project):
    return os.path.join(project, DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="session")
def end_to_end_story_file():
    return END_TO_END_STORY_FILE


@pytest.fixture(scope="session")
def end_to_end_story_file_with_unkown_entity():
    return END_TO_END_STORY_FILE_UNKNOWN_ENTITY


@pytest.fixture(scope="session")
def default_config(default_config_path):
    return config.load(default_config_path)


@pytest.fixture(scope="session")
def default_domain(default_domain_path):
    return Domain.load(default_domain_path)


#######
# AGENT
#######


@pytest.fixture
async def stack_agent(trained_model) -> Agent:
    return await load_agent(model_path=trained_model)


@pytest.fixture
async def core_agent(trained_core_model) -> Agent:
    return await load_agent(model_path=trained_core_model)


@pytest.fixture
async def nlu_agent(trained_nlu_model) -> Agent:
    return await load_agent(model_path=trained_nlu_model)


@pytest.fixture
async def default_agent(
    tmpdir_factory, default_stories_file, default_domain_path
) -> Agent:
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent(
        default_domain_path,
        policies=[MemoizationPolicy()],
        interpreter=RegexInterpreter(),
        tracker_store=InMemoryTrackerStore(default_domain_path),
    )

    training_data = await agent.load_data(default_stories_file)
    agent.train(training_data)
    agent.persist(model_path)

    return agent


#############
# RASA SERVER
#############


@pytest.fixture
async def rasa_server(stack_agent):
    app = server.create_app(agent=stack_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_core_server(core_agent):
    app = server.create_app(agent=core_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_nlu_server(nlu_agent):
    app = server.create_app(agent=nlu_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_server_secured(default_agent):
    app = server.create_app(agent=default_agent, auth_token="rasa", jwt_secret="core")
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_server_without_api():
    app = _create_app_without_api()
    channel.register([RestInput()], app, "/webhooks/")
    return app


################
# TRAINED MODELS
################


@pytest.fixture
async def trained_model(project) -> Text:
    return await train_model(project)


@pytest.fixture
async def trained_core_model(project) -> Text:
    return await train_model(project, model_type="core")


@pytest.fixture
async def trained_nlu_model(project) -> Text:
    return await train_model(project, model_type="nlu")


@pytest.fixture(scope="session")
async def trained_moodbot_path():
    return await train_async(
        domain="examples/moodbot/domain.yml",
        config="examples/moodbot/config.yml",
        training_files="examples/moodbot/data/",
        output_path=MOODBOT_MODEL_PATH,
    )


@pytest.fixture(scope="session")
async def unpacked_trained_moodbot_path(trained_moodbot_path):
    return get_model(trained_moodbot_path)


async def train_model(
    project: Text, filename: Text = "test.tar.gz", model_type: Text = "stack"
):
    output = os.path.join(project, DEFAULT_MODELS_PATH, filename)
    domain = os.path.join(project, DEFAULT_DOMAIN_PATH)
    config = os.path.join(project, DEFAULT_CONFIG_PATH)

    if model_type == "core":
        training_files = os.path.join(project, DEFAULT_DATA_PATH, "stories.md")
    elif model_type == "nlu":
        training_files = os.path.join(project, DEFAULT_DATA_PATH, "nlu.md")
    else:
        training_files = os.path.join(project, DEFAULT_DATA_PATH)

    await train_async(domain, config, training_files, output)

    return output
