import asyncio
import random
import pytest
import sys
import uuid

from sanic.request import Request
from sanic.testing import SanicTestClient

from typing import Iterator, Callable

from _pytest.tmpdir import TempdirFactory
from pathlib import Path
from sanic import Sanic
from typing import Text, List, Optional, Dict, Any
from unittest.mock import Mock

from rasa import server
from rasa.core import config
from rasa.core.agent import Agent, load_agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels import channel, RestInput
from rasa.core.domain import SessionConfig
from rasa.core.events import UserUttered
from rasa.core.exporter import Exporter
from rasa.core.policies import Policy
from rasa.core.policies.memoization import AugmentedMemoizationPolicy
import rasa.core.run
from rasa.core.tracker_store import InMemoryTrackerStore, TrackerStore
from rasa.model import get_model
from rasa.train import train_async
from rasa.utils.common import TempDirectoryPath
import rasa.utils.io as io_utils
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_STACK_CONFIG,
    DEFAULT_STORIES_FILE,
    END_TO_END_STORY_FILE,
    INCORRECT_NLU_DATA,
)

DEFAULT_CONFIG_PATH = "rasa/cli/default_config.yml"

DEFAULT_NLU_DATA = "examples/moodbot/data/nlu.yml"

# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.
pytest_plugins = ["pytester"]


# these tests are run separately
collect_ignore_glob = ["docs/*.py"]


# https://github.com/pytest-dev/pytest-asyncio/issues/68
# this event_loop is used by pytest-asyncio, and redefining it
# is currently the only way of changing the scope of this fixture
@pytest.yield_fixture(scope="session")
def event_loop(request: Request) -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def _trained_default_agent(tmpdir_factory: TempdirFactory) -> Agent:
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent(
        "data/test_domains/default_with_slots.yml",
        policies=[AugmentedMemoizationPolicy(max_history=3)],
    )

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    agent.persist(model_path)
    return agent


def reset_conversation_state(agent: Agent) -> Agent:
    # Clean tracker store after each test so tests don't affect each other
    agent.tracker_store = InMemoryTrackerStore(agent.domain)
    agent.domain.session_config = SessionConfig.default()
    return agent


@pytest.fixture
async def default_agent(_trained_default_agent: Agent) -> Agent:
    return reset_conversation_state(_trained_default_agent)


@pytest.fixture(scope="session")
async def trained_moodbot_path(trained_async) -> Text:
    return await trained_async(
        domain="examples/moodbot/domain.yml",
        config="examples/moodbot/config.yml",
        training_files="examples/moodbot/data/",
    )


@pytest.fixture(scope="session")
async def unpacked_trained_moodbot_path(
    trained_moodbot_path: Text,
) -> TempDirectoryPath:
    return get_model(trained_moodbot_path)


@pytest.fixture(scope="session")
async def stack_agent(trained_rasa_model: Text) -> Agent:
    return await load_agent(model_path=trained_rasa_model)


@pytest.fixture(scope="session")
async def core_agent(trained_core_model: Text) -> Agent:
    return await load_agent(model_path=trained_core_model)


@pytest.fixture(scope="session")
async def nlu_agent(trained_nlu_model: Text) -> Agent:
    return await load_agent(model_path=trained_nlu_model)


@pytest.fixture(scope="session")
def default_domain_path() -> Text:
    return DEFAULT_DOMAIN_PATH_WITH_SLOTS


@pytest.fixture(scope="session")
def default_stories_file() -> Text:
    return DEFAULT_STORIES_FILE


@pytest.fixture(scope="session")
def default_stack_config() -> Text:
    return DEFAULT_STACK_CONFIG


@pytest.fixture(scope="session")
def default_nlu_data() -> Text:
    return DEFAULT_NLU_DATA


@pytest.fixture(scope="session")
def incorrect_nlu_data() -> Text:
    return INCORRECT_NLU_DATA


@pytest.fixture(scope="session")
def end_to_end_story_file() -> Text:
    return END_TO_END_STORY_FILE


@pytest.fixture(scope="session")
def default_config() -> List[Policy]:
    return config.load(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="session")
def trained_async(tmpdir_factory: TempdirFactory) -> Callable:
    async def _train(
        *args: Any, output_path: Optional[Text] = None, **kwargs: Any
    ) -> Optional[Text]:
        if output_path is None:
            output_path = str(tmpdir_factory.mktemp("models"))

        return await train_async(*args, output_path=output_path, **kwargs)

    return _train


@pytest.fixture(scope="session")
async def trained_rasa_model(
    trained_async: Callable,
    default_domain_path: Text,
    default_nlu_data: Text,
    default_stories_file: Text,
) -> Text:
    trained_stack_model_path = await trained_async(
        domain=default_domain_path,
        config=DEFAULT_STACK_CONFIG,
        training_files=[default_nlu_data, default_stories_file],
    )

    return trained_stack_model_path


@pytest.fixture(scope="session")
async def trained_core_model(
    trained_async: Callable,
    default_domain_path: Text,
    default_nlu_data: Text,
    default_stories_file: Text,
) -> Text:
    trained_core_model_path = await trained_async(
        domain=default_domain_path,
        config=DEFAULT_STACK_CONFIG,
        training_files=[default_stories_file],
    )

    return trained_core_model_path


@pytest.fixture(scope="session")
async def trained_nlu_model(
    trained_async: Callable,
    default_domain_path: Text,
    default_config: List[Policy],
    default_nlu_data: Text,
    default_stories_file: Text,
) -> Text:
    trained_nlu_model_path = await trained_async(
        domain=default_domain_path,
        config=DEFAULT_STACK_CONFIG,
        training_files=[default_nlu_data],
    )

    return trained_nlu_model_path


@pytest.fixture
async def rasa_server(stack_agent: Agent) -> Sanic:
    app = server.create_app(agent=stack_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_core_server(core_agent: Agent) -> Sanic:
    app = server.create_app(agent=core_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_nlu_server(nlu_agent: Agent) -> Sanic:
    app = server.create_app(agent=nlu_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_server_secured(default_agent: Agent) -> Sanic:
    app = server.create_app(agent=default_agent, auth_token="rasa", jwt_secret="core")
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
async def rasa_server_without_api() -> Sanic:
    app = rasa.core.run._create_app_without_api()
    channel.register([RestInput()], app, "/webhooks/")
    return app


def get_test_client(server: Sanic) -> SanicTestClient:
    test_client = server.test_client
    test_client.port = None
    return test_client


def write_endpoint_config_to_yaml(
    path: Path, data: Dict[Text, Any], endpoints_filename: Text = "endpoints.yml"
) -> Path:
    endpoints_path = path / endpoints_filename

    # write endpoints config to file
    io_utils.write_yaml(data, endpoints_path)
    return endpoints_path


def random_user_uttered_event(timestamp: Optional[float] = None) -> UserUttered:
    return UserUttered(
        uuid.uuid4().hex,
        timestamp=timestamp if timestamp is not None else random.random(),
    )


def pytest_runtest_setup(item) -> None:
    if (
        "skip_on_windows" in [mark.name for mark in item.iter_markers()]
        and sys.platform == "win32"
    ):
        pytest.skip("cannot run on Windows")


class MockExporter(Exporter):
    """Mocked `Exporter` object."""

    def __init__(
        self,
        tracker_store: TrackerStore = Mock(),
        event_broker: EventBroker = Mock(),
        endpoints_path: Text = "",
    ) -> None:
        super().__init__(tracker_store, event_broker, endpoints_path)
