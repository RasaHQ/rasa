import asyncio
import copy
import functools
import os
import random
import pytest
import sys
import uuid

from _pytest.python import Function
from sanic.request import Request

from typing import Iterator, Callable, Generator

from _pytest.tmpdir import TempdirFactory
from pathlib import Path
from sanic import Sanic
from typing import Text, List, Optional, Dict, Any
from unittest.mock import Mock

import rasa.shared.utils.io
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa import server
from rasa.core import config
from rasa.core.agent import Agent, load_agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels import channel, RestInput
from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.core.domain import SessionConfig, Domain
from rasa.shared.core.events import UserUttered
from rasa.core.exporter import Exporter
from rasa.core.policies import Policy
from rasa.core.policies.memoization import AugmentedMemoizationPolicy
import rasa.core.run
from rasa.core.tracker_store import InMemoryTrackerStore, TrackerStore
from rasa.model import get_model
from rasa.train import train_async, train_nlu_async
from rasa.utils.common import TempDirectoryPath
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_E2E_STORIES_FILE,
    DEFAULT_STACK_CONFIG,
    DEFAULT_STORIES_FILE,
    DOMAIN_WITH_CATEGORICAL_SLOT,
    END_TO_END_STORY_MD_FILE,
    INCORRECT_NLU_DATA,
    SIMPLE_STORIES_FILE,
)
from rasa.shared.exceptions import RasaException

DEFAULT_CONFIG_PATH = "rasa/shared/importers/default_config.yml"

DEFAULT_NLU_DATA = "examples/moodbot/data/nlu.yml"

# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.
pytest_plugins = ["pytester"]


# these tests are run separately
collect_ignore_glob = ["docs/*.py"]

# Defines how tests are parallelized in the CI
PATH_PYTEST_MARKER_MAPPINGS = {
    "category_cli": [Path("tests", "cli").absolute()],
    "category_core_featurizers": [Path("tests", "core", "featurizers").absolute()],
    "category_policies": [
        Path("tests", "core", "test_policies.py").absolute(),
        Path("tests", "core", "policies").absolute(),
    ],
    "category_nlu_featurizers": [
        Path("tests", "nlu", "featurizers").absolute(),
        Path("tests", "nlu", "utils").absolute(),
    ],
    "category_nlu_predictors": [
        Path("tests", "nlu", "classifiers").absolute(),
        Path("tests", "nlu", "extractors").absolute(),
        Path("tests", "nlu", "selectors").absolute(),
    ],
    "category_full_model_training": [
        Path("tests", "test_train.py").absolute(),
        Path("tests", "nlu", "test_train.py").absolute(),
        Path("tests", "core", "test_training.py").absolute(),
        Path("tests", "core", "test_examples.py").absolute(),
    ],
}


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
        policies=[AugmentedMemoizationPolicy(max_history=3), RulePolicy()],
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
async def trained_moodbot_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="examples/moodbot/domain.yml",
        config="examples/moodbot/config.yml",
        training_files="examples/moodbot/data/",
    )


@pytest.fixture(scope="session")
async def trained_nlu_moodbot_path(trained_nlu_async: Callable) -> Text:
    return await trained_nlu_async(
        domain="examples/moodbot/domain.yml",
        config="examples/moodbot/config.yml",
        nlu_data="examples/moodbot/data/nlu.yml",
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
def domain_with_categorical_slot_path() -> Text:
    return DOMAIN_WITH_CATEGORICAL_SLOT


@pytest.fixture(scope="session")
def _default_domain() -> Domain:
    return Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)


@pytest.fixture()
def default_domain(_default_domain: Domain) -> Domain:
    return copy.deepcopy(_default_domain)


@pytest.fixture(scope="session")
def default_stories_file() -> Text:
    return DEFAULT_STORIES_FILE


@pytest.fixture(scope="session")
def default_e2e_stories_file() -> Text:
    return DEFAULT_E2E_STORIES_FILE


@pytest.fixture(scope="session")
def simple_stories_file() -> Text:
    return SIMPLE_STORIES_FILE


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
def end_to_end_test_story_md_file() -> Text:
    return END_TO_END_STORY_MD_FILE


@pytest.fixture(scope="session")
def default_config_path() -> Text:
    return DEFAULT_CONFIG_PATH


@pytest.fixture(scope="session")
def default_config(default_config_path) -> List[Policy]:
    return config.load(default_config_path)


@pytest.fixture(scope="session")
def trained_async(tmpdir_factory: TempdirFactory) -> Callable:
    async def _train(
        *args: Any, output_path: Optional[Text] = None, **kwargs: Any
    ) -> Optional[Text]:
        if output_path is None:
            output_path = str(tmpdir_factory.mktemp("models"))

        result = await train_async(*args, output=output_path, **kwargs)
        return result.model

    return _train


@pytest.fixture(scope="session")
def trained_nlu_async(tmpdir_factory: TempdirFactory) -> Callable:
    async def _train_nlu(
        *args: Any, output_path: Optional[Text] = None, **kwargs: Any
    ) -> Optional[Text]:
        if output_path is None:
            output_path = str(tmpdir_factory.mktemp("models"))

        return await train_nlu_async(*args, output=output_path, **kwargs)

    return _train_nlu


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
async def trained_simple_rasa_model(
    trained_async: Callable,
    default_domain_path: Text,
    default_nlu_data: Text,
    simple_stories_file: Text,
) -> Text:
    trained_stack_model_path = await trained_async(
        domain=default_domain_path,
        config=DEFAULT_STACK_CONFIG,
        training_files=[default_nlu_data, simple_stories_file],
    )

    return trained_stack_model_path


@pytest.fixture(scope="session")
async def unpacked_trained_rasa_model(
    trained_rasa_model: Text,
) -> Generator[Text, None, None]:
    with get_model(trained_rasa_model) as path:
        yield path


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
    trained_async: Callable, default_domain_path: Text, default_nlu_data: Text,
) -> Text:
    trained_nlu_model_path = await trained_async(
        domain=default_domain_path,
        config=DEFAULT_STACK_CONFIG,
        training_files=[default_nlu_data],
    )

    return trained_nlu_model_path


@pytest.fixture(scope="session")
async def trained_e2e_model(
    trained_async,
    default_domain_path,
    default_stack_config,
    default_nlu_data,
    default_e2e_stories_file,
) -> Text:
    return await trained_async(
        domain=default_domain_path,
        config=default_stack_config,
        training_files=[default_nlu_data, default_e2e_stories_file],
    )


@pytest.fixture(scope="session")
def moodbot_domain() -> Domain:
    domain_path = os.path.join("examples", "moodbot", "domain.yml")
    return Domain.load(domain_path)


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


@pytest.fixture(scope="session")
def project() -> Text:
    import tempfile
    from rasa.cli.scaffold import create_initial_project

    directory = tempfile.mkdtemp()
    create_initial_project(directory)

    return directory


@pytest.fixture(scope="session")
def component_builder():
    return ComponentBuilder()


@pytest.fixture(scope="session")
def spacy_nlp(component_builder: ComponentBuilder, blank_config: RasaNLUModelConfig):
    spacy_nlp_config = {"name": "SpacyNLP"}
    return component_builder.create_component(spacy_nlp_config, blank_config).nlp


@pytest.fixture(scope="session")
def blank_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig({"language": "en", "pipeline": []})


@pytest.fixture(scope="session")
async def trained_response_selector_bot(trained_async: Callable) -> Path:
    zipped_model = await trained_async(
        domain="examples/responseselectorbot/domain.yml",
        config="examples/responseselectorbot/config.yml",
        training_files=[
            "examples/responseselectorbot/data/rules.yml",
            "examples/responseselectorbot/data/stories.yml",
            "examples/responseselectorbot/data/nlu.yml",
        ],
    )

    if not zipped_model:
        raise RasaException("Model training for responseselectorbot failed.")

    return Path(zipped_model)


@pytest.fixture(scope="session")
async def response_selector_agent(
    trained_response_selector_bot: Optional[Path],
) -> Agent:
    return Agent.load_local_model(trained_response_selector_bot)


def write_endpoint_config_to_yaml(
    path: Path, data: Dict[Text, Any], endpoints_filename: Text = "endpoints.yml"
) -> Path:
    endpoints_path = path / endpoints_filename

    # write endpoints config to file
    rasa.shared.utils.io.write_yaml(data, endpoints_path)
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
    """Mocked `Exporter` class."""

    def __init__(
        self,
        tracker_store: TrackerStore = Mock(),
        event_broker: EventBroker = Mock(),
        endpoints_path: Text = "",
    ) -> None:
        super().__init__(tracker_store, event_broker, endpoints_path)


class AsyncMock(Mock):
    """Helper class to mock async functions and methods."""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


def raise_on_unexpected_train(f: Callable) -> Callable:
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if os.environ.get("RAISE_ON_TRAIN") == "True":
            raise ValueError(
                "Training called and RAISE_ON_TRAIN is set. "
                "See https://github.com/RasaHQ/rasa#tests-that-train"
            )
        return f(*args, **kwargs)

    return decorated


def wrap_training_methods() -> None:
    """Wrap methods that train so they fail if RAISE_ON_TRAIN is set.

        See "Tests that train" section in rasa/README.md.
    """
    import rasa.nlu as nlu
    import rasa.core as core
    from rasa.nlu.model import Trainer
    from rasa.core.agent import Agent

    for training_module in [nlu, core, Trainer, Agent]:
        training_module.train = raise_on_unexpected_train(training_module.train)


def pytest_configure() -> None:
    wrap_training_methods()


def _get_marker_for_ci_matrix(item: Function) -> Text:
    """Returns pytest marker which is used to parallelize the tests in GitHub actions.

    Args:
        item: The test case.

    Returns:
        A marker for this test based on which directory / python module the test is in.
    """
    test_path = Path(item.fspath).absolute()

    for marker, paths_for_marker in PATH_PYTEST_MARKER_MAPPINGS.items():
        if any(
            path == test_path or path in test_path.parents for path in paths_for_marker
        ):
            return marker

    return "category_other_unit_tests"


def pytest_collection_modifyitems(items: List[Function]) -> None:
    """Adds pytest markers dynamically when the tests are run.

    Args:
        items: The list of tests to be run.
    """
    for item in items:
        marker = _get_marker_for_ci_matrix(item)
        item.add_marker(marker)
