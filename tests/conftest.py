import asyncio
import contextlib
import copy
import os
import random
import re
import textwrap

import jwt
import pytest
import sys
import uuid

from pytest import TempdirFactory, MonkeyPatch, Function, TempPathFactory
from spacy import Language
from pytest import WarningsRecorder

from rasa.engine.caching import LocalTrainingCache
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelStorage
from sanic.request import Request

from typing import Generator, Iterator, Callable

from pathlib import Path
from sanic import Sanic
from typing import Text, List, Optional, Dict, Any
from unittest.mock import Mock

from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import METADATA_MODEL_ID
import rasa.shared.utils.io
from rasa import server
from rasa.core.agent import Agent, load_agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels import channel, RestInput
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from rasa.nlu.utils.spacy_utils import SpacyNLP, SpacyModel
from rasa.shared.constants import ASSISTANT_ID_KEY, LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import SessionConfig, Domain
from rasa.shared.core.events import (
    ActionExecuted,
    Event,
    Restarted,
    SessionStarted,
    UserUttered,
)
from rasa.core.exporter import Exporter

import rasa.core.run
from rasa.core.tracker_store import InMemoryTrackerStore, TrackerStore
from rasa.model_training import train, train_nlu
from rasa.shared.exceptions import RasaException
import rasa.utils.common
import rasa.utils.io


# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separately installable pytest-cli plugin.
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
        Path("tests", "test_model_training.py").absolute(),
        Path("tests", "nlu", "test_train.py").absolute(),
        Path("tests", "core", "test_training.py").absolute(),
        Path("tests", "core", "test_examples.py").absolute(),
    ],
    "category_performance": [Path("tests", "test_memory_leak.py").absolute()],
}


@pytest.fixture(scope="session")
def nlu_as_json_path() -> Text:
    return "data/examples/rasa/demo-rasa.json"


@pytest.fixture(scope="session")
def nlu_data_path() -> Text:
    return "data/test_moodbot/data/nlu.yml"


@pytest.fixture(scope="session")
def config_path() -> Text:
    return "rasa/engine/recipes/config_files/default_config.yml"


@pytest.fixture(scope="session")
def default_config(config_path: Text) -> Dict[Text, Any]:
    return rasa.shared.utils.io.read_yaml_file(config_path)


@pytest.fixture(scope="session")
def domain_with_categorical_slot_path() -> Text:
    return "data/test_domains/domain_with_categorical_slot.yml"


@pytest.fixture(scope="session")
def domain_with_mapping_path() -> Text:
    return "data/test_domains/default_with_mapping.yml"


@pytest.fixture(scope="session")
def stories_path() -> Text:
    return "data/test_yaml_stories/stories_defaultdomain.yml"


@pytest.fixture(scope="session")
def e2e_stories_path() -> Text:
    return "data/test_yaml_stories/stories_e2e.yml"


@pytest.fixture(scope="session")
def simple_stories_path() -> Text:
    return "data/test_yaml_stories/stories_simple.yml"


@pytest.fixture(scope="session")
def stack_config_path() -> Text:
    return "data/test_config/stack_config.yml"


@pytest.fixture(scope="session")
def incorrect_nlu_data_path() -> Text:
    return "data/test/incorrect_nlu_format.yml"


@pytest.fixture(scope="session")
def end_to_end_story_path() -> Text:
    return "data/test_evaluations/test_end_to_end_story.yml"


@pytest.fixture(scope="session")
def e2e_story_file_unknown_entity_path() -> Text:
    return "data/test_evaluations/test_story_unknown_entity.yml"


@pytest.fixture(scope="session")
def domain_path() -> Text:
    return "data/test_domains/default_with_slots.yml"


@pytest.fixture(scope="session")
def simple_config_path(tmp_path_factory: TempPathFactory) -> Text:
    project_path = tmp_path_factory.mktemp(uuid.uuid4().hex)

    config = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        assistant_id: placeholder_default
        pipeline:
        - name: WhitespaceTokenizer
        - name: KeywordIntentClassifier
        - name: RegexEntityExtractor
        policies:
        - name: AugmentedMemoizationPolicy
          max_history: 3
        - name: RulePolicy
        """
    )
    config_path = project_path / "config.yml"
    rasa.shared.utils.io.write_text_file(config, config_path)

    return str(config_path)


@pytest.fixture(scope="session")
def story_file_trips_circuit_breaker_path() -> Text:
    return "data/test_evaluations/test_stories_trip_circuit_breaker.yml"


@pytest.fixture(scope="session")
def e2e_story_file_trips_circuit_breaker_path() -> Text:
    return "data/test_evaluations/test_end_to_end_trips_circuit_breaker.yml"


@pytest.fixture(scope="session")
def endpoints_path() -> Text:
    return "data/test_endpoints/example_endpoints.yml"


# https://github.com/pytest-dev/pytest-asyncio/issues/68
# this event_loop is used by pytest-asyncio, and redefining it
# is currently the only way of changing the scope of this fixture
# update: implement fix to RuntimeError Event loop is closed issue described
# here: https://github.com/pytest-dev/pytest-asyncio/issues/371
@pytest.fixture(scope="session")
def event_loop(request: Request) -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    loop._close = loop.close
    loop.close = lambda: None
    yield loop
    loop._close()


# override loop fixture to prevent ScopeMismatch pytest error and
# align the result of the loop fixture with that of the event_loop fixture
@pytest.fixture(scope="session")
def loop(
    event_loop: asyncio.AbstractEventLoop,
) -> Generator[asyncio.AbstractEventLoop, None, None]:
    yield event_loop


@pytest.fixture(scope="session")
async def trained_default_agent_model(
    stories_path: Text,
    domain_path: Text,
    nlu_data_path: Text,
    trained_async: Callable,
    simple_config_path: Text,
) -> Text:
    model_path = await trained_async(
        domain_path, simple_config_path, [stories_path, nlu_data_path]
    )

    return model_path


@pytest.fixture()
def empty_agent() -> Agent:
    agent = Agent(domain=Domain.load("data/test_domains/default_with_slots.yml"))
    return agent


def reset_conversation_state(agent: Agent) -> Agent:
    # Clean tracker store after each test so tests don't affect each other
    agent.tracker_store = InMemoryTrackerStore(agent.domain)
    agent.domain.session_config = SessionConfig.default()
    agent.load_model(agent.processor.model_path)
    return agent


@pytest.fixture
def default_agent(trained_default_agent_model: Text) -> Agent:
    return Agent.load(trained_default_agent_model)


@pytest.fixture(scope="session")
async def trained_moodbot_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="data/test_moodbot/domain.yml",
        config="data/test_moodbot/config.yml",
        training_files="data/test_moodbot/data/",
    )


@pytest.fixture(scope="session")
async def trained_moodbot_core_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="data/test_moodbot/domain.yml",
        config="data/test_moodbot/config.yml",
        training_files="data/test_moodbot/data/stories.yml",
    )


@pytest.fixture(scope="session")
async def trained_moodbot_nlu_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="data/test_moodbot/domain.yml",
        config="data/test_moodbot/config.yml",
        training_files="data/test_moodbot/data/nlu.yml",
    )


@pytest.fixture(scope="session")
async def trained_unexpected_intent_policy_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="data/test_moodbot/domain.yml",
        config="data/test_moodbot/unexpected_intent_policy_config.yml",
        training_files="data/test_moodbot/data/",
    )


@pytest.fixture(scope="session")
def trained_nlu_moodbot_path(trained_nlu: Callable) -> Text:
    return trained_nlu(
        domain="data/test_moodbot/domain.yml",
        config="data/test_moodbot/config.yml",
        nlu_data="data/test_moodbot/data/nlu.yml",
    )


@pytest.fixture(scope="session")
async def trained_spacybot_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="data/test_spacybot/domain.yml",
        config="data/test_spacybot/config.yml",
        training_files="data/test_spacybot/data/",
    )


@pytest.fixture(scope="session")
async def stack_agent(trained_rasa_model: Text) -> Agent:
    return await load_agent(model_path=trained_rasa_model)


@pytest.fixture(scope="session")
async def core_agent(trained_core_model: Text) -> Agent:
    return await load_agent(model_path=trained_core_model)


@pytest.fixture(scope="session")
async def nlu_agent(trained_nlu_model: Text) -> Agent:
    return await load_agent(model_path=trained_nlu_model)


@pytest.fixture(scope="module")
async def unexpected_intent_policy_agent(
    trained_unexpected_intent_policy_path: Text,
) -> Agent:
    return await load_agent(model_path=trained_unexpected_intent_policy_path)


@pytest.fixture(scope="module")
async def mood_agent(trained_moodbot_path: Text) -> Agent:
    return await load_agent(model_path=trained_moodbot_path)


@pytest.fixture(scope="session")
def _domain(domain_path: Text) -> Domain:
    return Domain.load(domain_path)


@pytest.fixture()
def domain(_domain: Domain) -> Domain:
    return copy.deepcopy(_domain)


@pytest.fixture(scope="session")
def trained_async(tmp_path_factory: TempPathFactory) -> Callable:
    async def _train(
        *args: Any,
        output_path: Optional[Text] = None,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> Optional[Text]:
        if not cache_dir:
            cache_dir = tmp_path_factory.mktemp("cache")

        if output_path is None:
            output_path = str(tmp_path_factory.mktemp("models"))

        with enable_cache(cache_dir):
            result = train(*args, output=output_path, **kwargs)

        return result.model

    return _train


@pytest.fixture(scope="session")
def trained_nlu(tmp_path_factory: TempPathFactory) -> Callable:
    def _train_nlu(
        *args: Any, output_path: Optional[Text] = None, **kwargs: Any
    ) -> Optional[Text]:
        if output_path is None:
            output_path = str(tmp_path_factory.mktemp("models"))

        return train_nlu(*args, output=output_path, **kwargs)

    return _train_nlu


@pytest.fixture(scope="session")
async def trained_rasa_model(
    trained_async: Callable,
    domain_path: Text,
    nlu_data_path: Text,
    stories_path: Text,
    stack_config_path: Text,
) -> Text:
    trained_stack_model_path = await trained_async(
        domain=domain_path,
        config=stack_config_path,
        training_files=[nlu_data_path, stories_path],
    )

    return trained_stack_model_path


@pytest.fixture(scope="session")
async def trained_core_model(
    trained_async: Callable,
    domain_path: Text,
    stack_config_path: Text,
    stories_path: Text,
) -> Text:
    trained_core_model_path = await trained_async(
        domain=domain_path, config=stack_config_path, training_files=[stories_path]
    )

    return trained_core_model_path


@pytest.fixture(scope="session")
async def trained_nlu_model(
    trained_async: Callable,
    domain_path: Text,
    nlu_data_path: Text,
    stack_config_path: Text,
) -> Text:
    trained_nlu_model_path = await trained_async(
        domain=domain_path, config=stack_config_path, training_files=[nlu_data_path]
    )

    return trained_nlu_model_path


@pytest.fixture(scope="session")
def _trained_e2e_model_cache(tmp_path_factory: TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture()
def trained_e2e_model_cache(
    _trained_e2e_model_cache: Path,
    tmp_path_factory: TempPathFactory,
    monkeypatch: MonkeyPatch,
) -> Path:
    copied_cache = tmp_path_factory.mktemp("copy")
    rasa.utils.common.copy_directory(_trained_e2e_model_cache, copied_cache)

    with enable_cache(copied_cache):
        yield copied_cache


@pytest.fixture(scope="session")
async def trained_e2e_model(
    trained_async: Callable,
    moodbot_domain_path: Text,
    e2e_bot_config_file: Path,
    nlu_data_path: Text,
    e2e_stories_path: Text,
    _trained_e2e_model_cache: Path,
) -> Text:
    return await trained_async(
        domain=moodbot_domain_path,
        config=str(e2e_bot_config_file),
        training_files=[nlu_data_path, e2e_stories_path],
        cache_dir=_trained_e2e_model_cache,
    )


@pytest.fixture(scope="session")
def moodbot_domain_path() -> Path:
    return Path("data", "test_moodbot", "domain.yml")


@pytest.fixture(scope="session")
def moodbot_domain(moodbot_domain_path: Path) -> Domain:
    return Domain.load(moodbot_domain_path)


@pytest.fixture(scope="session")
def moodbot_nlu_data_path() -> Path:
    return Path(os.getcwd()) / "data" / "test_moodbot" / "data" / "nlu.yml"


@pytest.fixture
def rasa_server(stack_agent: Agent) -> Sanic:
    app = server.create_app(agent=stack_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_non_trained_server(empty_agent: Agent) -> Sanic:
    app = server.create_app(agent=empty_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_core_server(core_agent: Agent) -> Sanic:
    app = server.create_app(agent=core_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_nlu_server(nlu_agent: Agent) -> Sanic:
    app = server.create_app(agent=nlu_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_server_secured(default_agent: Agent) -> Sanic:
    app = server.create_app(agent=default_agent, auth_token="rasa", jwt_secret="core")
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def test_public_key() -> Text:
    test_public_key = """-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC34ht9inqGq79HecpyOAnu2Cgv
jvgcpFifpFLPmCNdiomAgE48tfUAXJRoOGlVtrqc8KgQWjTFLjqDjUh1sBFF69Fl
wQGt7pgH10ZbERWpMTAbpjI9EoH74gDcmZ6Fy1VgQPbAwty3liw5Q5zqZLj7JhuX
Sa0EqvZQP+Hnayab7QIDAQAB
-----END PUBLIC KEY-----"""

    return test_public_key


@pytest.fixture
def test_private_key() -> Text:
    test_private_key = """-----BEGIN RSA PRIVATE KEY-----
MIICXQIBAAKBgQC34ht9inqGq79HecpyOAnu2CgvjvgcpFifpFLPmCNdiomAgE48
tfUAXJRoOGlVtrqc8KgQWjTFLjqDjUh1sBFF69FlwQGt7pgH10ZbERWpMTAbpjI9
EoH74gDcmZ6Fy1VgQPbAwty3liw5Q5zqZLj7JhuXSa0EqvZQP+Hnayab7QIDAQAB
AoGBAIfUE25mjh9QWljX0/0O+/db4ENRHmE53OT/otQJk4YTQYKURDaASdvchxt9
IAHamno3Ik4B9Bz7CuoFwNJ+HiMBf32KwJ75n/NZL17lBKst71z3r0gYCz6jcJxv
brbNs8qsLFyRMQz6NvS4d4GnXpGhc54IoJqtr/vR+Q87UwtZAkEA3AG78E7Fd5zT
sU/BO9E0VisQOysGcwPd9+rQPSyF8ncvaiMJ7STNvVsgrtJuw4DJq2RsMSJ77QgS
Ku6BJxB58wJBANX3dOEiNEZLJR+4LdNYRoR4gx2LcJW5PthwLi8ZOHBZeh9q3f2i
r5X5iPJ5kBRqajtYm634f/j8P4fxSdWzKp8CQQCNimQR92udR3z+HxRvWml0YmIf
3s9YYY2FeUEdii5mznznqMEzGzFt+Fmvf1yZVJrqNEJS3h+iYEXn7ueSbUw3AkBm
xSK4d+tP0AwWvioUlxPX0OJ5MF51K7LJ1qf4K072d6O2r2fMyXU4vdBPVqAjjjFU
K+0qlG8zMkV5kCV8pT/VAkA8bM5KRa73JY0bfGX4i8UZMFHzIq2KGjHlRES4vd+L
h18+hpcBAAyUR/jDT8nnG5YaYFz8rf2DnOy+elmmaYVm
-----END RSA PRIVATE KEY-----"""

    return test_private_key


@pytest.fixture
def asymmetric_jwt_method() -> Text:
    return "RS256"


@pytest.fixture
def rasa_server_secured_asymmetric(
    default_agent: Agent,
    test_public_key: Text,
    test_private_key: Text,
    asymmetric_jwt_method: Text,
) -> Sanic:
    app = server.create_app(
        agent=default_agent,
        auth_token="rasa",
        jwt_secret=test_public_key,
        jwt_private_key=test_private_key,
        jwt_method=asymmetric_jwt_method,
    )
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def encoded_jwt(test_private_key: Text, asymmetric_jwt_method: Text) -> Text:
    payload = {"user": {"username": "myuser", "role": "admin"}}
    encoded_jwt = jwt.encode(
        payload=payload,
        key=test_private_key,
        algorithm=asymmetric_jwt_method,
    )
    return encoded_jwt


@pytest.fixture
def rasa_non_trained_server_secured(empty_agent: Agent) -> Sanic:
    app = server.create_app(agent=empty_agent, auth_token="rasa", jwt_secret="core")
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_server_without_api() -> Sanic:
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
def spacy_nlp_component() -> SpacyNLP:
    return SpacyNLP.create({"model": "en_core_web_md"}, Mock(), Mock(), Mock())


@pytest.fixture(scope="session")
def spacy_case_sensitive_nlp_component() -> SpacyNLP:
    return SpacyNLP.create(
        {"model": "en_core_web_md", "case_sensitive": True}, Mock(), Mock(), Mock()
    )


@pytest.fixture(scope="session")
def spacy_model(spacy_nlp_component: SpacyNLP) -> SpacyModel:
    return spacy_nlp_component.provide()


@pytest.fixture(scope="session")
def spacy_nlp(spacy_model: SpacyModel) -> Language:
    return spacy_model.model


@pytest.fixture(scope="session")
async def response_selector_test_stories() -> Path:
    return Path("data/test_response_selector_bot/tests/test_stories.yml")


@pytest.fixture(scope="session")
async def trained_response_selector_bot(trained_async: Callable) -> Path:
    zipped_model = await trained_async(
        domain="data/test_response_selector_bot/domain.yml",
        config="data/test_response_selector_bot/config.yml",
        training_files=[
            "data/test_response_selector_bot/data/rules.yml",
            "data/test_response_selector_bot/data/nlu.yml",
        ],
    )

    if not zipped_model:
        raise RasaException("Model training for responseselectorbot failed.")

    return Path(zipped_model)


@pytest.fixture(scope="session")
def e2e_bot_domain_file() -> Path:
    return Path("data/test_e2ebot/domain.yml")


@pytest.fixture(scope="session")
def e2e_bot_config_file() -> Path:
    return Path("data/test_e2ebot/config.yml")


@pytest.fixture(scope="session")
def e2e_bot_training_files() -> List[Path]:
    return [
        Path("data/test_e2ebot/data/stories.yml"),
        Path("data/test_e2ebot/data/nlu.yml"),
    ]


@pytest.fixture(scope="session")
def e2e_bot_test_stories_with_unknown_bot_utterances() -> Path:
    return Path("data/test_e2ebot/tests/test_stories_with_unknown_bot_utterances.yml")


# FIXME: This fixture is very slow, do not use it without fixing that first
@pytest.fixture(scope="session")
async def e2e_bot(
    trained_async: Callable,
    e2e_bot_domain_file: Path,
    e2e_bot_config_file: Path,
    e2e_bot_training_files: List[Path],
) -> Path:
    zipped_model = await trained_async(
        domain=e2e_bot_domain_file,
        config=e2e_bot_config_file,
        training_files=e2e_bot_training_files,
    )

    if not zipped_model:
        raise RasaException("Model training for e2ebot failed.")

    return Path(zipped_model)


@pytest.fixture(scope="module")
async def response_selector_agent(trained_response_selector_bot: Path) -> Agent:
    return await load_agent(str(trained_response_selector_bot))


@pytest.fixture(scope="module")
async def e2e_bot_agent(e2e_bot: Path) -> Agent:
    return await load_agent(str(e2e_bot))


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


def pytest_runtest_setup(item: Function) -> None:
    if (
        "skip_on_windows" in [mark.name for mark in item.iter_markers()]
        and sys.platform == "win32"
    ):
        pytest.skip("cannot run on Windows")
    if "skip_on_ci" in [mark.name for mark in item.iter_markers()] and os.environ.get(
        "CI"
    ) in ["true", "True", "yes", "t", "1"]:
        pytest.skip("cannot run on CI")


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


def _get_marker_for_ci_matrix(item: Function) -> Text:
    """Returns pytest marker which is used to parallelize the tests in GitHub actions.

    Args:
        item: The test case.

    Returns:
        A marker for this test based on which directory / python module the test is in.
    """
    test_path = Path(item.fspath).absolute()

    matching_markers = [
        marker
        for marker, paths_for_marker in PATH_PYTEST_MARKER_MAPPINGS.items()
        if any(
            path == test_path or path in test_path.parents for path in paths_for_marker
        )
    ]

    if not matching_markers:
        return "category_other_unit_tests"
    if len(matching_markers) > 1:
        raise ValueError(
            f"Each test should only be in one category. Test '{item.name}' is assigned "
            f"to these categories: {matching_markers}. Please fix the "
            "mapping in `PATH_PYTEST_MARKER_MAPPINGS`."
        )

    return matching_markers[0]


def pytest_collection_modifyitems(items: List[Function]) -> None:
    """Adds pytest markers dynamically when the tests are run.

    This is automatically called by pytest during its execution.

    Args:
        items: Tests to be run.
    """
    for item in items:
        marker = _get_marker_for_ci_matrix(item)
        item.add_marker(marker)


def create_test_file_with_size(directory: Path, size_in_mb: float) -> Path:
    file_path = directory / uuid.uuid4().hex
    with open(file_path, mode="wb") as f:
        f.seek(int(1024 * 1024 * size_in_mb))
        f.write(b"\0")

    return file_path


@pytest.fixture()
def default_model_storage(tmp_path: Path, monkeypatch: MonkeyPatch) -> ModelStorage:
    return LocalModelStorage.create(tmp_path)


@pytest.fixture()
def default_execution_context() -> ExecutionContext:
    return ExecutionContext(GraphSchema({}), uuid.uuid4().hex)


@pytest.fixture(scope="session", autouse=True)
def temp_cache_for_fixtures(tmp_path_factory: TempPathFactory) -> None:
    # This fixture makes sure that wide fixtures which don't have `function` scope
    # (session, package, module) don't use the global
    # cache. If you want to use the cache in a session scoped fixture, then please
    # consider using the `enable_cache` context manager.
    LocalTrainingCache._get_cache_location = lambda: tmp_path_factory.mktemp(
        f"cache-{uuid.uuid4()}"
    )

    # We can omit reverting the monkeypatch as this fixture is torn down after all the
    # tests ran


@pytest.fixture(autouse=True)
def use_temp_dir_for_cache(
    monkeypatch: MonkeyPatch, tmp_path_factory: TempdirFactory
) -> None:
    # This fixture makes sure that a single test function has a constant cache
    # cache.
    cache_dir = tmp_path_factory.mktemp(uuid.uuid4().hex)
    monkeypatch.setattr(LocalTrainingCache, "_get_cache_location", lambda: cache_dir)


@contextlib.contextmanager
def enable_cache(cache_dir: Path):
    old_get_cache_location = LocalTrainingCache._get_cache_location
    LocalTrainingCache._get_cache_location = Mock(return_value=cache_dir)

    yield

    LocalTrainingCache._get_cache_location = old_get_cache_location


@pytest.fixture()
def whitespace_tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer(WhitespaceTokenizer.get_default_config())


def with_model_ids(events: List[Event], model_id: Text) -> List[Event]:
    return [with_model_id(event, model_id) for event in events]


def with_model_id(event: Event, model_id: Text) -> Event:
    new_event = copy.deepcopy(event)
    new_event.metadata[METADATA_MODEL_ID] = model_id
    return new_event


def with_assistant_id(event: Event, assistant_id: Text) -> Event:
    event.metadata[ASSISTANT_ID_KEY] = assistant_id
    return event


def with_assistant_ids(events: List[Event], assistant_id: Text) -> List[Event]:
    return [with_assistant_id(event, assistant_id) for event in events]


@pytest.fixture(autouse=True)
def sanic_test_mode(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(Sanic, "test_mode", True)


def filter_expected_warnings(records: WarningsRecorder) -> WarningsRecorder:
    records_copy = copy.deepcopy(records.list)

    for warning_type, warning_message in rasa.utils.common.EXPECTED_WARNINGS:
        for record in records_copy:
            if type(record.message) == warning_type and re.search(
                warning_message, str(record.message)
            ):
                records.pop(type(record.message))

    return records


@pytest.fixture
def initial_events_including_restart() -> List[Event]:
    return [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=1),
        SessionStarted(timestamp=2),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=3),
        UserUttered("hi", timestamp=4),
        ActionExecuted("utter_greet", timestamp=5),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=6),
        UserUttered("/restart", timestamp=7),
        Restarted(timestamp=8),
        ActionExecuted(ACTION_RESTART_NAME, timestamp=9),
    ]


@pytest.fixture
def events_after_restart() -> List[Event]:
    return [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=10),
        SessionStarted(timestamp=11),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=12),
        UserUttered("Let's start again.", timestamp=13),
    ]


@pytest.fixture
def tracker_with_restarted_event(
    initial_events_including_restart: List[Event],
    events_after_restart: List[Event],
) -> DialogueStateTracker:
    sender_id = uuid.uuid4().hex
    events = initial_events_including_restart + events_after_restart

    return DialogueStateTracker.from_events(sender_id=sender_id, evts=events)
