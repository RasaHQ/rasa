import logging
from contextlib import contextmanager
from typing import Text, List

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.tmpdir import TempdirFactory
from sanic import Sanic

from rasa import server
from rasa.core import config
from rasa.core.agent import Agent, load_agent
from rasa.core.channels import channel
from rasa.core.channels.channel import RestInput
from rasa.core.policies import Policy
from rasa.core.policies.memoization import AugmentedMemoizationPolicy
from rasa.core.run import _create_app_without_api
from rasa.model import get_model
from rasa.train import train_async
from rasa.utils.common import TempDirectoryPath
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_NLU_DATA,
    DEFAULT_STACK_CONFIG,
    DEFAULT_STORIES_FILE,
    END_TO_END_STORY_FILE,
    MOODBOT_MODEL_PATH,
)

DEFAULT_CONFIG_PATH = "rasa/cli/default_config.yml"

# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.
pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def set_log_level_debug(caplog: LogCaptureFixture) -> None:
    # Set the post-test log level to DEBUG for failing tests.  For all tests
    # (failing and successful), the live log level can be additionally set in
    # `setup.cfg`. It should be set to WARNING.
    caplog.set_level(logging.DEBUG)


@pytest.fixture
async def default_agent(tmpdir_factory: TempdirFactory) -> Agent:
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent(
        "data/test_domains/default.yml",
        policies=[AugmentedMemoizationPolicy(max_history=3)],
    )

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    agent.persist(model_path)
    return agent


@pytest.fixture(scope="session")
async def trained_moodbot_path() -> Text:
    return await train_async(
        domain="examples/moodbot/domain.yml",
        config="examples/moodbot/config.yml",
        training_files="examples/moodbot/data/",
        output_path=MOODBOT_MODEL_PATH,
    )


@pytest.fixture(scope="session")
async def unpacked_trained_moodbot_path(
    trained_moodbot_path: Text,
) -> TempDirectoryPath:
    return get_model(trained_moodbot_path)


@pytest.fixture
async def stack_agent(trained_rasa_model: Text) -> Agent:
    return await load_agent(model_path=trained_rasa_model)


@pytest.fixture
async def core_agent(trained_core_model: Text) -> Agent:
    return await load_agent(model_path=trained_core_model)


@pytest.fixture
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
def end_to_end_story_file() -> Text:
    return END_TO_END_STORY_FILE


@pytest.fixture(scope="session")
def default_config() -> List[Policy]:
    return config.load(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="session")
def trained_async(tmpdir_factory):
    async def _train(*args, output_path=None, **kwargs):
        if output_path is None:
            output_path = str(tmpdir_factory.mktemp("models"))

        return await train_async(*args, output_path=output_path, **kwargs)

    return _train


@pytest.fixture()
async def trained_rasa_model(
    trained_async,
    default_domain_path: Text,
    default_config: List[Policy],
    default_nlu_data: Text,
    default_stories_file: Text,
) -> Text:
    trained_stack_model_path = await trained_async(
        domain="data/test_domains/default.yml",
        config=DEFAULT_STACK_CONFIG,
        training_files=[default_nlu_data, default_stories_file],
    )

    return trained_stack_model_path


@pytest.fixture()
async def trained_core_model(
    trained_async,
    default_domain_path: Text,
    default_config: List[Policy],
    default_nlu_data: Text,
    default_stories_file: Text,
) -> Text:
    trained_core_model_path = await trained_async(
        domain=default_domain_path,
        config=DEFAULT_STACK_CONFIG,
        training_files=[default_stories_file],
    )

    return trained_core_model_path


@pytest.fixture()
async def trained_nlu_model(
    trained_async,
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
    app = _create_app_without_api()
    channel.register([RestInput()], app, "/webhooks/")
    return app


def get_test_client(server):
    test_client = server.test_client
    test_client.port = None
    return test_client


@contextmanager
def assert_log_emitted(
    _caplog: LogCaptureFixture, logger_name: Text, log_level: int, text: Text = None
) -> None:
    """Context manager testing whether a logging message has been emitted.

    Provides a context in which an assertion is made about a logging message.
    Raises an `AssertionError` if the log isn't emitted as expected.

    Example usage:

    ```
    with assert_log_emitted(caplog, LOGGER_NAME, LOGGING_LEVEL, TEXT):
        <method supposed to emit TEXT at level LOGGING_LEVEL>
    ```

    Args:
        _caplog: `LogCaptureFixture` used to capture logs.
        logger_name: Name of the logger being examined.
        log_level: Log level to be tested.
        text: Logging message to be tested (optional). If left blank, assertion is made
            only about `log_level` and `logger_name`.

    Yields:
        `None`

    """

    yield

    record_tuples = _caplog.record_tuples

    if not any(
        (
            record[0] == logger_name
            and record[1] == log_level
            and (text in record[2] if text else True)
        )
        for record in record_tuples
    ):
        raise AssertionError(
            f"Did not detect expected logging output.\nExpected output is (logger "
            f"name, log level, text): ({logger_name}, {log_level}, {text})\n"
            f"Instead found records:\n{record_tuples}"
        )
