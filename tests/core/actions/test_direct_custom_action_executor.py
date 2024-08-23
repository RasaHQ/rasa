from typing import Callable

import pytest
from pytest import LogCaptureFixture

from rasa.core.actions.action import RemoteAction
from rasa.core.actions.direct_custom_actions_executor import DirectCustomActionExecutor
from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

DUMMY_ACTIONS_MODULE_PATH = "data.dummy_actions_module"
DUMMY_INVALID_ACTIONS_MODULE_PATH = "data.dummy_invalid_actions_module"
DUMMY_ACTION_NAME = "my_action"
DUMMY_DOMAIN_PATH = "data/test_domains/default.yml"

ENDPOINTS_FILE_PATH = "data/test_endpoints/endpoints_actions_module.yml"


@pytest.fixture
def mock_endpoint() -> EndpointConfig:
    return read_endpoint_config(ENDPOINTS_FILE_PATH, endpoint_type="action_endpoint")


@pytest.fixture
def direct_custom_action_executor(
    mock_endpoint: EndpointConfig,
) -> DirectCustomActionExecutor:
    return DirectCustomActionExecutor(
        action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
    )


@pytest.fixture
def remote_action(mock_endpoint: EndpointConfig) -> RemoteAction:
    return RemoteAction(DUMMY_ACTION_NAME, mock_endpoint)


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker(sender_id="test", slots={})


@pytest.fixture
def domain() -> Domain:
    return Domain.from_file(path=DUMMY_DOMAIN_PATH)


def test_executor_initialized_with_valid_actions_module(mock_endpoint: EndpointConfig):
    try:
        DirectCustomActionExecutor(
            action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
        )
    except Exception as exc:
        assert (
            False
        ), f"Instantiating 'DirectCustomActionExecutor' raised an exception {exc}"


async def test_executor_initialized_with_invalid_actions_module(
    tracker: DialogueStateTracker,
    domain: Domain,
):
    endpoint = EndpointConfig(actions_module=DUMMY_INVALID_ACTIONS_MODULE_PATH)
    executor = DirectCustomActionExecutor(
        action_name="some_action", action_endpoint=endpoint
    )

    message = (
        f"You've provided the custom actions module "
        f"'{DUMMY_INVALID_ACTIONS_MODULE_PATH}' to run directly by the rasa server, "
        f"however this module does not exist. "
        f"Please check for typos in your `endpoints.yml` file."
    )
    with pytest.raises(RasaException, match=message):
        await executor.run(tracker, domain)


def test_warning_raised_for_url_and_actions_module_defined():
    endpoint = EndpointConfig(
        url="http://localhost:5055/webhook", actions_module=DUMMY_ACTIONS_MODULE_PATH
    )
    with pytest.warns(
        UserWarning, match="Both 'actions_module' and 'url' are defined."
    ):
        RemoteAction(DUMMY_ACTION_NAME, endpoint)


def test_remote_action_initializes_direct_custom_action_executor(
    remote_action: RemoteAction,
):
    assert isinstance(remote_action.executor, DirectCustomActionExecutor)


def test_remote_action_uses_action_endpoint_with_url_and_actions_module_defined():
    endpoint = EndpointConfig(
        url="http://localhost:5055/webhook", actions_module=DUMMY_ACTIONS_MODULE_PATH
    )
    remote_action = RemoteAction(DUMMY_ACTION_NAME, endpoint)
    assert isinstance(remote_action.executor, DirectCustomActionExecutor)


def test_remote_action_executor_cached(mock_endpoint: EndpointConfig):
    """
    Ensure the executor for the RemoteAction instance is being
    cached after the action endpoint is updated.

    Assertions:
    - Initially, the executor is `DirectCustomActionExecutor`.
    - After recreating the executor instance, the executor is still
      `DirectCustomActionExecutor` at the same location.
    """
    remote_action = RemoteAction(DUMMY_ACTION_NAME, mock_endpoint)
    assert isinstance(remote_action.executor, DirectCustomActionExecutor)

    initial_executor_id = id(remote_action.executor)
    remote_action.executor = remote_action._create_executor()
    assert id(remote_action.executor) == initial_executor_id


def test_direct_custom_action_executor_valid_initialization(
    direct_custom_action_executor: DirectCustomActionExecutor,
    mock_endpoint: EndpointConfig,
):
    assert direct_custom_action_executor.action_name == DUMMY_ACTION_NAME
    assert direct_custom_action_executor.action_endpoint == mock_endpoint


@pytest.mark.asyncio
async def test_executor_runs_action(
    direct_custom_action_executor: DirectCustomActionExecutor,
    tracker: DialogueStateTracker,
    domain: Domain,
):
    result = await direct_custom_action_executor.run(tracker, domain=domain)
    assert isinstance(result, dict)
    assert "events" in result


async def test_executor_runs_action_invalid_actions_module(
    trained_async: Callable, caplog: LogCaptureFixture, custom_actions_agent: Agent
):
    """
    Ensure that the inappropriately configured actions_module doesn't
    break the execution of the assistant, but raises an exception log.
    """
    # Set MessageProcessor to use the DirectCustomActionExecutor
    # with an invalid actions_module
    processor = custom_actions_agent.processor
    endpoint = EndpointConfig(actions_module=DUMMY_INVALID_ACTIONS_MODULE_PATH)
    processor.action_endpoint = endpoint

    # Trigger the custom action execution and ensure the exception log is raised
    message = UserMessage(text="Activate custom action.")
    await processor.handle_message(message)

    message = (
        "Encountered an exception while running action 'action_force_next_utter'."
        "Bot will continue, but the actions events are lost. "
        "Please check the logs of your action server for more information."
    )
    assert message in caplog.messages
