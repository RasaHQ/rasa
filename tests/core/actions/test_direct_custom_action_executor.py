from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest import CaptureFixture, MonkeyPatch

from rasa.core.actions.action import RemoteAction, RemoteActionJSONValidator
from rasa.core.actions.direct_custom_actions_executor import DirectCustomActionExecutor
from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from tests.conftest import TrainedAsync

DUMMY_ACTIONS_MODULE_PATH = "data.dummy_actions_module"
DUMMY_INVALID_ACTIONS_MODULE_PATH = "data.dummy_invalid_actions_module"
DUMMY_ACTION_NAME = "my_action"
DUMMY_DOMAIN_PATH = "data/test_domains/default.yml"

ENDPOINTS_FILE_PATH = "data/test_endpoints/endpoints_actions_module.yml"


@pytest.fixture(autouse=True)
def setup():
    DirectCustomActionExecutor._actions_module_registered = False
    DirectCustomActionExecutor._create_action_executor.cache_clear()


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

    message = (
        f"You've provided the custom actions module "
        f"'{DUMMY_INVALID_ACTIONS_MODULE_PATH}' to run directly by the rasa server, "
        f"however this module does not exist. "
        f"Please check for typos in your `endpoints.yml` file."
    )
    with pytest.raises(RasaException, match=message):
        executor = DirectCustomActionExecutor(
            action_name="some_action", action_endpoint=endpoint
        )
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


@pytest.mark.asyncio
async def test_executor_runs_action_without_response_validation(
    direct_custom_action_executor: DirectCustomActionExecutor,
    tracker: DialogueStateTracker,
    domain: Domain,
    monkeypatch: MonkeyPatch,
):
    mock_validate = MagicMock()
    monkeypatch.setattr(RemoteActionJSONValidator, "validate", mock_validate)
    await direct_custom_action_executor.run(tracker, domain=domain)
    mock_validate.assert_not_called()


async def test_executor_runs_action_invalid_actions_module(
    trained_async: TrainedAsync, capsys: CaptureFixture, custom_actions_agent: Agent
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
    error_message = (
        "You've provided the custom actions module "
        f"'{DUMMY_INVALID_ACTIONS_MODULE_PATH}' to run directly by the rasa server, "
        "however this module does not exist. "
        "Please check for typos in your `endpoints.yml` file."
    )
    with pytest.raises(RasaException, match=error_message):
        await processor.handle_message(message)


def test_action_executor_is_being_cached(mock_endpoint: EndpointConfig):
    executor_1 = DirectCustomActionExecutor(
        action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
    )
    executor_2 = DirectCustomActionExecutor(
        action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
    )
    assert executor_1.action_executor == executor_2.action_executor


# FIXME: This test passes locally but is flaky in CI.
@pytest.mark.skip_on_ci
@pytest.mark.asyncio
async def test_custom_actions_hot_reloading():
    def create_action_code(value: str) -> str:
        return f"""from typing import Any, Dict
from rasa_sdk.interfaces import Action
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher

class CustomAction(Action):
    def name(self) -> str:
        return "custom_action"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> Any:
        return [{{"event": "slot", "name": "test_slot", "value": "{value}"}}]
"""

    # Create a custom action file with initial value
    action_module = Path(DUMMY_ACTIONS_MODULE_PATH.replace(".", "/"))
    action_file = action_module / "custom_action.py"
    initial_value = "initial_value"
    action_file.write_text(create_action_code(initial_value))

    # Create an endpoint and executor with the initial custom action
    endpoint = EndpointConfig(actions_module=DUMMY_ACTIONS_MODULE_PATH)
    executor = DirectCustomActionExecutor("custom_action", endpoint)

    # Run the custom action with the initial value
    tracker = DialogueStateTracker("default", [])
    domain = Domain.empty()
    result_initial = await executor.run(tracker, domain)
    assert result_initial["events"][0]["value"] == initial_value

    # Modify the custom action file with a new value
    modified_value = "modified_value"
    action_file.write_text(create_action_code(modified_value))

    # Run the custom action with the modified value
    executor = DirectCustomActionExecutor("custom_action", endpoint)
    result_modified = await executor.run(tracker, domain)
    assert result_modified["events"][0]["value"] == modified_value
    action_file.unlink()
