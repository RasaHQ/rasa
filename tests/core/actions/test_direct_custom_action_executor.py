import copy

import pytest

from rasa.core.actions.action import RemoteAction
from rasa.core.actions.direct_custom_actions_executor import DirectCustomActionExecutor
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

DUMMY_ACTIONS_MODULE_PATH = "data.dummy_actions_module"
DUMMY_INVALID_ACTIONS_MODULE_PATH = "data.dummy_invalid_actions_module"
DUMMY_ACTION_NAME = "my_action"
DUMMY_DOMAIN_PATH = "data/test_domains/default.yml"

ENDPOINTS_FILE_PATH = "data/test_endpoints/endpoints_actions_module.yml"
mock_endpoint = read_endpoint_config(
    ENDPOINTS_FILE_PATH, endpoint_type="action_endpoint"
)


@pytest.fixture
def direct_custom_action_executor() -> DirectCustomActionExecutor:
    return DirectCustomActionExecutor(
        action_name=DUMMY_ACTION_NAME, action_endpoint=mock_endpoint
    )


@pytest.fixture
def remote_action() -> RemoteAction:
    return RemoteAction(DUMMY_ACTION_NAME, mock_endpoint)


def test_executor_initialized_with_valid_actions_module():
    DirectCustomActionExecutor(action_name="some_action", action_endpoint=mock_endpoint)


def test_executor_initialized_with_invalid_actions_module():
    endpoint = EndpointConfig(actions_module=DUMMY_INVALID_ACTIONS_MODULE_PATH)
    message = (
        f"You've provided the custom actions module '{DUMMY_INVALID_ACTIONS_MODULE_PATH}' "
        f"to run directly by the rasa server, however this module does "
        f"not exist. Please check for typos in your `endpoints.yml` file."
    )
    with pytest.raises(RasaException, match=message):
        DirectCustomActionExecutor(action_name="some_action", action_endpoint=endpoint)


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


def test_remote_action_executor_cached():
    """
    Ensure the executor for a RemoteAction instance remains
    `DirectCustomActionExecutor` after the action endpoint is updated.

    Assertions:
    - Initially, the executor is `DirectCustomActionExecutor`.
    - After setting a new HTTP endpoint, the executor is still
      `DirectCustomActionExecutor` at the same location.
    """
    remote_action = RemoteAction(DUMMY_ACTION_NAME, mock_endpoint)
    assert isinstance(remote_action.executor, DirectCustomActionExecutor)

    initial_executor_id = id(remote_action.executor)
    remote_action.executor = remote_action._create_executor()
    assert id(remote_action.executor) == initial_executor_id


def test_direct_custom_action_executor_valid_initialization(
    direct_custom_action_executor: DirectCustomActionExecutor,
):
    assert direct_custom_action_executor.action_name == DUMMY_ACTION_NAME
    assert direct_custom_action_executor.action_endpoint == mock_endpoint


@pytest.mark.asyncio
async def test_executor_runs_action(
    direct_custom_action_executor: DirectCustomActionExecutor,
):
    tracker = DialogueStateTracker(sender_id="test", slots={})
    from rasa.shared.core.domain import Domain

    domain = Domain.from_file(path=DUMMY_DOMAIN_PATH)
    result = await direct_custom_action_executor.run(tracker, domain=domain)
    assert isinstance(result, dict)
    assert "events" in result
