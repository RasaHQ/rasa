from unittest.mock import AsyncMock
from unittest.mock import patch, Mock

import pytest

from rasa.core.actions.action import RemoteAction
from rasa.core.actions.direct_custom_actions_executor import DirectCustomActionExecutor
from rasa.shared.constants import DEFAULT_ACTIONS_PATH
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def direct_custom_action_executor() -> DirectCustomActionExecutor:
    endpoint = EndpointConfig(actions_module=DEFAULT_ACTIONS_PATH)
    with patch(
        "rasa.core.actions.direct_custom_actions_executor.find_spec",
        return_value=Mock(),
    ), patch("importlib.import_module", return_value=Mock()):
        return DirectCustomActionExecutor(
            action_name="my_action", action_endpoint=endpoint
        )


@pytest.fixture
def remote_action() -> RemoteAction:
    endpoint = EndpointConfig(actions_module=DEFAULT_ACTIONS_PATH)
    with patch(
        "rasa.core.actions.direct_custom_actions_executor.find_spec",
        return_value=Mock(),
    ), patch("importlib.import_module", return_value=Mock()):
        return RemoteAction("my_action", endpoint)


@pytest.mark.parametrize(
    "actions_module, should_raise_exception",
    [
        ("valid_actions_module", False),
        ("invalid_actions_module", True),
    ],
)
def test_executor_with_valid_and_invalid_module(actions_module, should_raise_exception):
    endpoint = EndpointConfig(url=None, actions_module=actions_module)
    return_value = None if should_raise_exception else Mock()

    with patch(
        "rasa.core.actions.direct_custom_actions_executor.find_spec",
        return_value=return_value,
    ), patch("importlib.import_module", return_value=return_value):
        if should_raise_exception:
            with pytest.raises(
                RasaException,
                match="Actions module 'invalid_actions_module' does not exist.",
            ):
                DirectCustomActionExecutor(
                    action_name="some_action", action_endpoint=endpoint
                )
        else:
            DirectCustomActionExecutor(
                action_name="some_action", action_endpoint=endpoint
            )


def test_warning_raised_for_url_and_actions_module_defined():
    endpoint = EndpointConfig(
        url="http://localhost:5055/webhook", actions_module=DEFAULT_ACTIONS_PATH
    )
    with pytest.warns(
        UserWarning, match="Both 'actions_module' and 'url' are defined."
    ):
        try:
            RemoteAction("my_action", endpoint)
        except RasaException:
            pass


def test_remote_action_initializes_direct_custom_action_executor(
    remote_action: RemoteAction,
):
    assert isinstance(remote_action.executor, DirectCustomActionExecutor)


def test_remote_action_uses_action_endpoint_with_url_and_actions_module_defined():
    endpoint = EndpointConfig(
        url="http://localhost:5055/webhook", actions_module=DEFAULT_ACTIONS_PATH
    )
    with patch(
        "rasa.core.actions.direct_custom_actions_executor.find_spec",
        return_value=Mock(),
    ), patch("importlib.import_module", return_value=Mock()):
        remote_action = RemoteAction("my_action", endpoint)
        assert isinstance(remote_action.executor, DirectCustomActionExecutor)


def test_remote_action_executor_cached():
    with patch(
        "rasa.core.actions.direct_custom_actions_executor.find_spec",
        return_value=Mock(),
    ), patch("importlib.import_module", return_value=Mock()):
        actions_module_endpoint = EndpointConfig(actions_module=DEFAULT_ACTIONS_PATH)
        remote_action = RemoteAction("my_action", actions_module_endpoint)

        remote_action.action_endpoint = actions_module_endpoint
        assert isinstance(remote_action.executor, DirectCustomActionExecutor)

        url_endpoint = EndpointConfig(url="http://localhost:5055/webhook")
        remote_action.action_endpoint = url_endpoint
        remote_action.executor = remote_action._create_executor()
        assert isinstance(remote_action.executor, DirectCustomActionExecutor)


def test_direct_custom_action_executor_valid_initialization(
    direct_custom_action_executor: DirectCustomActionExecutor,
):
    endpoint = EndpointConfig(actions_module=DEFAULT_ACTIONS_PATH)
    assert direct_custom_action_executor.action_name == "my_action"
    assert direct_custom_action_executor.action_endpoint == endpoint


@pytest.mark.asyncio
async def test_executor_runs_action(
    direct_custom_action_executor: DirectCustomActionExecutor,
):
    tracker = DialogueStateTracker(sender_id="test", slots={})
    with patch.object(
        direct_custom_action_executor.action_executor, "run", new_callable=AsyncMock
    ) as mock_run:
        mock_run.return_value = {"events": [], "responses": [{"text": "Hello"}]}
        result = await direct_custom_action_executor.run(tracker)
        assert isinstance(result, dict)
        assert "events" in result
