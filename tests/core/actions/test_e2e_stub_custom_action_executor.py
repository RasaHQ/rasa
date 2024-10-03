from typing import Dict, Any

import pytest

from rasa.core.actions.action import RemoteAction
from rasa.core.actions.e2e_stub_custom_action_executor import (
    E2EStubCustomActionExecutor,
)
from rasa.e2e_test.stub_custom_action import StubCustomAction
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def invalid_action_name() -> str:
    return "invalid_action_name"


@pytest.fixture
def remote_action(
    endpoint_stub_config: EndpointConfig, action_name_test_file: str
) -> RemoteAction:
    return RemoteAction(action_name_test_file, endpoint_stub_config)


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker(sender_id="test_user", slots={})


@pytest.fixture
def domain() -> Domain:
    return Domain.empty()


def test_e2e_stub_custom_action_executor_init(
    endpoint_stub_config: EndpointConfig,
    action_test_file_stub: StubCustomAction,
    action_name_test_file: str,
):
    executor = E2EStubCustomActionExecutor(action_name_test_file, endpoint_stub_config)
    assert executor.action_name == action_name_test_file
    assert executor.action_endpoint == endpoint_stub_config
    assert executor.stub_custom_action == action_test_file_stub


def test_init_executor_with_invalid_action_name(
    endpoint_stub_config: EndpointConfig, invalid_action_name: str
):
    with pytest.raises(RasaException) as excinfo:
        E2EStubCustomActionExecutor(invalid_action_name, endpoint_stub_config)
    assert "You are using custom action stubs" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_with_valid_stub_action(
    endpoint_stub_config: EndpointConfig,
    tracker: DialogueStateTracker,
    domain: Domain,
    stub_data: Dict[str, Any],
    action_name_test_file: str,
):
    executor = E2EStubCustomActionExecutor(action_name_test_file, endpoint_stub_config)
    result = await executor.run(tracker, domain)
    assert result == stub_data


def test_remote_action_initializes_e2e_stub_custom_action_executor(
    remote_action: RemoteAction,
):
    assert isinstance(remote_action.executor, E2EStubCustomActionExecutor)
