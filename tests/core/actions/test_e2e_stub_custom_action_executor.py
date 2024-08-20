import pytest

from rasa.core.actions.action import RemoteAction
from rasa.core.actions.e2e_stub_custom_action_executor import (
    E2EStubCustomActionExecutor,
)
from rasa.e2e_test.constants import (
    STUB_CUSTOM_ACTION_NAME_SEPARATOR,
    TEST_FILE_NAME,
    TEST_CASE_NAME,
    KEY_STUB_CUSTOM_ACTIONS,
)
from rasa.e2e_test.stub_custom_action import StubCustomAction
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

DUMMY_STUB_DATA = {
    "events": [{"event": "sample_event"}],
    "responses": [{"response": "sample_response"}],
}
DUMMY_TEST_FILE_NAME = "test_file_name.yml"
DUMMY_TEST_CASE_NAME = "test_case_name"

DUMMY_ACTION_NAME_TEST_FILE = "action_test_file"
DUMMY_ACTION_NAME_TEST_CASE = "action_test_case"
DUMMY_INVALID_ACTION_NAME = "invalid_action_name"

DUMMY_ACTION_NAME_TEST_FILE_WITH_SEPARATOR = (
    f"{DUMMY_TEST_FILE_NAME}"
    f"{STUB_CUSTOM_ACTION_NAME_SEPARATOR}"
    f"{DUMMY_ACTION_NAME_TEST_FILE}"
)
DUMMY_ACTION_NAME_TEST_CASE_WITH_SEPARATOR = (
    f"{DUMMY_TEST_CASE_NAME}"
    f"{STUB_CUSTOM_ACTION_NAME_SEPARATOR}"
    f"{DUMMY_ACTION_NAME_TEST_CASE}"
)

DUMMY_ACTION_TEST_FILE_STUB = StubCustomAction.from_dict(
    DUMMY_ACTION_NAME_TEST_FILE, DUMMY_STUB_DATA
)
DUMMY_ACTION_TEST_CASE_STUB = StubCustomAction.from_dict(
    DUMMY_ACTION_NAME_TEST_CASE, DUMMY_STUB_DATA
)


@pytest.fixture
def endpoint_config() -> EndpointConfig:
    return EndpointConfig(
        url="http://localhost:5055/webhook",
        **{
            TEST_FILE_NAME: DUMMY_TEST_FILE_NAME,
            TEST_CASE_NAME: DUMMY_TEST_CASE_NAME,
            KEY_STUB_CUSTOM_ACTIONS: {
                DUMMY_ACTION_NAME_TEST_FILE_WITH_SEPARATOR: DUMMY_ACTION_TEST_FILE_STUB,
                DUMMY_ACTION_NAME_TEST_CASE_WITH_SEPARATOR: DUMMY_ACTION_TEST_CASE_STUB,
            },
        },
    )


@pytest.fixture
def remote_action(endpoint_config: EndpointConfig) -> RemoteAction:
    return RemoteAction(DUMMY_ACTION_NAME_TEST_FILE, endpoint_config)


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker(sender_id="test_user", slots={})


@pytest.fixture
def domain() -> Domain:
    return Domain.empty()


def test_e2e_stub_custom_action_executor_init(endpoint_config: EndpointConfig):
    executor = E2EStubCustomActionExecutor(DUMMY_ACTION_NAME_TEST_CASE, endpoint_config)
    assert executor.action_name == DUMMY_ACTION_NAME_TEST_CASE
    assert executor.action_endpoint == endpoint_config
    assert executor.stub_custom_action == DUMMY_ACTION_TEST_CASE_STUB


def test_init_executor_with_invalid_action_name(endpoint_config: EndpointConfig):
    with pytest.raises(RasaException) as excinfo:
        E2EStubCustomActionExecutor(DUMMY_INVALID_ACTION_NAME, endpoint_config)
    assert f"Action `{DUMMY_INVALID_ACTION_NAME}` has not been stubbed." in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_run_with_valid_stub_action(
    endpoint_config: EndpointConfig, tracker: DialogueStateTracker, domain: Domain
):
    executor = E2EStubCustomActionExecutor(DUMMY_ACTION_NAME_TEST_FILE, endpoint_config)
    result = await executor.run(tracker, domain)
    assert result == DUMMY_STUB_DATA


def test_remote_action_initializes_e2e_stub_custom_action_executor(
    remote_action: RemoteAction,
):
    assert isinstance(remote_action.executor, E2EStubCustomActionExecutor)
