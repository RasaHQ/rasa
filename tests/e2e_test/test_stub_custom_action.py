import pytest

from rasa.e2e_test.constants import (
    STUB_CUSTOM_ACTION_NAME_SEPARATOR,
    TEST_FILE_NAME,
    TEST_CASE_NAME,
    KEY_STUB_CUSTOM_ACTIONS,
)
from rasa.e2e_test.stub_custom_action import (
    StubCustomAction,
    get_stub_custom_action_key,
    get_stub_custom_action,
)
from rasa.utils.endpoints import EndpointConfig

DUMMY_STUB_DATA = {
    "events": [{"event": "sample_event"}],
    "responses": [{"response": "sample_response"}],
}
DUMMY_TEST_FILE_NAME = "test_file_name.yml"
DUMMY_TEST_CASE_NAME = "test_case_name"

DUMMY_ACTION_NAME_TEST_FILE = "action_test_file"
DUMMY_ACTION_NAME_TEST_CASE = "action_test_case"

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


@pytest.fixture
def endpoint_config():
    return EndpointConfig(
        url="http://localhost:5055/webhook",
        **{
            TEST_FILE_NAME: DUMMY_TEST_FILE_NAME,
            TEST_CASE_NAME: DUMMY_TEST_CASE_NAME,
            KEY_STUB_CUSTOM_ACTIONS: {
                DUMMY_ACTION_NAME_TEST_FILE_WITH_SEPARATOR: StubCustomAction.from_dict(
                    DUMMY_ACTION_NAME_TEST_FILE, DUMMY_STUB_DATA
                ),
                DUMMY_ACTION_NAME_TEST_CASE_WITH_SEPARATOR: StubCustomAction.from_dict(
                    DUMMY_ACTION_NAME_TEST_CASE, DUMMY_STUB_DATA
                ),
            },
        },
    )


def test_stub_custom_action_from_dict():
    stub_action = StubCustomAction.from_dict(
        DUMMY_ACTION_NAME_TEST_FILE, DUMMY_STUB_DATA
    )
    assert stub_action.action_name == DUMMY_ACTION_NAME_TEST_FILE
    assert stub_action.events == DUMMY_STUB_DATA["events"]
    assert stub_action.responses == DUMMY_STUB_DATA["responses"]


def test_stub_custom_action_as_dict():
    stub_action = StubCustomAction.from_dict(
        DUMMY_ACTION_NAME_TEST_FILE, DUMMY_STUB_DATA
    )
    assert stub_action.as_dict() == DUMMY_STUB_DATA


def test_get_stub_custom_action_key():
    assert (
        get_stub_custom_action_key(DUMMY_TEST_FILE_NAME, DUMMY_ACTION_NAME_TEST_FILE)
        == DUMMY_ACTION_NAME_TEST_FILE_WITH_SEPARATOR
    )


def test_get_stub_custom_action_key_with_separator_in_name():
    assert (
        get_stub_custom_action_key(
            DUMMY_TEST_FILE_NAME, DUMMY_ACTION_NAME_TEST_CASE_WITH_SEPARATOR
        )
        == DUMMY_ACTION_NAME_TEST_CASE_WITH_SEPARATOR
    )


def test_get_stub_test_file_custom_action(endpoint_config):
    stub_action = get_stub_custom_action(endpoint_config, DUMMY_ACTION_NAME_TEST_FILE)
    assert stub_action.action_name == DUMMY_ACTION_NAME_TEST_FILE
    assert stub_action.events == DUMMY_STUB_DATA["events"]
    assert stub_action.responses == DUMMY_STUB_DATA["responses"]


def test_get_stub_test_case_custom_action(endpoint_config):
    stub_action = get_stub_custom_action(endpoint_config, DUMMY_ACTION_NAME_TEST_CASE)
    assert stub_action.action_name == DUMMY_ACTION_NAME_TEST_CASE
    assert stub_action.events == DUMMY_STUB_DATA["events"]
    assert stub_action.responses == DUMMY_STUB_DATA["responses"]


def test_get_stub_custom_action_fallback(endpoint_config):
    new_action_name = "new_action"
    stub_action = get_stub_custom_action(endpoint_config, new_action_name)
    assert stub_action is None
