from typing import Dict, Any

import pytest

from rasa.e2e_test.constants import (
    KEY_STUB_CUSTOM_ACTIONS,
    TEST_CASE_NAME,
    TEST_FILE_NAME,
)
from rasa.e2e_test.stub_custom_action import (
    StubCustomAction,
    get_stub_custom_action_key,
    get_stub_custom_action,
)
from rasa.e2e_test.utils.validation import read_e2e_test_schema
from rasa.shared.utils.yaml import (
    YamlValidationException,
    parse_raw_yaml,
    validate_yaml_data_using_schema_with_assertions,
)
from rasa.utils.endpoints import EndpointConfig


def test_stub_custom_action_from_dict(
    stub_data: Dict[str, Any], action_name_test_file: str
):
    stub_action = StubCustomAction.from_dict(action_name_test_file, stub_data)
    assert stub_action.action_name == action_name_test_file
    assert stub_action.events == stub_data["events"]
    assert stub_action.responses == stub_data["responses"]


def test_stub_custom_action_as_dict(
    stub_data: Dict[str, Any], action_name_test_file: str
):
    stub_action = StubCustomAction.from_dict(action_name_test_file, stub_data)
    assert stub_action.as_dict() == stub_data


def test_get_stub_custom_action_key(
    action_name_test_file_with_separator: str,
    test_file_name: str,
    action_name_test_file: str,
):
    assert (
        get_stub_custom_action_key(test_file_name, action_name_test_file)
        == action_name_test_file_with_separator
    )


def test_get_stub_test_file_custom_action(
    endpoint_stub_config: EndpointConfig,
    stub_data: Dict[str, Any],
    action_name_test_file: str,
):
    stub_action = get_stub_custom_action(endpoint_stub_config, action_name_test_file)
    assert stub_action.action_name == action_name_test_file
    assert stub_action.events == stub_data["events"]
    assert stub_action.responses == stub_data["responses"]


def test_get_stub_test_case_custom_action(
    endpoint_stub_config: EndpointConfig,
    stub_data: Dict[str, Any],
    action_name_test_case: str,
):
    stub_action = get_stub_custom_action(endpoint_stub_config, action_name_test_case)
    assert stub_action.action_name == action_name_test_case
    assert stub_action.events == stub_data["events"]
    assert stub_action.responses == stub_data["responses"]


def test_get_stub_test_case_custom_action_with_separator(
    stub_data: Dict[str, Any],
    action_name_test_case_with_separator: str,
    action_name_test_case: str,
    test_file_name: str,
    test_case_name: str,
):
    endpoint_config = EndpointConfig(
        url="http://localhost:5055/webhook",
        **{
            TEST_FILE_NAME: test_file_name,
            TEST_CASE_NAME: test_case_name,
            KEY_STUB_CUSTOM_ACTIONS: {
                action_name_test_case_with_separator: StubCustomAction.from_dict(
                    action_name_test_case_with_separator, stub_data
                ),
            },
        },
    )
    stub_action = get_stub_custom_action(endpoint_config, action_name_test_case)
    assert stub_action.action_name == action_name_test_case_with_separator
    assert stub_action.events == stub_data["events"]
    assert stub_action.responses == stub_data["responses"]


def test_get_stub_custom_action_fallback(endpoint_stub_config):
    new_action_name = "new_action"
    stub_action = get_stub_custom_action(endpoint_stub_config, new_action_name)
    assert stub_action is None


@pytest.mark.parametrize(
    "stub_custom_action",
    [
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: user
                  text: "Hello"
                  input_channel: "slack"
                  parse_data:
                    entities: []
                    intent:
                      name: "greet"
                      confidence: 0.9
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: bot
                  text: "Welcome back!"
                  data:
                    buttons:
                    - title: "Button 1"
                      payload: "/button1"
                  timestamp: 1234567890
                  metadata:
                    metadata_key: "metadata_value"
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: action
                  name: "action_listen"
                  policy: "FlowPolicy"
                  confidence: 0.9
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: slot
                  name: "city"
                  value: "London"
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: restart
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: session_started
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: rewind
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: reset_slots
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: reminder
                  date_time: "2022-01-01T00:00:00"
                  name: "reminder_test"
                  entities: []
                  intent: "set_reminder"
                  kill_on_user_message: true
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: cancel_reminder
                  name: "reminder_test"
                  intent: "cancel_reminder"
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: undo
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: export
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: followup
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: pause
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: resume
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: agent
                  text: "Hello"
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: active_loop
                  name: restaurant_form
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: loop_interrupted
                  is_interrupted: true
                responses: []
        """,
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: action_execution_rejected
                  name: action_transfer_money
                responses: []
        """,
    ],
)
def test_stub_custom_action_event_respects_schema(stub_custom_action: str) -> None:
    e2e_test_schema = read_e2e_test_schema()
    yaml_content = parse_raw_yaml(stub_custom_action)

    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=yaml_content, schema_content=e2e_test_schema
        )
    except Exception as exc:
        pytest.fail(f"Failed to validate the schema: {exc!s}")


def test_stub_custom_action_responses_respect_schema() -> None:
    e2e_test_schema = read_e2e_test_schema()
    yaml_content = parse_raw_yaml(
        """
        stub_custom_actions:
            mock_action:
                events: []
                responses:
                - text: "Hello"
                  buttons:
                  - title: "Button 1"
                    payload: "/button1"
                  image: "http://image.jpg"
                  custom: {}
                  elements: []
                  attachment: "file"
                  response: "utter_greet"
        """
    )

    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=yaml_content, schema_content=e2e_test_schema
        )
    except Exception as exc:
        pytest.fail(f"Failed to validate the schema: {exc!s}")


def test_stub_custom_action_invalid_responses() -> None:
    e2e_test_schema = read_e2e_test_schema()
    yaml_content = parse_raw_yaml(
        """
        stub_custom_actions:
            mock_action:
                events: []
                responses:
                - text: "Hello"
                  invalid_key: "invalid_value"
        """
    )

    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=yaml_content, schema_content=e2e_test_schema
        )
        assert exc.value.message == (
            "Additional properties are not allowed ('invalid_key' was unexpected)"
        )


def test_stub_custom_action_invalid_event() -> None:
    e2e_test_schema = read_e2e_test_schema()
    yaml_content = parse_raw_yaml(
        """
        stub_custom_actions:
            mock_action:
                events:
                - event: "dialogue_stack_updated"
                  invalid_key: "invalid_value"
                responses: []
        """
    )

    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=yaml_content, schema_content=e2e_test_schema
        )

    assert "Enum 'dialogue_stack_updated' does not exist." in str(exc.value)
    assert "Key 'invalid_key' was not defined." in str(exc.value)
