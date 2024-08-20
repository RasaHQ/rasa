from dataclasses import dataclass
from typing import Any, Dict, List, Text

from rasa.e2e_test.constants import (
    KEY_STUB_CUSTOM_ACTIONS,
    STUB_CUSTOM_ACTION_NAME_SEPARATOR,
    TEST_FILE_NAME,
    TEST_CASE_NAME,
)
from rasa.utils.endpoints import EndpointConfig


@dataclass
class StubCustomAction:
    """Class for storing the stub response of the custom action."""

    action_name: str
    events: List[Dict[Text, Any]]
    responses: List[Dict[Text, Any]]

    @staticmethod
    def from_dict(action_name: str, stub_data: Dict[Text, Any]) -> "StubCustomAction":
        """Creates a stub custom action from a dictionary.

        Example:
            >>> StubCustomAction.from_dict(
            >>>        {"action_name": {"events": [], "responses": []}}
            >>>    )
            StubCustomAction(name="action_name", events=[], responses=[])

        Args:
            action_name (str): Name of the custom action.
            stub_data (Dict[Text, Any]): Stub custom action response.
        """
        return StubCustomAction(
            action_name=action_name,
            events=[event for event in stub_data.get("events", [])],
            responses=[response for response in stub_data.get("responses", [])],
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the metadata as a dictionary."""
        return {"events": self.events, "responses": self.responses}


def get_stub_custom_action_key(prefix: str, action_name: str) -> str:
    """Returns the key used to store the StubCustomAction object"""
    if STUB_CUSTOM_ACTION_NAME_SEPARATOR in action_name:
        return action_name
    return f"{prefix}{STUB_CUSTOM_ACTION_NAME_SEPARATOR}{action_name}"


def get_stub_custom_action(
    action_endpoint: EndpointConfig, action_name: str
) -> "StubCustomAction":
    """Returns the StubCustomAction object"""
    # Fetch the name of the test file and of the test case
    test_file_name = action_endpoint.kwargs.get(TEST_FILE_NAME)
    test_case_name = action_endpoint.kwargs.get(TEST_CASE_NAME)

    # Generate keys for custom action stub
    stub_test_file_key = get_stub_custom_action_key(test_file_name, action_name)
    stub_test_case_key = get_stub_custom_action_key(test_case_name, action_name)

    # Fetch the custom action stub, prioritizing the test case naming
    stub_custom_actions = action_endpoint.kwargs.get(KEY_STUB_CUSTOM_ACTIONS, {})
    return stub_custom_actions.get(
        stub_test_case_key,
        stub_custom_actions.get(
            stub_test_file_key, StubCustomAction.from_dict(action_name, {})
        ),
    )
