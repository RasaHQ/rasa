from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text

from rasa.e2e_test.constants import (
    KEY_STUB_CUSTOM_ACTIONS,
    STUB_CUSTOM_ACTION_NAME_SEPARATOR,
    TEST_CASE_NAME,
    TEST_FILE_NAME,
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
    return f"{prefix}{STUB_CUSTOM_ACTION_NAME_SEPARATOR}{action_name}"


def get_stub_custom_action(
    action_endpoint: EndpointConfig, action_name: str
) -> Optional["StubCustomAction"]:
    """Returns the StubCustomAction object"""
    test_case_name = action_endpoint.kwargs.get(TEST_CASE_NAME)
    stub_custom_action_test_key = get_stub_custom_action_key(
        test_case_name, action_name
    )

    test_file_name = action_endpoint.kwargs.get(TEST_FILE_NAME)
    stub_custom_action_file_key = get_stub_custom_action_key(
        test_file_name, action_name
    )
    stub_custom_actions = action_endpoint.kwargs.get(KEY_STUB_CUSTOM_ACTIONS, {})

    stub = stub_custom_actions.get(
        stub_custom_action_test_key
    ) or stub_custom_actions.get(stub_custom_action_file_key)

    return stub
