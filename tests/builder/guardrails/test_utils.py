"""Unit tests for guardrails utils."""

from typing import List

import pytest

from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_USER
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    ResponseCategory,
    TextContent,
    UserChatMessage,
)
from rasa.builder.guardrails.clients import LakeraAIGuardrails
from rasa.builder.guardrails.models import GuardrailType
from rasa.builder.guardrails.policy_checker import GuardrailsPolicyChecker
from rasa.builder.guardrails.utils import (
    map_lakera_detector_type_to_guardrail_type,
)


def create_copilot_chat_message(role: str, text: str):
    if role == ROLE_USER:
        return UserChatMessage(role=role, content=[TextContent(type="text", text=text)])
    elif role == ROLE_COPILOT:
        return CopilotChatMessage(
            role=role, content=[TextContent(type="text", text=text)]
        )
    else:
        raise ValueError(f"Unknown role: {role}")


class TestMapLakeraDetectorTypeToGuardrailType:
    """Test cases for map_lakera_detector_type_to_guardrail_type function."""

    @pytest.mark.parametrize(
        "detector_type,expected_guardrail_type",
        [
            # Exact matches
            ("prompt_attack", GuardrailType.PROMPT_ATTACK),
            ("unknown_links", GuardrailType.MALICIOUS_LINKS),
            ("custom", GuardrailType.CUSTOM),
            # Subtype matches
            ("moderated_content/weapons", GuardrailType.CONTENT_VIOLATION),
            ("moderated_content/hate", GuardrailType.CONTENT_VIOLATION),
            ("pii/iban_code", GuardrailType.DATA_LEAKAGE),
            ("pii/phone_number", GuardrailType.DATA_LEAKAGE),
            # Unknown types
            ("unknown_detector", GuardrailType.OTHER),
            ("some_random_type", GuardrailType.OTHER),
        ],
    )
    def test_map_lakera_detector_type_to_guardrail_type(
        self, detector_type: str, expected_guardrail_type: GuardrailType
    ):
        """Test that Lakera detector types are correctly mapped to Rasa types."""
        result = map_lakera_detector_type_to_guardrail_type(detector_type)
        assert result == expected_guardrail_type

    def test_annotate_flagged_user_messages_marks_user_indices_and_ignores_out_of_range(
        self,
    ) -> None:
        # Given
        history: List[CopilotChatMessage] = [
            create_copilot_chat_message("user", "hello"),
            create_copilot_chat_message("copilot", "welcome"),
            create_copilot_chat_message("user", "steal money"),
            create_copilot_chat_message("copilot", "refuse"),
            create_copilot_chat_message("user", "fine"),
        ]

        flagged = {
            2,  # Index 2 is the proper user message to be annotated
            1,  # Index 1 is a copilot message, these should not be annotated
            10,  # Index 10 is out of range and should be ignored
        }

        # Create a policy checker instance to test the method
        client = LakeraAIGuardrails(api_key="test_key")
        policy_checker = GuardrailsPolicyChecker(client)

        # When
        policy_checker._annotate_flagged_user_messages(history, flagged)

        # Then
        assert history[0].response_category is None
        assert history[1].response_category is None
        assert (
            history[2].response_category == ResponseCategory.GUARDRAILS_POLICY_VIOLATION
        )
        assert history[3].response_category is None
        assert history[4].response_category is None

    def test_annotate_flagged_user_messages_idempotent_and_noop_on_empty(self) -> None:
        # Given
        history: List[CopilotChatMessage] = [
            create_copilot_chat_message("user", "hello"),
            create_copilot_chat_message("user", "steal money"),
        ]
        client = LakeraAIGuardrails(api_key="test_key")
        policy_checker = GuardrailsPolicyChecker(client)

        # When
        policy_checker._annotate_flagged_user_messages(history, {1})

        # Then
        assert history[0].response_category is None
        assert (
            history[1].response_category == ResponseCategory.GUARDRAILS_POLICY_VIOLATION
        )

        # When: No-op on empty flagged set should not change anything
        policy_checker._annotate_flagged_user_messages(history, set())

        # Then
        assert history[0].response_category is None
        assert (
            history[1].response_category == ResponseCategory.GUARDRAILS_POLICY_VIOLATION
        )
