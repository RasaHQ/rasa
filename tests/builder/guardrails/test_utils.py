"""Unit tests for guardrails utils."""

import pytest

from rasa.builder.guardrails.models import GuardrailType
from rasa.builder.guardrails.utils import map_lakera_detector_type_to_guardrail_type


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
