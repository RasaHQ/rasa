"""Unit tests for guardrails models."""

from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from rasa.builder.guardrails.constants import LAKERA_PROJECT_ID_KEY
from rasa.builder.guardrails.models import (
    GuardrailDetection,
    GuardrailRequestKey,
    GuardrailType,
    LakeraGuardrailResponse,
)


class TestLakeraGuardrailResponse:
    """Test cases for LakeraGuardrailResponse class."""

    @pytest.mark.parametrize(
        "raw_response,"
        "expected_flagged,"
        "expected_detections_count,"
        "expected_detection_types",
        [
            # Test case 1: Response flagged with multiple detections
            (
                {
                    "flagged": True,
                    "metadata": {"test": "metadata"},
                    "breakdown": [
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_prompt_attack_0",
                            "detector_type": "prompt_attack",
                            "detected": True,
                            "message_id": 0,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_moderated_content_0",
                            "detector_type": "moderated_content/weapons",
                            "detected": True,
                            "message_id": 1,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_pii_0",
                            "detector_type": "pii/iban_code",
                            "detected": True,
                            "message_id": 2,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_unknown_links_0",
                            "detector_type": "unknown_links",
                            "detected": True,
                            "message_id": 3,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_custom_0",
                            "detector_type": "custom",
                            "detected": True,
                            "message_id": 4,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_unknown_detector_0",
                            "detector_type": "unknown_detector",
                            "detected": False,  # Should be ignored
                            "message_id": 5,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_prompt_attack_1",
                            "detector_type": "prompt_attack",
                            "detected": False,  # Should be ignored
                            "message_id": None,
                        },
                    ],
                },
                True,
                5,
                [
                    GuardrailType.PROMPT_ATTACK,
                    GuardrailType.CONTENT_VIOLATION,
                    GuardrailType.DATA_LEAKAGE,
                    GuardrailType.MALICIOUS_LINKS,
                    GuardrailType.CUSTOM,
                ],
            ),
            # Test case 2: Response not flagged
            (
                {
                    "flagged": False,
                    "metadata": {"test": "metadata"},
                    "breakdown": [
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_type": "prompt_attack",
                            "detected": False,
                            "message_id": None,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_type": "moderated_content/weapons",
                            "detected": False,
                            "message_id": None,
                        },
                    ],
                },
                False,
                0,
                [],
            ),
            # Test case 3 and 4: Response flagged but no breakdown and empty breakdown
            (
                {
                    "flagged": True,
                    "metadata": {"test": "metadata"},
                },
                True,
                0,
                [],
            ),
            (
                {
                    "flagged": True,
                    "metadata": {"test": "metadata"},
                    "breakdown": [],
                },
                True,
                0,
                [],
            ),
            # Test case 5: Response flagged with only non-detected items
            (
                {
                    "flagged": True,
                    "metadata": {"test": "metadata"},
                    "breakdown": [
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_type": "prompt_attack",
                            "detected": False,
                            "message_id": None,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_type": "moderated_content/weapons",
                            "detected": False,
                            "message_id": None,
                        },
                    ],
                },
                True,
                0,
                [],
            ),
        ],
    )
    def test_from_raw_response(
        self,
        raw_response: Dict[str, Any],
        expected_flagged: bool,
        expected_detections_count: int,
        expected_detection_types: List[GuardrailType],
    ):
        """Test from_raw_response method with various scenarios."""
        # When
        response = LakeraGuardrailResponse.from_raw_response(
            raw_response,
            hello_rasa_user_id="test_user_id",
            hello_rasa_project_id="test_project_id",
        )

        # Then
        assert response.flagged == expected_flagged
        assert response.hello_rasa_user_id == "test_user_id"
        assert response.hello_rasa_project_id == "test_project_id"
        assert response.metadata == raw_response.get("metadata")

        if expected_detections_count > 0:
            assert response.detections is not None
            assert len(response.detections) == expected_detections_count

            # Check detection types
            actual_types = [detection.type for detection in response.detections]
            assert actual_types == expected_detection_types

            # Check that each detection has the correct structure
            for detection in response.detections:
                assert isinstance(detection, GuardrailDetection)
                assert detection.original_type is not None
                assert detection.metadata is not None
                # Verify that detector_type and detected are removed from metadata
                assert "detector_type" not in detection.metadata
                assert "detected" not in detection.metadata

            # Check that detection metadata is preserved
            for detection in response.detections:
                assert detection.metadata is not None
                assert "detector_type" not in detection.metadata
                assert "detected" not in detection.metadata
                assert "project_id" in detection.metadata
                assert "policy_id" in detection.metadata
                assert "message_id" in detection.metadata

        else:
            assert response.detections is None or len(response.detections) == 0

    def test_from_raw_response_all_known_detector_types(self):
        """Test that all known detector types are mapped correctly."""
        raw_response = {
            "flagged": True,
            "breakdown": [
                {
                    "detector_type": "prompt_attack",
                    "detected": True,
                    "message_id": "msg_1",
                },
                {
                    "detector_type": "moderated_content",
                    "detected": True,
                    "message_id": "msg_2",
                },
                {
                    "detector_type": "pii",
                    "detected": True,
                    "message_id": "msg_3",
                },
                {
                    "detector_type": "unknown_links",
                    "detected": True,
                    "message_id": "msg_4",
                },
                {
                    "detector_type": "custom",
                    "detected": True,
                    "message_id": "msg_5",
                },
            ],
        }

        response = LakeraGuardrailResponse.from_raw_response(
            raw_response,
            hello_rasa_user_id="test_user",
            hello_rasa_project_id="test_project",
        )

        expected_types = [
            GuardrailType.PROMPT_ATTACK,
            GuardrailType.CONTENT_VIOLATION,
            GuardrailType.DATA_LEAKAGE,
            GuardrailType.MALICIOUS_LINKS,
            GuardrailType.CUSTOM,
        ]

        actual_types = [detection.type for detection in response.detections]
        assert actual_types == expected_types


class TestGuardrailRequestKey:
    def test_guardrail_request_key_value_equality_and_hash(self) -> None:
        k1 = GuardrailRequestKey(
            user_text="hi",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )
        k2 = GuardrailRequestKey(
            user_text="hi",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )

        # Different instances, equal by value, same hash
        assert k1 is not k2
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_guardrail_request_key_as_dict_key(self) -> None:
        k1 = GuardrailRequestKey(
            user_text="hello",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )
        k2 = GuardrailRequestKey(
            user_text="hello",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )

        mapping = {k1: "value"}
        assert mapping[k2] == "value"
        assert len(mapping) == 1

    def test_guardrail_request_key_inequality_by_field(self) -> None:
        base = GuardrailRequestKey(
            user_text="hello",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )
        diff_text = GuardrailRequestKey(
            user_text="hello!",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )
        diff_user = GuardrailRequestKey(
            user_text="hello",
            hello_rasa_user_id="u2",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )
        diff_project = GuardrailRequestKey(
            user_text="hello",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p2",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )
        diff_policy = GuardrailRequestKey(
            user_text="hello",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk2"},
        )

        assert base != diff_text
        assert base != diff_user
        assert base != diff_project
        assert base != diff_policy

        # As dict keys, they should create distinct entries
        mapping = {base: 1, diff_text: 2, diff_user: 3, diff_project: 4, diff_policy: 5}
        assert len(mapping) == 5

    def test_guardrail_request_key_is_frozen_immutable(self) -> None:
        key = GuardrailRequestKey(
            user_text="immutable",
            hello_rasa_user_id="u1",
            hello_rasa_project_id="p1",
            metadata={LAKERA_PROJECT_ID_KEY: "lk1"},
        )
        with pytest.raises(ValidationError):
            key.user_text = "mutated"
