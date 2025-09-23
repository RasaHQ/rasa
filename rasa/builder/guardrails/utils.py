from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

import structlog

from rasa.builder.guardrails.clients import GuardrailsClient, LakeraAIGuardrails
from rasa.builder.guardrails.models import (
    GuardrailRequest,
    LakeraGuardrailRequest,
)
from rasa.shared.constants import ROLE_USER

if TYPE_CHECKING:
    from rasa.builder.guardrails.models import GuardrailType


structlogger = structlog.get_logger()


def map_lakera_detector_type_to_guardrail_type(
    lakera_detector_type: str,
) -> Optional["GuardrailType"]:
    """Map a Lakera detector type to a Rasa guardrail type."""
    from rasa.builder.guardrails.models import GuardrailType

    # Check for exact matches first
    LAKERA_DETECTOR_TYPES_2_RASA_GUARDRAIL_TYPES_MAPPING = {
        "prompt_attack": GuardrailType.PROMPT_ATTACK,
        "unknown_links": GuardrailType.MALICIOUS_LINKS,
        "custom": GuardrailType.CUSTOM,
    }

    # Check for exact match first
    if lakera_detector_type in LAKERA_DETECTOR_TYPES_2_RASA_GUARDRAIL_TYPES_MAPPING:
        return LAKERA_DETECTOR_TYPES_2_RASA_GUARDRAIL_TYPES_MAPPING[
            lakera_detector_type
        ]

    # Check for subtypes that start with specific prefixes
    # https://docs.lakera.ai/docs/policies/self-hosted-policies#detectors-section
    if lakera_detector_type.startswith("moderated_content"):
        return GuardrailType.CONTENT_VIOLATION
    if lakera_detector_type.startswith("pii"):
        return GuardrailType.DATA_LEAKAGE

    # If no match found, return OTHER
    return GuardrailType.OTHER


def create_guardrail_request(
    client_type: Type[GuardrailsClient],
    user_text: str,
    hello_rasa_user_id: str,
    hello_rasa_project_id: str,
    **kwargs: Any,
) -> GuardrailRequest:
    """Create a guardrail request."""

    def _create_lakera_guardrail_request(
        user_text: str,
        hello_rasa_user_id: str,
        hello_rasa_project_id: str,
        **kwargs: Any,
    ) -> LakeraGuardrailRequest:
        """Create a Lakera guardrail request."""
        return LakeraGuardrailRequest(
            hello_rasa_user_id=hello_rasa_user_id,
            hello_rasa_project_id=hello_rasa_project_id,
            messages=[{"role": ROLE_USER, "content": user_text}],
            **kwargs,
        )

    map_client_to_request: Dict[
        Type[GuardrailsClient], Callable[..., GuardrailRequest]
    ] = {
        LakeraAIGuardrails: _create_lakera_guardrail_request,
    }

    if client_type in map_client_to_request:
        return map_client_to_request[client_type](
            user_text,
            hello_rasa_user_id,
            hello_rasa_project_id,
            **kwargs,
        )
    else:
        message = f"Unsupported guardrail client: {type(client_type)}"
        structlogger.error(
            "guardrails_policy_checker"
            ".create_guardrail_request"
            ".unsupported_guardrail_client",
            message=message,
            guardrail_client=client_type,
        )
        raise ValueError(message)
