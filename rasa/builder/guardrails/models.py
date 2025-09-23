"""Models for guardrails system."""

import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GuardrailType(Enum):
    """Types of guardrails that can be applied with Lakera AI."""

    PROMPT_ATTACK = "prompt_attack"
    CONTENT_VIOLATION = "content_violation"
    DATA_LEAKAGE = "data_leakage"
    MALICIOUS_LINKS = "malicious_content"
    CUSTOM = "custom"
    OTHER = "other"


class GuardrailRequestKey(BaseModel):
    user_text: str
    hello_rasa_user_id: str = ""
    hello_rasa_project_id: str = ""
    # Generic metadata field for provider-specific configurations
    metadata: Dict[str, str] = Field(default_factory=dict)

    # hashable by value
    model_config = ConfigDict(frozen=True)

    def __hash__(self) -> int:
        """Custom hash implementation that handles the metadata dictionary."""
        # Convert metadata dict to a sorted tuple of items for consistent hashing
        metadata_tuple = tuple(sorted(self.metadata.items())) if self.metadata else ()
        hash_tuple = (
            self.user_text,
            self.hello_rasa_user_id,
            self.hello_rasa_project_id,
            tuple(sorted(metadata_tuple)),
        )
        return hash(hash_tuple)


class GuardrailRequest(BaseModel, ABC):
    """Request for guardrails check."""

    hello_rasa_user_id: Optional[str] = Field(
        default=None,
        description="Required. User identifier for the Hello Rasa project. ",
    )
    hello_rasa_project_id: Optional[str] = Field(
        default=None,
        description="Required. Project identifier for the Hello Rasa project. ",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata for the guardrails endpoint."
    )

    @abstractmethod
    def to_json_payload(self) -> Dict[str, Any]:
        """Convert the request to a JSON payload."""
        ...


class LakeraGuardrailRequest(GuardrailRequest):
    """Request for Lakera guardrails check."""

    lakera_project_id: str = Field(
        description="Required. Project identifier for the Lakera AI project."
    )
    payload: bool = Field(
        default=True,
        description=(
            "From Lakera AI: When true the response will return a payload object "
            "containing any PII, profanity or custom detector regex matches detected, "
            "along with their location within the contents."
        ),
    )
    breakdown: bool = Field(
        default=True,
        description=(
            "From Lakera AI: When true the response will return a breakdown list of "
            "the detectors that were run, as defined in the policy, and whether each "
            "of them detected something or not."
        ),
    )

    messages: List[Dict[str, Any]] = Field(
        description=(
            "Required. From Lakera AI: List of messages comprising the interaction "
            "history with the LLM in OpenAI API Chat Completions format. Can be "
            "multiple messages of any role: user, assistant, system, tool, or "
            "developer."
        ),
    )

    # Make the model hashable by value for use as cache keys in @lru_cache decorator.
    model_config = ConfigDict(frozen=True)

    def to_json_payload(self) -> Dict[str, Any]:
        """Convert the request to a JSON payload to be sent to the Lakera endpoint."""
        metadata = self.metadata or {}
        metadata["hello_rasa_project_id"] = self.hello_rasa_project_id
        metadata["hello_rasa_user_id"] = self.hello_rasa_user_id

        json_payload: Dict[str, Any] = {
            "messages": self.messages,
            "project_id": self.lakera_project_id,
            "metadata": metadata,
        }

        if self.payload:
            json_payload["payload"] = self.payload
        if self.breakdown:
            json_payload["breakdown"] = self.breakdown

        return json_payload

    def __hash__(self) -> int:
        """Custom hash implementation that handles the messages list."""
        # Convert messages list to a tuple for consistent hashing
        if self.messages:
            messages_tuple = tuple(tuple(msg.items()) for msg in self.messages)
        else:
            messages_tuple = ()

        hash_tuple = (
            self.hello_rasa_user_id,
            self.hello_rasa_project_id,
            self.lakera_project_id,
            self.payload,
            self.breakdown,
            messages_tuple,
            tuple(sorted(self.metadata.items())) if self.metadata else (),
        )
        return hash(hash_tuple)


class GuardrailDetection(BaseModel):
    """Represents a single guardrail detection."""

    type: GuardrailType = Field(description="Type of guardrail detection.")
    original_type: str = Field(description="Original detection from the provider.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata about the detection itself."
    )


class GuardrailResponse(BaseModel):
    """Response from guardrails system."""

    hello_rasa_user_id: Optional[str] = Field(
        default=None,
        description="Required. User identifier for the Hello Rasa project. ",
    )
    hello_rasa_project_id: Optional[str] = Field(
        default=None,
        description="Required. Project identifier for the Hello Rasa project. ",
    )
    flagged: bool = Field(description="Whether any policy violations were detected.")
    detections: List[GuardrailDetection] = Field(
        default_factory=list, description="List of detected policy violations."
    )
    processing_time_ms: Optional[float] = Field(
        default=None, description="Processing time in milliseconds."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata from the provider."
    )


class LakeraGuardrailResponse(GuardrailResponse):
    """Response from Lakera AI `/guard` endpoint."""

    @classmethod
    def from_raw_response(
        cls,
        raw_response: Dict[str, Any],
        hello_rasa_user_id: str,
        hello_rasa_project_id: str,
    ) -> "LakeraGuardrailResponse":
        """Create a LakeraGuardrailResponse from a response."""
        from rasa.builder.guardrails.utils import (
            map_lakera_detector_type_to_guardrail_type,
        )

        # Get the basic information from the response and create the response object.
        flagged = raw_response.get("flagged", False)
        metadata = raw_response.get("metadata")
        response = cls(
            flagged=flagged,
            metadata=metadata,
            hello_rasa_user_id=hello_rasa_user_id,
            hello_rasa_project_id=hello_rasa_project_id,
        )

        # If the response is not flagged, return the response object.
        if not flagged:
            return response

        # If the response is flagged, parse the breakdown section.
        breakdown = raw_response.get("breakdown", [])

        # Parse the breakdown.
        detections: List[GuardrailDetection] = []
        for detector in breakdown:
            if detector.get("detected", True):
                detector_type = detector.get("detector_type")
                rasa_detection_type = map_lakera_detector_type_to_guardrail_type(
                    detector_type,
                )
                if not rasa_detection_type:
                    continue
                # Remove the detector type and the detected flag from the detector.
                # And keep the rest as part of the metadata. In case of a Lakera this
                # will include the message_id, project_id, policy_id, detector_id, etc.
                metadata = copy.deepcopy(detector)
                metadata.pop("detected")
                metadata.pop("detector_type")

                detections.append(
                    GuardrailDetection(
                        type=rasa_detection_type,
                        original_type=detector_type,
                        metadata=metadata,
                    )
                )

        # If there are detections, add them to the response.
        if detections:
            response.detections = detections

        return response


class ScopeState(BaseModel):
    blocked_until: Optional[float] = Field(
        default=None,
        description="UNIX timestamp in seconds until this scope is blocked.",
    )
    violations: List[float] = Field(
        default_factory=list,
        description="UNIX timestamps in seconds when violations occurred.",
    )

    def is_blocked(self, now: float) -> bool:
        """Return whether the scope is currently blocked.

        Args:
            now: Current time as a UNIX timestamp.

        Returns:
            True if blocked_until is set and in the future, else False.
        """
        return self.blocked_until is not None and now < self.blocked_until


class ProjectState(BaseModel):
    project: ScopeState = Field(default_factory=ScopeState)
    users: Dict[str, ScopeState] = Field(default_factory=dict)


class BlockResult(BaseModel):
    user_blocked_now: bool = False
    project_blocked_now: bool = False
