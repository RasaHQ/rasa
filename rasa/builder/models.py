"""Pydantic models for request/response validation."""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rasa.cli.scaffold import ProjectTemplateName
from rasa.shared.importers.importer import TrainingDataImporter

structlogger = structlog.get_logger()


class PromptRequest(BaseModel):
    """Request model for prompt-to-bot endpoint."""

    prompt: str = Field(
        ..., min_length=1, max_length=10000, description="The skill description prompt"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()


class TemplateRequest(BaseModel):
    """Request model for template-to-bot endpoint."""

    template_name: ProjectTemplateName = Field(
        ...,
        description=(
            f"The template name to use ({ProjectTemplateName.supported_values()})"
        ),
    )

    @field_validator("template_name")
    @classmethod
    def validate_template_name(cls, v: Any) -> Any:
        if v not in ProjectTemplateName:
            raise ValueError(
                f"Template name must be one of {ProjectTemplateName.supported_values()}"
            )
        return v


class RestoreFromBackupRequest(BaseModel):
    """Request model for backup-to-bot endpoint."""

    presigned_url: str = Field(
        ...,
        min_length=1,
        description="Presigned URL to download tar.gz backup file.",
    )

    @field_validator("presigned_url")
    @classmethod
    def validate_presigned_url(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Presigned URL cannot be empty or whitespace only")

        # Basic URL validation
        url = v.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("Presigned URL must be a valid HTTP/HTTPS URL")

        return url


class BotDataUpdateRequest(BaseModel):
    """Request model for bot data updates."""

    domain_yml: Optional[str] = Field(None, alias="domain.yml")
    flows_yml: Optional[str] = Field(None, alias="flows.yml")
    config_yml: Optional[str] = Field(None, alias="config.yml")

    # Allow using either field names or aliases when creating the model
    model_config = ConfigDict(populate_by_name=True)


class BotData(BaseModel):
    """Data of a running assistant."""

    domain: Dict[str, Any] = Field(..., description="The domain of the assistant.")
    flows: Dict[str, Any] = Field(..., description="The flows of the assistant.")


class AssistantInfo(BaseModel):
    """Basic information about the loaded assistant."""

    assistant_id: Optional[str] = Field(
        None,
        description=(
            "Assistant identifier coming from `assistant_id` in the model "
            "metadata (config.yml)."
        ),
    )


class ApiErrorResponse(BaseModel):
    """API error response model."""

    status: Literal["error"] = "error"
    error: str = Field(...)
    details: Optional[Dict[str, Any]] = Field(None)


class ServerSentEventType(str, Enum):
    progress = "progress"
    error = "error"
    _EOF = "_EOF"


class ServerSentEvent(BaseModel):
    """Generic Server-Sent Event payload."""

    event: str = Field(..., description="SSE event name / type")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON-serializable payload",
    )

    @classmethod
    def build(cls, event: str, data: Any) -> "ServerSentEvent":
        """General-purpose constructor.

        Args:
            event: The event name (e.g. "progress", "error").
            data: Arbitrary key-value pairs to include in the payload.

        Returns:
            A ServerSentEvent instance with the specified event and data.
        """
        return cls(event=event, data=data)

    @classmethod
    def eof(cls) -> "ServerSentEvent":
        """Helper that returns the special end-of-stream marker."""
        return cls(event=ServerSentEventType._EOF.value, data={})

    def format(self) -> str:
        """Return the text representation used by SSE protocols."""
        return (
            f"event: {self.event}\n"
            f"data: {json.dumps(self.data, separators=(', ', ': '))}\n\n"
        )


class JobStatusEvent(ServerSentEvent):
    """Job status event with special handling for progress and error states."""

    @classmethod
    def from_status(
        cls,
        status: str,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "JobStatusEvent":
        """Factory for job-status events.

        Args:
            status: The job status (e.g. "training", "train_success").
            message: Optional error message for error events.
            payload: Optional additional payload data to include in the event.

        Returns:
            A JobStatusEvent instance with the appropriate event type and data.
        """
        event_type = (
            ServerSentEventType.error if message else ServerSentEventType.progress
        )

        event_payload: Dict[str, Any] = {"status": status}
        if message:
            event_payload["message"] = message
        if payload:
            event_payload.update(payload)

        return cls(event=event_type.value, data=event_payload)


class ValidationResult(BaseModel):
    """Result of validation operation."""

    is_valid: bool = Field(...)
    errors: Optional[List[str]] = Field(None)
    warnings: Optional[List[str]] = Field(None)


class TrainingResult(BaseModel):
    """Result of training operation."""

    success: bool = Field(...)
    model_path: Optional[str] = Field(None)
    error: Optional[str] = Field(None)


BotFiles = Dict[str, Optional[str]]


class JobStatus(str, Enum):
    received = "received"
    done = "done"
    error = "error"

    generating = "generating"
    generation_success = "generation_success"
    generation_error = "generation_error"

    training = "training"
    train_success = "train_success"
    train_success_message = "train_success_message"
    train_error = "train_error"

    validating = "validating"
    validation_success = "validation_success"
    validation_error = "validation_error"

    copilot_analysis_start = "copilot_analysis_start"
    copilot_analyzing = "copilot_analyzing"
    copilot_analysis_success = "copilot_analysis_success"
    copilot_analysis_error = "copilot_analysis_error"

    copilot_template_prompt = "copilot_template_prompt"
    copilot_welcome_message = "copilot_welcome_message"


class JobCreateResponse(BaseModel):
    job_id: str = Field(...)
    status: JobStatus = JobStatus.received


class TrainingInput(BaseModel):
    """Input for training a model."""

    model_config = {"arbitrary_types_allowed": True}

    importer: TrainingDataImporter = Field(..., description="Training data importer")
    endpoints_file: Path = Field(..., description="Path to the endpoints file")
    config_file: Path = Field(..., description="Path to the config file")


class AgentStatus(str, Enum):
    """Status of the agent."""

    not_loaded = "not_loaded"
    ready = "ready"
    not_ready = "not_ready"
