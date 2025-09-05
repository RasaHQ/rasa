from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import structlog
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import BaseModel, Field, field_serializer, model_validator
from typing_extensions import Annotated

from rasa.builder.copilot.constants import (
    ROLE_ASSISTANT,
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_USER,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.models import ServerSentEvent
from rasa.builder.shared.tracker_context import TrackerContext

structlogger = structlog.get_logger()


class ResponseCompleteness(Enum):
    """Enum for response completeness levels."""

    TOKEN = "token"  # Streaming token/chunk
    COMPLETE = "complete"  # Complete response (e.g., templated responses)


class ResponseCategory(Enum):
    """Enum for different categories of responses."""

    # Copilot generated content
    COPILOT = "copilot"
    REFERENCE = "reference"
    REFERENCE_ENTRY = "reference_entry"
    # When Copilot detects a roleplay request / intent
    ROLEPLAY_DETECTION = "roleplay_detection"
    # When Copilot detects an out-of-scope request
    OUT_OF_SCOPE_DETECTION = "out_of_scope_detection"
    # When Copilot does not understand what caused the error
    ERROR_FALLBACK = "error_fallback"
    # When a policy violation is detected
    GUARDRAILS_POLICY_VIOLATION = "guardrails_policy_violation"
    # When Copilot access is blocked after repeated violations
    GUARDRAILS_BLOCKED = "guardrails_blocked"
    # When Copilot detects request for KB content
    KNOWLEDGE_BASE_ACCESS_REQUESTED = "knowledge_base_access_requested"
    # When Copilot analyzes error logs and provides suggestions
    TRAINING_ERROR_LOG_ANALYSIS = "training_error_log_analysis"
    E2E_TESTING_ERROR_LOG_ANALYSIS = "e2e_testing_error_log_analysis"

    # Conversation history signature
    SIGNATURE = "signature"


class BaseContent(BaseModel):
    type: str


class LinkContent(BaseContent):
    type: Literal["link"]
    url: str
    label: str


class ButtonContent(BaseContent):
    type: Literal["button"]
    payload: str
    label: str


class TextContent(BaseContent):
    type: Literal["text"]
    text: str


class CodeContent(BaseContent):
    type: Literal["code"]
    text: str


class FileContent(BaseContent):
    type: Literal["file"]
    file_path: str
    file_content: str


class LogContent(BaseContent):
    type: Literal["log"]
    content: str = Field(..., description="Logs, error messages, stack traces, etc.")
    context: Optional[str] = Field(
        None,
        description=(
            "Additional, optional context description for the logs "
            "(e.g., 'training session', 'e2e testing run', 'deployment process')"
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the log.",
    )


class EventContent(BaseModel):
    type: Literal["event"]
    event: str = Field(..., description="The event's type_name")

    event_data: Dict[str, Any] = Field(
        default_factory=dict, description="Contains event-specific data fields."
    )

    @model_validator(mode="before")
    @classmethod
    def _collect_event_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        generic = {"type", "event"}
        data["event_data"] = {
            key: data.pop(key) for key in list(data.keys()) if key not in generic
        }
        return data

    class Config:
        extra = "forbid"


ContentBlock = Annotated[
    Union[
        TextContent,
        CodeContent,
        FileContent,
        LogContent,
        EventContent,
        LinkContent,
        ButtonContent,
    ],
    Field(
        discriminator="type",
        description=(
            "The content of the message. "
            "The content is expected to be a list of content blocks. "
            "The content blocks are expected to be one of the following types: "
            "text, link, code, or file."
        ),
    ),
]


class CopilotChatMessage(BaseModel):
    """Model for a single chat messages between the user and the copilot."""

    role: str = Field(
        ...,
        pattern=f"^({ROLE_USER}|{ROLE_COPILOT}|{ROLE_COPILOT_INTERNAL})$",
        description="The role of the message sender.",
    )
    content: List[ContentBlock]
    response_category: Optional[ResponseCategory] = Field(
        None,
        description=(
            "The category/source of this message. For user role messages, only `None` "
            "or `GUARDRAILS_POLICY_VIOLATION` are allowed. For copilot role messages, "
            "any category is permitted."
        ),
    )

    @model_validator(mode="after")
    def validate_response_category_for_role(self) -> "CopilotChatMessage":
        """Validate value of response_category for the role of the message.

        For 'user' role messages, only None or GUARDRAILS_POLICY_VIOLATION are allowed.
        For 'copilot' role messages, any category is permitted.
        For 'rasa_internal' role messages, any category is permitted.
        """
        if (
            self.role == ROLE_USER
            and self.response_category is not None
            and self.response_category != ResponseCategory.GUARDRAILS_POLICY_VIOLATION
        ):
            message = (
                f"User role messages can only have response_category of `None` or "
                f"`{ResponseCategory.GUARDRAILS_POLICY_VIOLATION}`, "
                f"got `{self.response_category}`."
            )
            structlogger.error(
                "copilot_chat_message.validate_response_category_for_role"
                ".invalid_response_category",
                event_info=message,
                response_category=self.response_category,
                role=self.role,
            )
            raise ValueError(message)

        return self

    @field_serializer("response_category", when_used="always")
    def _serialize_response_category(
        self, v: Optional[ResponseCategory]
    ) -> Optional[str]:
        """Serializing CopilotChatMessage, response_category should be a string."""
        return None if v is None else v.value

    def get_text_content(self) -> str:
        """Concatenate all 'text' content blocks into a single string."""
        return "\n".join(
            content_block.text
            for content_block in self.content
            if isinstance(content_block, TextContent)
        )

    def get_log_content(self) -> str:
        """Concatenate all 'log' content blocks into a single string."""
        return "\n".join(
            content_block.content
            for content_block in self.content
            if isinstance(content_block, LogContent)
        )

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI message format for API calls."""
        role_to_openai_format = {
            ROLE_USER: self._user_message_to_openai_format,
            ROLE_COPILOT: self._copilot_message_to_openai_format,
            ROLE_COPILOT_INTERNAL: self._copilot_message_to_openai_format,
        }
        return role_to_openai_format[self.role]()

    def _user_message_to_openai_format(self) -> Dict[str, Any]:
        role = self._map_role_to_openai()
        content = self.get_text_content()
        return {"role": role, "content": content}

    def _copilot_message_to_openai_format(self) -> Dict[str, Any]:
        role = self._map_role_to_openai()
        # For now the Copilot responds only with the text content and all the content
        # is formatted as a markdown.
        # TODO: Once we start predicting the files, and expecting other content blocks
        #       we should update this.
        content = self.get_text_content()
        return {"role": role, "content": content}

    def _map_role_to_openai(self) -> str:
        """Map internal roles to OpenAI-compatible roles."""
        role_mapping = {
            ROLE_USER: ROLE_USER,
            ROLE_COPILOT: ROLE_ASSISTANT,
            ROLE_COPILOT_INTERNAL: ROLE_USER,
        }
        if self.role not in role_mapping.keys():
            structlogger.error(
                "copilot_chat_message.to_openai_format.invalid_role",
                event_info=(
                    f"Invalid role: `{self.role}`. "
                    f"Only {', '.join(role_mapping.keys())} roles are supported."
                ),
                role=self.role,
            )
            raise ValueError(f"Invalid role: {self.role}")

        return role_mapping[self.role]


class CopilotRequest(BaseModel):
    """Request model for the copilot endpoint."""

    copilot_chat_history: List[CopilotChatMessage] = Field(
        ...,
        description=(
            "The chat history between the user and the copilot. "
            "Used to generate a new response based on the previous conversation."
        ),
    )
    session_id: str = Field(
        ...,
        description=(
            "The session ID of chat session with the assistant. "
            "Used to fetch the conversation from the tracker."
        ),
    )
    history_signature: Optional[str] = Field(
        default=None,
        description="HMAC signature (base64url) for the provided chat history.",
    )
    signature_version: Optional[str] = Field(
        default=None,
        description='Signature scheme version (e.g. "v1").',
    )

    @property
    def last_message(self) -> Optional[CopilotChatMessage]:
        """Get the last message from the copilot chat history."""
        if not self.copilot_chat_history:
            return None
        return self.copilot_chat_history[-1]


class CopilotOutput(BaseModel, ABC):
    """Base class for response events."""

    response_completeness: ResponseCompleteness = Field(
        description=(
            "Indicates whether this is a streaming token (TOKEN) or a complete "
            "response (COMPLETE)"
        ),
    )
    response_category: ResponseCategory = Field(
        description=(
            "The category/source of this response. Each response type has a fixed "
            "category that cannot be changed. "
        ),
        frozen=True,
    )

    @abstractmethod
    def to_sse_event(self) -> ServerSentEvent:
        """Convert to SSE event format."""
        pass


class GeneratedContent(CopilotOutput):
    """Represents generated content from the LLM to be streamed."""

    content: str
    response_category: ResponseCategory = Field(frozen=True)
    response_completeness: ResponseCompleteness = ResponseCompleteness.TOKEN

    def to_sse_event(self) -> ServerSentEvent:
        """Convert to SSE event format."""
        return ServerSentEvent(
            event="copilot_response",
            data={
                "content": self.content,
                "response_category": self.response_category.value,
                "completeness": self.response_completeness.value,
            },
        )


class ReferenceEntry(CopilotOutput):
    """Represents a reference entry with title and url."""

    index: int
    title: str
    url: str
    response_category: ResponseCategory = Field(
        default=ResponseCategory.REFERENCE_ENTRY,
        frozen=True,
    )
    response_completeness: ResponseCompleteness = ResponseCompleteness.COMPLETE

    @model_validator(mode="after")
    def validate_response_category(self) -> "ReferenceEntry":
        """Validate that response_category has the correct default value."""
        if self.response_category != ResponseCategory.REFERENCE_ENTRY:
            raise ValueError(
                f"ReferenceEntry response_category must be "
                f"{ResponseCategory.REFERENCE_ENTRY}, got `{self.response_category}`."
            )
        return self

    def to_sse_event(self) -> ServerSentEvent:
        """Convert to SSE event format."""
        return ServerSentEvent(
            event="copilot_response",
            data={
                "index": self.index,
                "title": self.title,
                "url": self.url,
                "response_category": self.response_category.value,
                "completeness": self.response_completeness.value,
            },
        )


class ReferenceSection(CopilotOutput):
    """Represents a reference section with documentation links."""

    references: list[ReferenceEntry]
    response_category: ResponseCategory = Field(
        default=ResponseCategory.REFERENCE,
        frozen=True,
    )
    response_completeness: ResponseCompleteness = ResponseCompleteness.COMPLETE

    @model_validator(mode="after")
    def validate_response_category(self) -> "ReferenceSection":
        """Validate that response_category has the correct default value."""
        if self.response_category != ResponseCategory.REFERENCE:
            raise ValueError(
                f"ReferenceSection response_category must be "
                f"{ResponseCategory.REFERENCE}, got `{self.response_category}`."
            )
        return self

    def to_sse_event(self) -> ServerSentEvent:
        """Convert to SSE event format."""
        return ServerSentEvent(
            event="copilot_response",
            data={
                "references": [
                    reference.model_dump(include={"index", "title", "url"})
                    for reference in self.references
                ],
                "response_category": self.response_category.value,
                "completeness": self.response_completeness.value,
            },
        )

    def sort_references(self) -> None:
        """Sort references by index value."""
        sorted_references = sorted(
            self.references, key=lambda reference: (0, int(reference.index))
        )

        self.references = sorted_references


class CopilotContext(BaseModel):
    """Model containing the context used by the copilot to generate a response."""

    assistant_logs: str = Field("")
    assistant_files: Dict[str, str] = Field({})
    copilot_chat_history: List["CopilotChatMessage"] = Field([])
    tracker_context: Optional[TrackerContext] = Field(None)

    class Config:
        """Config for LLMBuilderContext."""

        arbitrary_types_allowed = True


class UsageStatistics(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None

    def reset(self) -> None:
        """Reset usage statistics to their default values."""
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.model = None

    def update_from_stream_chunk(self, chunk: ChatCompletionChunk) -> None:
        """Update usage statistics from an OpenAI stream chunk.

        Args:
            chunk: The OpenAI stream chunk containing usage statistics.
        """
        if not (usage := getattr(chunk, "usage", None)):
            return

        self.prompt_tokens = usage.prompt_tokens
        self.completion_tokens = usage.completion_tokens
        self.total_tokens = usage.total_tokens
        self.model = getattr(chunk, "model", None)


class SigningContext(BaseModel):
    secret: Optional[str] = Field(None)
    default_version: str = Field("v1", description="Default signature version")

    @property
    def available(self) -> bool:
        """Signing is enabled if a non-empty secret is present."""
        secret = (self.secret or "").strip()
        return bool(secret)


class CopilotGenerationContext(BaseModel):
    """Container for copilot generation context and supporting evidence.

    This class organizes the context and supporting evidence information used by the
    copilot's generate_response method, providing a cleaner interface than returning
    a tuple for the non-streaming data.
    """

    relevant_documents: List["Document"] = Field(
        ...,
        description=(
            "The relevant documents used as supporting evidence for the respons."
        ),
    )
    system_message: Dict[str, Any] = Field(
        ..., description="The system message with instructions."
    )
    chat_history: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "The chat history messages (excluding the last message) used as a context."
        ),
    )
    last_user_message: Optional[Dict[str, Any]] = Field(
        None, description="The last user message with context that was processed."
    )

    class Config:
        """Config for CopilotGenerationContext."""

        arbitrary_types_allowed = True
