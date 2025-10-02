from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import structlog
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from rasa.builder.copilot.constants import (
    ROLE_ASSISTANT,
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_SYSTEM,
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
    TRAINING_ERROR_LOG = "training_error_log"
    E2E_TESTING_ERROR_LOG = "e2e_testing_error_log"
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
        default=None,
        description=(
            "Additional, optional context description for the logs "
            "(e.g., 'training session', 'e2e testing run', 'deployment process')"
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the log.",
    )


class EventContent(BaseContent):
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

TContentBlock = TypeVar("TContentBlock", bound=BaseContent)


class BaseCopilotChatMessage(BaseModel, ABC):
    role: str
    response_category: Optional[ResponseCategory] = Field(default=None)

    @abstractmethod
    def build_openai_message(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        pass

    @field_serializer("response_category", when_used="always")
    def _serialize_response_category(
        self, v: Optional[ResponseCategory]
    ) -> Optional[str]:
        """Serializing CopilotChatMessage, response_category should be a string."""
        return None if v is None else v.value


class BaseContentBlockCopilotChatMessage(BaseCopilotChatMessage, ABC):
    """Base class for messages that contain ContentBlock lists."""

    content: List[ContentBlock]

    def get_flattened_text_content(self) -> str:
        """Get the text content from the message."""
        return "\n".join(
            content_block.text
            for content_block in self.content
            if isinstance(content_block, TextContent)
        )

    def get_flattened_log_content(self) -> str:
        """Get the log content from the message."""
        return "\n".join(
            content_block.content
            for content_block in self.content
            if isinstance(content_block, LogContent)
        )

    def get_content_blocks_by_type(
        self, content_type: Type[TContentBlock]
    ) -> List[TContentBlock]:
        """Get the content blocks from the message by type."""
        return [
            content_block
            for content_block in self.content
            if isinstance(content_block, content_type)
        ]


class CopilotSystemMessage(BaseCopilotChatMessage):
    role: Literal["system"] = Field(
        default=ROLE_SYSTEM,
        pattern=f"^{ROLE_SYSTEM}",
        description="The system message that sets the system instructions for the LLM.",
    )

    def build_openai_message(self, prompt: str, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        """Render the system message template and return OpenAI format."""
        return {"role": ROLE_SYSTEM, "content": prompt}


class UserChatMessage(BaseContentBlockCopilotChatMessage):
    role: Literal["user"] = Field(
        default=ROLE_USER,
        pattern=f"^{ROLE_USER}",
        description="The user who sent the message.",
    )

    @classmethod
    @field_validator("content")
    def must_have_at_least_one_text(cls, v: List[ContentBlock]) -> List[ContentBlock]:
        if not any(isinstance(content_block, TextContent) for content_block in v):
            message = "User role messages must have at least one `TextContent` block."
            structlogger.error(
                "user_chat_message.missing_text_content",
                event_info=message,
                content=v,
            )
            raise ValueError(
                "UserChatMessage must contain at least one TextContent block."
            )
        return v

    @model_validator(mode="after")
    def validate_response_category(self) -> "UserChatMessage":
        """Validate value of response_category for user message.

        For 'user' role messages, only None or GUARDRAILS_POLICY_VIOLATION are allowed.
        """
        allowed_response_categories = [ResponseCategory.GUARDRAILS_POLICY_VIOLATION]
        if (
            self.response_category is not None
            and self.response_category not in allowed_response_categories
        ):
            message = (
                f"User role messages can only have response_category of `None` or "
                f"{', '.join(category.value for category in allowed_response_categories)}."  # noqa: E501
                f"Got `{self.response_category}`."
            )
            structlogger.error(
                "user_chat_message.validate_response_category"
                ".invalid_response_category",
                event_info=message,
                response_category=self.response_category,
                allowed_response_categories=allowed_response_categories,
                role=self.role,
            )
            raise ValueError(message)

        return self

    def build_openai_message(  # type: ignore[no-untyped-def]
        self, prompt: Optional[str] = None, *args, **kwargs
    ) -> Dict[str, Any]:
        # If a prompt is provided, add it to the message content as additional
        # instructions
        if prompt:
            return {
                "role": ROLE_USER,
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": self.get_flattened_text_content()},
                ],
            }
        # Return simple text content (useful for showing the history)
        else:
            return {"role": ROLE_USER, "content": self.get_flattened_text_content()}


class CopilotChatMessage(BaseContentBlockCopilotChatMessage):
    role: Literal["copilot"]

    def build_openai_message(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        # For now the Copilot responds only with the text content and all the content
        # is formatted as a markdown.
        return {"role": ROLE_ASSISTANT, "content": self.get_flattened_text_content()}


class InternalCopilotRequestChatMessage(BaseContentBlockCopilotChatMessage):
    role: Literal["internal_copilot_request"]

    @model_validator(mode="after")
    def validate_response_category(self) -> "InternalCopilotRequestChatMessage":
        """Validate value of response_category for internal copilot request message.

        For 'internal_copilot_request' role messages, only `TRAINING_ERROR_LOG_ANALYSIS`
        and `E2E_TESTING_ERROR_LOG_ANALYSIS` response categories are allowed.
        """
        allowed_response_categories = [
            ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
            ResponseCategory.E2E_TESTING_ERROR_LOG_ANALYSIS,
        ]
        if self.response_category not in allowed_response_categories:
            message = (
                f"Copilot Internal Roles request messages can only have of "
                f"{', '.join(category.value for category in allowed_response_categories)}. "  # noqa: E501
                f"Got `{self.response_category}`."
            )
            structlogger.error(
                "internal_copilot_request_chat_message.validate_response_category"
                ".invalid_response_category",
                event_info=message,
                response_category=self.response_category,
                allowed_response_categories=allowed_response_categories,
                role=self.role,
            )
            raise ValueError(message)

        return self

    def build_openai_message(self, prompt: str, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
        """Build OpenAI message with pre-rendered prompt.

        The prompt should be rendered externally using the content from this message
        (logs, files, any additional context outside of this message, etc.) before
        being passed to this method.
        """
        return {"role": ROLE_USER, "content": prompt}


# Union type for all possible chat message types
ChatMessage = Union[
    CopilotSystemMessage,
    UserChatMessage,
    CopilotChatMessage,
    InternalCopilotRequestChatMessage,
]


class CopilotContext(BaseModel):
    """Model containing the context used by the copilot to generate a response."""

    assistant_logs: str = Field(default="")
    assistant_files: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "The assistant files. Key is the file path, value is the file content."
        ),
    )
    copilot_chat_history: List[ChatMessage] = Field(default_factory=list)
    tracker_context: Optional[TrackerContext] = Field(default=None)

    class Config:
        """Config for LLMBuilderContext."""

        arbitrary_types_allowed = True


class CopilotRequest(BaseModel):
    """Request model for the copilot endpoint."""

    copilot_chat_history: List[ChatMessage] = Field(
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

    @field_validator("copilot_chat_history", mode="before")
    @classmethod
    def parse_chat_history(cls, v: List[Dict[str, Any]]) -> List[ChatMessage]:
        """Manually parse chat history messages based on role field."""
        parsed_messages: List[ChatMessage] = []
        available_roles = [ROLE_USER, ROLE_COPILOT, ROLE_COPILOT_INTERNAL]
        for message_data in v:
            role = message_data.get("role")

            if role == ROLE_USER:
                parsed_messages.append(UserChatMessage(**message_data))

            elif role == ROLE_COPILOT:
                parsed_messages.append(CopilotChatMessage(**message_data))

            elif role == ROLE_COPILOT_INTERNAL:
                parsed_messages.append(
                    InternalCopilotRequestChatMessage(**message_data)
                )

            else:
                message = (
                    f"Unknown role '{role}' in chat message. "
                    f"Available roles are: {', '.join(available_roles)}."
                )
                structlogger.error(
                    "copilot_request.parse_chat_history.unknown_role",
                    event_info=message,
                    role=role,
                    available_roles=available_roles,
                )
                raise ValueError(message)

        return parsed_messages

    @property
    def last_message(self) -> Optional[ChatMessage]:
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

    @property
    @abstractmethod
    def sse_data(self) -> Dict[str, Any]:
        """Extract the SSE data payload."""
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
            data=self.sse_data,
        )

    @property
    def sse_data(self) -> Dict[str, Any]:
        """Extract the SSE data payload."""
        return {
            "content": self.content,
            "response_category": self.response_category.value,
            "completeness": self.response_completeness.value,
        }


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
            data=self.sse_data,
        )

    @property
    def sse_data(self) -> Dict[str, Any]:
        """Extract the SSE data payload."""
        return {
            "index": self.index,
            "title": self.title,
            "url": self.url,
            "response_category": self.response_category.value,
            "completeness": self.response_completeness.value,
        }


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
            data=self.sse_data,
        )

    @property
    def sse_data(self) -> Dict[str, Any]:
        """Extract the SSE data payload."""
        return {
            "references": [
                reference.model_dump(include={"index", "title", "url"})
                for reference in self.references
            ],
            "response_category": self.response_category.value,
            "completeness": self.response_completeness.value,
        }

    def sort_references(self) -> None:
        """Sort references by index value."""
        sorted_references = sorted(
            self.references, key=lambda reference: (0, int(reference.index))
        )

        self.references = sorted_references


class TrainingErrorLog(CopilotOutput):
    """Represents an error log."""

    logs: List[LogContent]
    response_category: ResponseCategory = Field(
        default=ResponseCategory.TRAINING_ERROR_LOG,
        frozen=True,
    )
    response_completeness: ResponseCompleteness = ResponseCompleteness.COMPLETE

    @model_validator(mode="after")
    def validate_response_category(self) -> "TrainingErrorLog":
        """Validate that response_category has the correct default value."""
        if self.response_category != ResponseCategory.TRAINING_ERROR_LOG:
            raise ValueError(
                f"TrainingErrorLog response_category must be "
                f"{ResponseCategory.TRAINING_ERROR_LOG}, "
                f"got `{self.response_category}`."
            )
        return self

    def to_sse_event(self) -> ServerSentEvent:
        """Convert to SSE event format."""
        return ServerSentEvent(
            event="copilot_response",
            data=self.sse_data,
        )

    @property
    def sse_data(self) -> Dict[str, Any]:
        """Extract the SSE data payload."""
        return {
            "logs": [log.model_dump() for log in self.logs],
            "response_category": self.response_category.value,
            "completeness": self.response_completeness.value,
        }


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
    tracker_event_attachments: List[EventContent] = Field(
        ...,
        description=(
            "The tracker event attachments passed with the user message used as "
            "an additional context."
        ),
    )

    class Config:
        """Config for CopilotGenerationContext."""

        arbitrary_types_allowed = True
