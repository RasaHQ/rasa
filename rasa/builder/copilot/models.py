import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import structlog
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.responses import ResponseCompletedEvent
from pydantic import (
    BaseModel,
    Field,
    computed_field,
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

    # Copilot generated content. Used for regular generated responses.
    # For agent handler, content part types (START, DELTA, END) are categorized
    # as COPILOT when extracting response category for telemetry/categorization.
    COPILOT = "copilot"
    # The three categories for the agentic copilot response with streaming tokens.
    # These are internal categories for content parts. When categorizing responses as a
    # whole (via extract_response_category), these map to COPILOT.
    COPILOT_TEXT_CONTENT_PART_START = "copilot_text_content_part_start"
    COPILOT_TEXT_CONTENT_PART_DELTA = "copilot_text_content_part_delta"
    COPILOT_TEXT_CONTENT_PART_END = "copilot_text_content_part_end"
    # Reference categories
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
    # When Copilot is generating reasoning steps
    REASONING = "reasoning"
    # When Copilot analyzes error logs and provides suggestions
    TRAINING_ERROR_LOG_ANALYSIS = "training_error_log_analysis"
    E2E_TESTING_ERROR_LOG_ANALYSIS = "e2e_testing_error_log_analysis"
    TRAINING_ERROR_LOG = "training_error_log"
    E2E_TESTING_ERROR_LOG = "e2e_testing_error_log"
    # Conversation history signature
    SIGNATURE = "signature"
    # When an exception occurs during streaming
    EXCEPTION = "exception"
    # When Copilot invokes an MCP tool
    MCP_TOOL_CALL = "mcp_tool_call"
    # When Copilot creates or updates a task plan
    TASK_PLANNING = "task_planning"
    # When a commit info is sent
    COMMIT = "commit"

    # Orchestrator detection categories
    # When orchestrator cannot understand the user's input (unclear/gibberish)
    UNCLEAR_INPUT_DETECTION = "unclear_input_detection"
    # When orchestrator detects a greeting message
    GREETING_DETECTION = "greeting_detection"
    # When orchestrator detects a goodbye message
    GOODBYE_DETECTION = "goodbye_detection"
    # When orchestrator detects a rasa introduction message
    RASA_INTRODUCTION_DETECTION = "rasa_introduction_detection"
    # When orchestrator detects a copilot introduction message
    COPILOT_INTRODUCTION_DETECTION = "copilot_introduction_detection"


class MCPToolCallStatus(str, Enum):
    """Status of an MCP tool call."""

    CALLED = "called"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Status of a plan task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskType(str, Enum):
    """Type of event source in the merged stream."""

    STREAM = "stream"
    MCP = "mcp"
    PLAN = "plan"


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
        """Config for EventContent."""

        extra = "forbid"


class ReferenceItem(BaseModel):
    index: int
    title: str
    url: str


class ReferencesContent(BaseContent):
    type: Literal["references"]
    references: List[ReferenceItem]


class CommitContent(BaseContent):
    type: Literal["commit"]
    commit: Dict[str, Any]


class PlanContent(BaseContent):
    """Content block for task planning data."""

    type: Literal["plan"]
    tasks: List[Dict[str, Any]] = Field(
        description="List of task items with id, content, and status"
    )


class LogItem(BaseModel):
    type: Literal["log"] = "log"
    content: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LogsContent(BaseContent):
    type: Literal["logs"]
    logs: List[LogItem]


ContentBlock = Annotated[
    Union[
        TextContent,
        CodeContent,
        FileContent,
        LogContent,
        EventContent,
        LinkContent,
        ButtonContent,
        ReferencesContent,
        LogsContent,
        CommitContent,
        PlanContent,
    ],
    Field(
        discriminator="type",
        description=(
            "The content of the message. "
            "The content is expected to be a list of content blocks. "
            "The content blocks are expected to be one of the following types: "
            "text, link, code, file, references, logs, plan, or event."
        ),
    ),
]

TContentBlock = TypeVar("TContentBlock", bound=BaseContent)


class BaseCopilotChatMessage(BaseModel, ABC):
    role: str
    response_category: Optional[ResponseCategory] = Field(default=None)
    timestamp: Optional[float] = Field(
        default=None, description="Unix timestamp (UTC) when the message was created"
    )

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


def create_chat_message_from_dict(message_data: Dict[str, Any]) -> ChatMessage:
    """Parse a single chat message dictionary into a ChatMessage object.

    This utility function manually parses a chat message dictionary into the
    appropriate ChatMessage type based on its role field.

    Args:
        message_data: Dictionary containing chat message data

    Returns:
        Parsed ChatMessage object

    Raises:
        ValueError: If an unknown role is encountered

    Example:
        >>> message_data = {
        ...     "role": "user",
        ...     "content": [{"type": "text", "text": "Hello"}]
        ... }
        >>> message = parse_chat_message_from_dict(message_data)
        >>> isinstance(message, UserChatMessage)
        True
        >>> message.role
        'user'
    """
    available_roles = [ROLE_USER, ROLE_COPILOT, ROLE_COPILOT_INTERNAL]
    role = message_data.get("role")

    if role == ROLE_USER:
        return UserChatMessage(**message_data)
    elif role == ROLE_COPILOT:
        return CopilotChatMessage(**message_data)
    elif role == ROLE_COPILOT_INTERNAL:
        return InternalCopilotRequestChatMessage(**message_data)
    else:
        message = (
            f"Unknown role '{role}' in chat message. "
            f"Available roles are: {', '.join(available_roles)}."
        )
        structlogger.error(
            "models.create_chat_message_from_dict.unknown_role",
            event_info=message,
            role=role,
            available_roles=available_roles,
        )
        raise ValueError(message)


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

    def get_last_user_message(self) -> Optional[UserChatMessage]:
        """Get the last user message from the chat history if available.

        The method will return the last message if it is a UserChatMessage, otherwise
        it will return None.
        """
        if not self.copilot_chat_history:
            return None

        last_message = self.copilot_chat_history[-1]
        if isinstance(last_message, UserChatMessage):
            return last_message

        return None

    def get_last_request_message(
        self,
    ) -> Optional[Union[UserChatMessage, InternalCopilotRequestChatMessage]]:
        """Get the last request message from the chat history if available.

        The method will return the last message if it is either a UserChatMessage or
        an InternalCopilotRequestChatMessage (both are request messages, not responses).
        Otherwise it will return None.

        Returns:
            The last request message (UserChatMessage or
            InternalCopilotRequestChatMessage), or None if the last message is not a
            request message.
        """
        if not self.copilot_chat_history:
            return None

        last_message = self.copilot_chat_history[-1]
        if isinstance(
            last_message,
            (
                UserChatMessage,
                InternalCopilotRequestChatMessage,
            ),
        ):
            return last_message

        return None


class CopilotTurnRequest(BaseModel):
    """Request model for a single copilot turn.

    Only accepts user messages - copilot responses are generated by the system.
    """

    session_id: str = Field(
        ...,
        description=(
            "The session ID of chat session with the assistant. "
            "Used to fetch the conversation from the tracker."
        ),
    )
    message: UserChatMessage = Field(
        ...,
        description="The user message to process.",
    )
    chat_id: Optional[str] = Field(
        default=None,
        description="The chat ID to store the message in.",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="The project ID to store the message in.",
    )


class CopilotHistoryResponse(BaseModel):
    """Response model for history retrieval."""

    messages: list[ChatMessage]


class ConversationKey(BaseModel):
    """Conversation identity used for server-side history storage."""

    chat_id: str

    def to_tuple(self) -> tuple[str]:
        """Convert to a tuple for use as SQL parameters."""
        return (self.chat_id.strip(),)


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


class ExceptionContent(GeneratedContent):
    """Generated content for exceptions.

    Captures full diagnostic context for exceptions including the exception
    type, message, and stack trace.  Diagnostic details are exposed as
    computed properties derived from ``original_exception`` and bundled
    into ``exception_metadata`` for easy consumption by telemetry.
    """

    content: str
    response_category: ResponseCategory = Field(
        default=ResponseCategory.EXCEPTION, frozen=True
    )
    response_completeness: ResponseCompleteness = Field(
        default=ResponseCompleteness.COMPLETE, frozen=True
    )
    original_exception: Optional[BaseException] = Field(
        default=None,
        description="The original exception that occurred.",
    )

    class Config:
        """Config for ExceptionContent."""

        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Computed diagnostic properties
    # ------------------------------------------------------------------

    @property
    def exception_type(self) -> Optional[str]:
        """Fully-qualified type/class name of the exception."""
        if self.original_exception is None:
            return None
        return type(self.original_exception).__qualname__

    @property
    def exception_message(self) -> Optional[str]:
        """Exception message, falling back to ``repr`` when empty."""
        if self.original_exception is None:
            return None
        msg = str(self.original_exception)
        return msg if msg else repr(self.original_exception)

    @property
    def exception_stack_trace(self) -> Optional[str]:
        """Full stack trace captured from the exception's traceback."""
        if self.original_exception is None:
            return None
        return "".join(
            traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__,
            )
        )

    @property
    def exception_cause(self) -> Optional[str]:
        """The chained cause (``__cause__`` or ``__context__``) if present."""
        if self.original_exception is None:
            return None
        cause = self.original_exception.__cause__ or self.original_exception.__context__
        if cause is None:
            return None
        return repr(cause)

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Diagnostic metadata dict for telemetry / tracing.

        Returns ``None`` when no exception is stored; otherwise a dict
        containing ``exception_type``, ``exception_message``, and
        ``exception_stack_trace``.
        """
        if self.original_exception is None:
            return None
        return {
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "exception_stack_trace": self.exception_stack_trace,
            "exception_cause": self.exception_cause,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def safe_metadata(self) -> Optional[Dict[str, Any]]:
        """Safe metadata dict for telemetry / tracing."""
        if self.metadata is None:
            return None
        safe_metadata = deepcopy(self.metadata)
        safe_metadata.pop("exception_stack_trace", None)
        return safe_metadata

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @property
    def stringified_original_exception(self) -> Optional[str]:
        """Get the stringified original exception."""
        if self.original_exception is None:
            return None
        return self.serialize_exception(self.original_exception)

    @field_serializer("original_exception")
    def serialize_exception(self, value: Optional[Exception]) -> Optional[str]:
        """Serialize exception to string representation."""
        if value is None:
            return None
        return repr(value)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_response_category(self) -> "ExceptionContent":
        """Validate that response_category is EXCEPTION."""
        _validate_response_category(
            ResponseCategory.EXCEPTION,
            self.response_category,
            type(self),
        )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "ExceptionContent":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
        )
        return self


class GuardrailPolicyViolationContent(GeneratedContent):
    """Generated content for guardrail policy violations."""

    content: str
    response_category: ResponseCategory = Field(
        default=ResponseCategory.GUARDRAILS_POLICY_VIOLATION, frozen=True
    )
    response_completeness: ResponseCompleteness = Field(
        default=ResponseCompleteness.COMPLETE, frozen=True
    )

    @model_validator(mode="after")
    def validate_response_category(self) -> "GuardrailPolicyViolationContent":
        """Validate that response_category is GUARDRAILS_POLICY_VIOLATION."""
        _validate_response_category(
            ResponseCategory.GUARDRAILS_POLICY_VIOLATION,
            self.response_category,
            type(self),
        )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "GuardrailPolicyViolationContent":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
        )
        return self


class GuardrailBlockedContent(GeneratedContent):
    """Generated content for guardrail blocks."""

    content: str
    response_category: ResponseCategory = Field(
        default=ResponseCategory.GUARDRAILS_BLOCKED, frozen=True
    )
    response_completeness: ResponseCompleteness = Field(
        default=ResponseCompleteness.COMPLETE, frozen=True
    )

    @model_validator(mode="after")
    def validate_response_category(self) -> "GuardrailBlockedContent":
        """Validate that response_category is GUARDRAILS_BLOCKED."""
        _validate_response_category(
            ResponseCategory.GUARDRAILS_BLOCKED,
            self.response_category,
            type(self),
        )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "GuardrailBlockedContent":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
        )
        return self


class ControlledPredictionContent(GeneratedContent):
    """Content for controlled predictions.

    Controlled predictions result in predefined templated responses.
    """

    content: str
    response_category: ResponseCategory = Field(frozen=True)
    response_completeness: ResponseCompleteness = Field(
        default=ResponseCompleteness.COMPLETE, frozen=True
    )

    @model_validator(mode="after")
    def validate_response_category(self) -> "ControlledPredictionContent":
        """Validate that response_category is a valid controlled prediction category."""
        valid_categories = {
            ResponseCategory.ROLEPLAY_DETECTION,
            ResponseCategory.OUT_OF_SCOPE_DETECTION,
            ResponseCategory.UNCLEAR_INPUT_DETECTION,
            ResponseCategory.ERROR_FALLBACK,
            ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
            ResponseCategory.RASA_INTRODUCTION_DETECTION,
            ResponseCategory.COPILOT_INTRODUCTION_DETECTION,
        }
        if self.response_category not in valid_categories:
            raise ValueError(
                f"ControlledPredictionContent response_category must be one of "
                f"{[c.value for c in valid_categories]}, "
                f"got `{self.response_category}`."
            )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "ControlledPredictionContent":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
        )
        return self


class CopilotTextContent(GeneratedContent):
    """Content for Copilot text deltas."""

    content: str
    response_category: ResponseCategory = Field(
        default=ResponseCategory.COPILOT_TEXT_CONTENT_PART_DELTA, frozen=True
    )
    response_completeness: ResponseCompleteness = Field(
        default=ResponseCompleteness.TOKEN, frozen=True
    )

    @model_validator(mode="after")
    def validate_response_category(self) -> "CopilotTextContent":
        """Validate that response_category is COPILOT_TEXT_PART_DELTA."""
        _validate_response_category(
            ResponseCategory.COPILOT_TEXT_CONTENT_PART_DELTA,
            self.response_category,
            type(self),
        )
        return self


class CopilotTextStartContent(GeneratedContent):
    """Content for the start of a Copilot text part."""

    content: str = Field(default="")
    response_category: ResponseCategory = Field(
        default=ResponseCategory.COPILOT_TEXT_CONTENT_PART_START, frozen=True
    )
    response_completeness: ResponseCompleteness = Field(
        default=ResponseCompleteness.COMPLETE, frozen=True
    )

    @model_validator(mode="after")
    def validate_response_category(self) -> "CopilotTextStartContent":
        """Validate that response_category is COPILOT_TEXT_PART_START."""
        _validate_response_category(
            ResponseCategory.COPILOT_TEXT_CONTENT_PART_START,
            self.response_category,
            type(self),
        )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "CopilotTextStartContent":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
        )
        return self


class CopilotTextEndContent(GeneratedContent):
    """Content for the end of a Copilot text part."""

    content: str = Field(default="")
    response_category: ResponseCategory = Field(
        default=ResponseCategory.COPILOT_TEXT_CONTENT_PART_END, frozen=True
    )
    response_completeness: ResponseCompleteness = Field(
        default=ResponseCompleteness.COMPLETE, frozen=True
    )

    @model_validator(mode="after")
    def validate_response_category(self) -> "CopilotTextEndContent":
        """Validate that response_category is COPILOT_CONTENT_PART_END."""
        _validate_response_category(
            ResponseCategory.COPILOT_TEXT_CONTENT_PART_END,
            self.response_category,
            type(self),
        )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "CopilotTextEndContent":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
        )
        return self


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
        _validate_response_category(
            ResponseCategory.REFERENCE_ENTRY,
            self.response_category,
            type(self),
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
        _validate_response_category(
            ResponseCategory.REFERENCE,
            self.response_category,
            type(self),
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
        _validate_response_category(
            ResponseCategory.TRAINING_ERROR_LOG,
            self.response_category,
            type(self),
        )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "TrainingErrorLog":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
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


class E2ETestingErrorLog(CopilotOutput):
    """Represents an E2E testing error log."""

    logs: List[LogContent]
    response_category: ResponseCategory = Field(
        default=ResponseCategory.E2E_TESTING_ERROR_LOG,
        frozen=True,
    )
    response_completeness: ResponseCompleteness = ResponseCompleteness.COMPLETE

    @model_validator(mode="after")
    def validate_response_category(self) -> "E2ETestingErrorLog":
        """Validate that response_category has the correct default value."""
        _validate_response_category(
            ResponseCategory.E2E_TESTING_ERROR_LOG,
            self.response_category,
            type(self),
        )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "E2ETestingErrorLog":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
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


class MCPToolCall(CopilotOutput):
    """Represents MCP tool execution status."""

    tool_name: str
    status: MCPToolCallStatus
    response_category: ResponseCategory = Field(
        default=ResponseCategory.MCP_TOOL_CALL,
        frozen=True,
    )
    response_completeness: ResponseCompleteness = ResponseCompleteness.COMPLETE
    output: Optional[Any] = Field(default=None)

    @model_validator(mode="after")
    def validate_response_category(self) -> "MCPToolCall":
        """Validate that response_category has the correct default value."""
        if self.response_category != ResponseCategory.MCP_TOOL_CALL:
            raise ValueError(
                f"MCPToolCall response_category must be "
                f"{ResponseCategory.MCP_TOOL_CALL}, got `{self.response_category}`."
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
            "tool_name": self.tool_name,
            "status": self.status.value,
            "response_category": self.response_category.value,
            "completeness": self.response_completeness.value,
        }


class TodoItem(BaseModel):
    """Represents a single task in a plan."""

    id: str = Field(description="Unique identifier for the task")
    content: str = Field(description="Description of the task")
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current status of the task",
    )


class TaskPlan(BaseModel):
    """Input model for creating a task plan."""

    tasks: List[str] = Field(
        description=(
            "List of task descriptions. Each task should be a clear, "
            "actionable item that describes a step in completing the user's request."
        ),
    )


class TaskStatusUpdate(BaseModel):
    """Input model for updating a task's status."""

    task_id: str = Field(
        description="The ID of the task to update (e.g., '1', '2', '3')."
    )
    status: TaskStatus = Field(description="The new status of the task.")


class TodoPlanUpdate(CopilotOutput):
    """Represents a task plan update event for the frontend."""

    tasks: List[TodoItem] = Field(description="List of tasks in the plan")
    response_category: ResponseCategory = Field(
        default=ResponseCategory.TASK_PLANNING,
        frozen=True,
    )
    response_completeness: ResponseCompleteness = ResponseCompleteness.COMPLETE

    @model_validator(mode="after")
    def validate_response_category(self) -> "TodoPlanUpdate":
        """Validate that response_category has the correct default value."""
        if self.response_category != ResponseCategory.TASK_PLANNING:
            raise ValueError(
                f"TodoPlanUpdate response_category must be "
                f"{ResponseCategory.TASK_PLANNING}, got `{self.response_category}`."
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
            "tasks": [task.model_dump() for task in self.tasks],
            "response_category": self.response_category.value,
            "completeness": self.response_completeness.value,
        }


class CommitInformationContent(GeneratedContent):
    """Represents commit information content to be sent via SSE."""

    content: str = Field(default="")
    commit: Dict[str, Any] = Field(
        ...,
        description="The commit information dictionary.",
    )
    response_category: ResponseCategory = Field(
        default=ResponseCategory.COMMIT,
        frozen=True,
    )
    response_completeness: ResponseCompleteness = Field(
        default=ResponseCompleteness.COMPLETE,
        frozen=True,
    )

    @model_validator(mode="after")
    def validate_response_category(self) -> "CommitInformationContent":
        """Validate that response_category is COMMIT."""
        _validate_response_category(
            ResponseCategory.COMMIT,
            self.response_category,
            type(self),
        )
        return self

    @model_validator(mode="after")
    def validate_response_completeness(self) -> "CommitInformationContent":
        """Validate that response_completeness is COMPLETE."""
        _validate_response_completeness(
            ResponseCompleteness.COMPLETE,
            self.response_completeness,
            type(self),
        )
        return self

    @property
    def sse_data(self) -> Dict[str, Any]:
        """Extract the SSE data payload including commit information."""
        base_data = super().sse_data
        base_data["commit"] = self.commit
        return base_data


class UsageStatistics(BaseModel):
    """Usage statistics for a copilot generation."""

    # Token usage statistics
    prompt_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Total number of prompt tokens used to generate completion. "
            "Should include cached prompt tokens."
        ),
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        description="Number of generated tokens.",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total number of tokens used (input + output).",
    )
    cached_prompt_tokens: Optional[int] = Field(
        default=None,
        description="Number of cached prompt tokens.",
    )
    model: Optional[str] = Field(
        default=None,
        description="The model used to generate the response.",
    )

    # Token prices
    input_token_price: float = Field(
        default=0.0,
        description="Price per 1K input tokens in dollars.",
    )
    output_token_price: float = Field(
        default=0.0,
        description="Price per 1K output tokens in dollars.",
    )
    cached_token_price: float = Field(
        default=0.0,
        description="Price per 1K cached tokens in dollars.",
    )

    @property
    def non_cached_prompt_tokens(self) -> Optional[int]:
        """Get the non-cached prompt tokens."""
        if self.cached_prompt_tokens is not None and self.prompt_tokens is not None:
            return self.prompt_tokens - self.cached_prompt_tokens
        return self.prompt_tokens

    @property
    def non_cached_cost(self) -> Optional[float]:
        """Calculate the non-cached token cost based on configured pricing."""
        if self.non_cached_prompt_tokens is None:
            return None
        if self.non_cached_prompt_tokens == 0:
            return 0.0

        return (self.non_cached_prompt_tokens / 1000.0) * self.input_token_price

    @property
    def cached_cost(self) -> Optional[float]:
        """Calculate the cached token cost based on configured pricing."""
        if self.cached_prompt_tokens is None:
            return None
        if self.cached_prompt_tokens == 0:
            return 0.0

        return (self.cached_prompt_tokens / 1000.0) * self.cached_token_price

    @property
    def input_cost(self) -> Optional[float]:
        """Calculate the input token cost based on configured pricing.

        The calculation takes into account the cached prompt tokens (if available) too.
        """
        # If both non-cached and cached costs are None, there's no input cost
        if self.non_cached_cost is None and self.cached_cost is None:
            return None

        # If only non-cached cost is available, return it
        if self.non_cached_cost is not None and self.cached_cost is None:
            return self.non_cached_cost

        # If only cached cost is available, return it
        if self.non_cached_cost is None and self.cached_cost is not None:
            return self.cached_cost

        # If both are available, return the sum
        return self.non_cached_cost + self.cached_cost  # type: ignore[operator]

    @property
    def output_cost(self) -> Optional[float]:
        """Calculate the output token cost based on configured pricing."""
        if self.completion_tokens is None:
            return None
        if self.completion_tokens == 0:
            return 0.0

        return (self.completion_tokens / 1000.0) * self.output_token_price

    @property
    def total_cost(self) -> Optional[float]:
        """Calculate the total cost based on configured pricing.

        Returns:
            Total cost in dollars, or None if insufficient data.
        """
        if self.input_cost is None or self.output_cost is None:
            return None

        return self.input_cost + self.output_cost

    def update_token_prices(
        self,
        input_token_price: float,
        output_token_price: float,
        cached_token_price: float,
    ) -> None:
        """Update token prices with provided values.

        Args:
            input_token_price: Price per 1K input tokens in dollars.
            output_token_price: Price per 1K output tokens in dollars.
            cached_token_price: Price per 1K cached tokens in dollars.
        """
        self.input_token_price = input_token_price
        self.output_token_price = output_token_price
        self.cached_token_price = cached_token_price

    @classmethod
    def from_chat_completion_response(
        cls,
        response: ChatCompletion,
        input_token_price: float = 0.0,
        output_token_price: float = 0.0,
        cached_token_price: float = 0.0,
    ) -> Optional["UsageStatistics"]:
        """Create a UsageStatistics object from a ChatCompletionChunk."""
        if not (usage := getattr(response, "usage", None)):
            return None

        usage_statistics = cls(
            input_token_price=input_token_price,
            output_token_price=output_token_price,
            cached_token_price=cached_token_price,
        )

        usage_statistics.prompt_tokens = usage.prompt_tokens
        usage_statistics.completion_tokens = usage.completion_tokens
        usage_statistics.total_tokens = usage.total_tokens
        usage_statistics.model = getattr(response, "model", None)

        # Extract cached tokens if available
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            usage_statistics.cached_prompt_tokens = getattr(
                usage.prompt_tokens_details, "cached_tokens", None
            )

        return usage_statistics

    def reset(self) -> None:
        """Reset usage statistics to their default values."""
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.cached_prompt_tokens = None
        self.model = None

    def update_from_stream_chunk(self, chunk: ChatCompletionChunk) -> None:
        """Update usage statistics from an OpenAI stream chunk.

        Args:
            chunk: The OpenAI stream chunk containing usage statistics.
        """
        # Reset the usage statistics to their default values
        self.reset()

        # If the chunk has no usage statistics, return
        if not (usage := getattr(chunk, "usage", None)):
            return

        # Update the usage statistics with the values from the chunk
        self.prompt_tokens = usage.prompt_tokens
        self.completion_tokens = usage.completion_tokens
        self.total_tokens = usage.total_tokens
        self.model = getattr(chunk, "model", None)

        # Extract cached tokens if available
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            self.cached_prompt_tokens = getattr(
                usage.prompt_tokens_details, "cached_tokens", None
            )

    def update_from_response_completed_event(
        self, event: ResponseCompletedEvent
    ) -> None:
        """Update usage statistics from a ResponseCompletedEvent.

        This method checks if the event is a ResponseCompletedEvent and, if so,
        extracts the usage statistics from the response and updates this
        UsageStatistics object.

        Args:
            event: The stream event to extract usage statistics from.
        """
        self.reset()

        if not event.response.usage:
            return

        # Convert model to string if needed
        model_str = str(event.response.model) if event.response.model else None
        if model_str:
            self.model = model_str

        # Update the usage statistics with the values from ResponseUsage
        self.prompt_tokens = event.response.usage.input_tokens
        self.completion_tokens = event.response.usage.output_tokens
        self.total_tokens = event.response.usage.total_tokens

        # Extract cached tokens from input_tokens_details
        self.cached_prompt_tokens = (
            event.response.usage.input_tokens_details.cached_tokens
        )

    def __add__(self, other: "UsageStatistics") -> "UsageStatistics":
        """Add two UsageStatistics objects together.

        Args:
            other: Another UsageStatistics object to add.

        Returns:
            A new UsageStatistics object with aggregated values.
        """
        # NOTE: Model field preference is arbitrary when aggregating
        # different models. We prefer 'other' to report the dominant model
        # in typical usage patterns. Individual model usage is tracked
        # separately in Langfuse observations.
        return UsageStatistics(
            model=other.model or self.model,
            prompt_tokens=(self.prompt_tokens or 0) + (other.prompt_tokens or 0),
            completion_tokens=(self.completion_tokens or 0)
            + (other.completion_tokens or 0),
            total_tokens=(self.total_tokens or 0) + (other.total_tokens or 0),
            cached_prompt_tokens=(self.cached_prompt_tokens or 0)
            + (other.cached_prompt_tokens or 0),
            input_token_price=other.input_token_price or self.input_token_price,
            output_token_price=other.output_token_price or self.output_token_price,
            cached_token_price=other.cached_token_price or self.cached_token_price,
        )


class CopilotGenerationContext(BaseModel):
    """Container for copilot generation context and supporting evidence.

    This class organizes the context and supporting evidence information used by the
    copilot's generate_response method, providing a cleaner interface than returning
    a tuple for the non-streaming data.
    """

    # TODO: (agent-sdk) remove once LegacyCopilot is removed
    # this is not needed by the agent copilot, as that will retrieve docs using
    # a tool call.
    relevant_documents: List["Document"] = Field(
        description=(
            "The relevant documents used as supporting evidence for the respons."
        ),
        default_factory=list,
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


def _validate_response_category(
    expected_response_category: ResponseCategory,
    actual_response_category: ResponseCategory,
    pydantic_class: Type[Any],
) -> None:
    """Validate that the response category is valid for the given pydantic class.

    Args:
        expected_response_category: The expected ResponseCategory value.
        actual_response_category: The actual ResponseCategory value to validate.
        pydantic_class: The Pydantic class type (used for error messages).

    Raises:
        ValueError: If the actual category doesn't match the expected value.
    """
    if expected_response_category != actual_response_category:
        error_message = (
            f"{pydantic_class.__name__} response_category must be "
            f"{expected_response_category}, got `{actual_response_category}`."
        )
        structlogger.error(
            f"{pydantic_class.__name__}.validate_response_category",
            event_info=error_message,
        )
        raise ValueError(error_message)


def _validate_response_completeness(
    expected_response_completeness: ResponseCompleteness,
    actual_response_completeness: ResponseCompleteness,
    pydantic_class: Type[Any],
) -> None:
    """Validate that the response completeness is valid for the given pydantic class.

    Args:
        expected_response_completeness: The expected ResponseCompleteness value.
        actual_response_completeness: The actual ResponseCompleteness value to validate.
        pydantic_class: The Pydantic class type (used for error messages).

    Raises:
        ValueError: If the actual completeness doesn't match the expected value.
    """
    if expected_response_completeness != actual_response_completeness:
        error_message = (
            f"{pydantic_class.__name__} response_completeness must be "
            f"{expected_response_completeness}, got `{actual_response_completeness}`."
        )
        structlogger.error(
            f"{pydantic_class.__name__}.validate_response_completeness",
            event_info=error_message,
            expected_response_completeness=expected_response_completeness,
            actual_response_completeness=actual_response_completeness,
        )
        raise ValueError(error_message)
