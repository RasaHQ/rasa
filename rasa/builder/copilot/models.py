from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import structlog
from openai.types.chat import ChatCompletion
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


class ReferenceItem(BaseModel):
    index: int
    title: str
    url: str


class ReferencesContent(BaseContent):
    type: Literal["references"]
    references: List[ReferenceItem]


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
    ],
    Field(
        discriminator="type",
        description=(
            "The content of the message. "
            "The content is expected to be a list of content blocks. "
            "The content blocks are expected to be one of the following types: "
            "text, link, code, file, references, logs, or event."
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
