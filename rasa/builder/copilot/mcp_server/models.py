"""Pydantic models for MCP server input/output validation."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, validator

# Common field description constants
ERROR_FIELD_DESCRIPTION = "Error message if failed"
SUCCESS_FIELD_DESCRIPTION = "Whether the operation succeeded"


class FileUpdate(BaseModel):
    """Model for a single file update operation."""

    path: str = Field(
        ...,
        description="Relative path to the file within the project",
        min_length=1,
        max_length=255,
    )
    content: str = Field(..., description="Complete content to write to the file")

    @validator("path")
    def validate_path(cls, v: str) -> str:
        """Validate file path for security concerns."""
        # Prevent path traversal
        if ".." in v:
            raise ValueError("Path traversal is not allowed (contains '..')")

        # Prevent absolute paths
        if v.startswith("/"):
            raise ValueError("Absolute paths are not allowed")

        # Prevent hidden files
        if v.startswith(".") or "/." in v:
            raise ValueError("Cannot write to hidden files or directories")

        # Basic character validation
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./"
        )
        if not all(c in allowed_chars for c in v):
            raise ValueError("Path contains invalid characters")

        return v


class MultiFileUpdate(BaseModel):
    """Model for updating multiple files at once."""

    files: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping file paths to their new contents",
    )

    @validator("files")
    def validate_files(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate all file paths in the batch update."""
        # Check minimum and maximum number of files
        if len(v) < 1:
            raise ValueError("At least one file must be provided")
        if len(v) > 50:
            raise ValueError("Cannot update more than 50 files at once")

        # Validate each file path
        for path in v.keys():
            # Reuse the path validation from FileUpdate
            FileUpdate(path=path, content=v[path])
        return v


class SearchQuery(BaseModel):
    """Model for documentation search queries."""

    query: str = Field(
        ...,
        description="The search query to find relevant documentation",
        min_length=2,
        max_length=500,
    )

    @validator("query")
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize search query."""
        # Trim whitespace
        v = v.strip()

        if len(v) < 2:
            raise ValueError("Query must be at least 2 characters long")

        return v


class FilePathInput(BaseModel):
    """Model for file path inputs."""

    file_path: str = Field(
        ...,
        description="Relative path to the file within the project",
        min_length=1,
        max_length=255,
    )

    @validator("file_path")
    def validate_file_path(cls, v: str) -> str:
        """Validate file path."""
        # Prevent path traversal
        if ".." in v:
            raise ValueError("Path traversal is not allowed")

        # Prevent absolute paths
        if v.startswith("/"):
            raise ValueError("Absolute paths are not allowed")

        return v


# =============================================================================
# OUTPUT MODELS - Structured outputs for MCP tools
# =============================================================================


class DocumentSearchResult(BaseModel):
    """A single document from the search results."""

    index: int = Field(description="Position in the search results")
    title: str = Field(description="Document title")
    url: str = Field(description="Document URL")
    content: str = Field(description="Document content snippet")


class DocumentSearchResponse(BaseModel):
    """Response from documentation search."""

    documents: List[DocumentSearchResult] = Field(
        default_factory=list, description="List of matching documents"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if search failed"
    )


class FileListResponse(BaseModel):
    """Response from listing project files."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    tree: Optional[str] = Field(default=None, description="Visual tree representation")
    files: List[str] = Field(default_factory=list, description="List of file paths")
    count: int = Field(default=0, description="Number of files")
    directories: List[str] = Field(
        default_factory=list, description="List of directories"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)


class FileContentResponse(BaseModel):
    """Response from reading a single file."""

    file_path: Optional[str] = Field(default=None, description="Path to the file")
    content: Optional[str] = Field(default=None, description="File content")
    exists: bool = Field(default=True, description="Whether the file exists")
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)


class ReadFilesResponse(BaseModel):
    """Response from reading multiple project files."""

    files: Dict[str, Optional[str]] = Field(
        default_factory=dict, description="Map of file paths to contents"
    )
    count: int = Field(default=0, description="Number of files read")
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)


class WriteFileResponse(BaseModel):
    """Response from writing a file."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    file_path: str = Field(description="Path to the file")
    message: str = Field(description="Status message")


class FileUpdateFailure(BaseModel):
    """Details about a failed file update."""

    file_path: Optional[str] = Field(default=None, description="Path that failed")
    error: str = Field(description="Error message")


class UpdateFilesResponse(BaseModel):
    """Response from updating multiple files."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    updated: List[str] = Field(default_factory=list, description="Files updated")
    failed: List[FileUpdateFailure] = Field(
        default_factory=list, description="Files that failed"
    )
    message: str = Field(description="Status message")


class ValidationErrorDetail(BaseModel):
    """A single validation error."""

    level: Optional[str] = Field(default="error", description="Error severity")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )


class ValidationResponse(BaseModel):
    """Response from project validation."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    errors: Optional[List[ValidationErrorDetail]] = Field(
        default=None, description="List of validation errors"
    )
    message: str = Field(description="Validation status message")


class TrainingResponse(BaseModel):
    """Response from model training."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    model_path: Optional[str] = Field(default=None, description="Path to trained model")
    message: str = Field(description="Training status message")
    agent_reloaded: Optional[bool] = Field(
        default=None, description="Whether the agent was reloaded in the server"
    )


class BotResponse(BaseModel):
    """A single response from the bot."""

    text: Optional[str] = Field(default=None, description="Response text")
    image: Optional[str] = Field(default=None, description="Image URL")
    buttons: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Button options"
    )
    custom: Optional[Dict[str, Any]] = Field(default=None, description="Custom payload")


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    user_message: str = Field(description="The user's message")
    bot_responses: List[Dict[str, Any]] = Field(
        default_factory=list, description="Bot responses to the message"
    )


class TrackerContextOutput(BaseModel):
    """Tracker context from the conversation."""

    conversation_turns: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conversation history"
    )
    current_state: Dict[str, Any] = Field(
        default_factory=dict, description="Current conversation state"
    )


class TalkToAssistantResponse(BaseModel):
    """Response from talking to the assistant."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    session_id: str = Field(description="Unique session ID for this conversation")
    message_count: int = Field(description="Number of messages sent")
    conversation: List[ConversationTurn] = Field(
        default_factory=list, description="The conversation history"
    )
    tracker_context: Optional[TrackerContextOutput] = Field(
        default=None, description="Full tracker context after conversation"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)


class CustomActionInfo(BaseModel):
    """Information about a custom action defined in the domain."""

    name: str = Field(description="The action name")
    file_path: Optional[str] = Field(
        default=None, description="Path to the file containing the action definition"
    )


class CustomActionImplementationInfo(BaseModel):
    """Information about a custom action implementation (Python class)."""

    name: Optional[str] = Field(
        default=None,
        description="The action name (from the name() method)",
    )
    class_name: str = Field(
        description="The Python class name of the action",
    )
    file_path: str = Field(
        description=(
            "Path to the Python file containing the action, relative to project root"
        ),
    )


class CustomActionsResponse(BaseModel):
    """Response from listing custom action implementations."""

    actions: List[CustomActionImplementationInfo] = Field(
        default_factory=list, description="List of custom action implementations found"
    )
    actions_folder: str = Field(description="The actions folder that was scanned")
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of actions found."""
        return len(self.actions)


# =============================================================================
# PROJECT CONTEXT MODELS - Structured outputs for project context tools
# =============================================================================


class FlowInfo(BaseModel):
    """Information about a single flow in the project."""

    id: str = Field(description="The unique identifier of the flow")
    name: Optional[str] = Field(default=None, description="Human-readable name of flow")
    file_path: Optional[str] = Field(
        default=None, description="Path to the file containing the flow"
    )
    definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Full flow definition (name, description, steps, etc.) "
            "when returned by get_flow"
        ),
    )


class ListFlowsResponse(BaseModel):
    """Response from listing project flows."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    flows: List[FlowInfo] = Field(
        default_factory=list, description="List of flows in the project"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of flows."""
        return len(self.flows)


class SlotInfo(BaseModel):
    """Information about a single slot in the domain."""

    name: str = Field(description="The name of the slot")
    type: str = Field(description="The slot type (text, bool, categorical, etc.)")
    file_path: Optional[str] = Field(
        default=None, description="Path to the file containing the slot definition"
    )
    definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Full slot definition (type, mappings, etc.) when returned by get_slot"
        ),
    )


class ListSlotsResponse(BaseModel):
    """Response from listing project slots."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    slots: List[SlotInfo] = Field(
        default_factory=list, description="List of slots in the domain"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of slots."""
        return len(self.slots)


class ResponseInfo(BaseModel):
    """Information about a single response in the domain."""

    name: str = Field(description="The response identifier (e.g., utter_greet)")
    file_path: Optional[str] = Field(
        default=None, description="Path to the file containing the response definition"
    )
    definition: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Full response definition (list of variants) when returned by get_response"
        ),
    )


class ListResponsesResponse(BaseModel):
    """Response from listing project responses."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    responses: List[ResponseInfo] = Field(
        default_factory=list, description="List of responses in the domain"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of responses."""
        return len(self.responses)


class GetFlowResponse(BaseModel):
    """Response from get_flow tool."""

    success: bool = Field(description="Whether the flow was found")
    flow: Optional[FlowInfo] = Field(
        default=None, description="Flow metadata and definition when found"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)


class GetSlotResponse(BaseModel):
    """Response from get_slot tool."""

    success: bool = Field(description="Whether the slot was found")
    slot: Optional[SlotInfo] = Field(
        default=None, description="Slot metadata and definition when found"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)


class GetResponseResponse(BaseModel):
    """Response from get_response tool."""

    success: bool = Field(description="Whether the response was found")
    response: Optional[ResponseInfo] = Field(
        default=None, description="Response metadata and definition when found"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)


class ListCustomActionsResponse(BaseModel):
    """Response from listing custom actions in the domain."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    actions: List[CustomActionInfo] = Field(
        default_factory=list, description="List of custom actions in the domain"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of custom actions."""
        return len(self.actions)


class ListDefaultActionsResponse(BaseModel):
    """Response from listing default/built-in action names."""

    success: bool = Field(description=SUCCESS_FIELD_DESCRIPTION)
    actions: List[str] = Field(
        default_factory=list,
        description="List of default action names provided by Rasa",
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of default actions."""
        return len(self.actions)


# =============================================================================
# SCHEMA TOOLS - Responses for get_flow_schema, get_domain_schema, get_e2e_schema
# =============================================================================


class SchemaType(str, Enum):
    """Schema type returned by schema tools."""

    FLOW = "flow"
    DOMAIN = "domain"
    E2E = "e2e"


class SchemaResponse(BaseModel):
    """Response from get_flow_schema, get_domain_schema, or get_e2e_schema."""

    success: bool = Field(description="Whether the schema was loaded successfully")
    schema_type: SchemaType = Field(description="Type of schema: flow, domain, or e2e")
    schema_content: str = Field(
        description="Schema as JSON string for validation or code generation"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)
