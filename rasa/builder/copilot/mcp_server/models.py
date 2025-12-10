"""Pydantic models for MCP server input/output validation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

# Common field description constant
ERROR_FIELD_DESCRIPTION = "Error message if failed"


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

    success: bool = Field(description="Whether the operation succeeded")
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

    success: bool = Field(description="Whether the write succeeded")
    file_path: str = Field(description="Path to the file")
    message: str = Field(description="Status message")


class FileUpdateFailure(BaseModel):
    """Details about a failed file update."""

    file_path: Optional[str] = Field(default=None, description="Path that failed")
    error: str = Field(description="Error message")


class UpdateFilesResponse(BaseModel):
    """Response from updating multiple files."""

    success: bool = Field(description="Whether all updates succeeded")
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

    success: bool = Field(description="Whether validation passed")
    errors: Optional[List[ValidationErrorDetail]] = Field(
        default=None, description="List of validation errors"
    )
    message: str = Field(description="Validation status message")


class TrainingResponse(BaseModel):
    """Response from model training."""

    success: bool = Field(description="Whether training succeeded")
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

    success: bool = Field(description="Whether the conversation succeeded")
    session_id: str = Field(description="Unique session ID for this conversation")
    message_count: int = Field(description="Number of messages sent")
    conversation: List[ConversationTurn] = Field(
        default_factory=list, description="The conversation history"
    )
    tracker_context: Optional[TrackerContextOutput] = Field(
        default=None, description="Full tracker context after conversation"
    )
    error: Optional[str] = Field(default=None, description=ERROR_FIELD_DESCRIPTION)
