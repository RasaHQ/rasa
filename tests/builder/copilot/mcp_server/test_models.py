"""Tests for MCP server Pydantic models and validators."""

import pytest
from pydantic import ValidationError

from rasa.builder.copilot.mcp_server.models import (
    DocumentSearchResponse,
    DocumentSearchResult,
    FileContentResponse,
    FileListResponse,
    FilePathInput,
    FileUpdate,
    MultiFileUpdate,
    ReadFilesResponse,
    SearchQuery,
    TrainingResponse,
    UpdateFilesResponse,
    ValidationErrorDetail,
    ValidationResponse,
    WriteFileResponse,
)


class TestFileUpdateValidator:
    """Test FileUpdate model path validation."""

    def test_valid_simple_path(self) -> None:
        """Test that simple valid paths are accepted."""
        update = FileUpdate(path="domain.yml", content="version: '3.1'")
        assert update.path == "domain.yml"

    def test_valid_nested_path(self) -> None:
        """Test that nested paths are accepted."""
        update = FileUpdate(path="data/nlu/training.yml", content="nlu: []")
        assert update.path == "data/nlu/training.yml"

    def test_path_with_hyphen_and_underscore(self) -> None:
        """Test that paths with hyphens and underscores are accepted."""
        update = FileUpdate(path="my-file_name.yml", content="content")
        assert update.path == "my-file_name.yml"

    def test_path_traversal_rejected(self) -> None:
        """Test that path traversal is rejected."""
        with pytest.raises(ValidationError, match="Path traversal"):
            FileUpdate(path="../outside.yml", content="content")

    def test_path_traversal_in_middle_rejected(self) -> None:
        """Test that path traversal in the middle of path is rejected."""
        with pytest.raises(ValidationError, match="Path traversal"):
            FileUpdate(path="data/../config.yml", content="content")

    def test_absolute_path_rejected(self) -> None:
        """Test that absolute paths are rejected."""
        with pytest.raises(ValidationError, match="Absolute paths"):
            FileUpdate(path="/etc/passwd", content="content")

    def test_hidden_file_rejected(self) -> None:
        """Test that hidden files are rejected."""
        with pytest.raises(ValidationError, match="hidden"):
            FileUpdate(path=".hidden", content="content")

    def test_hidden_directory_rejected(self) -> None:
        """Test that paths with hidden directories are rejected."""
        with pytest.raises(ValidationError, match="hidden"):
            FileUpdate(path="data/.git/config", content="content")

    def test_invalid_characters_rejected(self) -> None:
        """Test that invalid characters are rejected."""
        with pytest.raises(ValidationError, match="invalid characters"):
            FileUpdate(path="file name.yml", content="content")  # Space not allowed

    def test_special_characters_rejected(self) -> None:
        """Test that special characters are rejected."""
        invalid_paths = [
            "file@name.yml",
            "file#name.yml",
            "file$name.yml",
            "file%name.yml",
            "file&name.yml",
            "file*name.yml",
        ]
        for path in invalid_paths:
            with pytest.raises(ValidationError, match="invalid characters"):
                FileUpdate(path=path, content="content")


class TestMultiFileUpdateValidator:
    """Test MultiFileUpdate model validation."""

    def test_valid_single_file(self) -> None:
        """Test that a single file update is valid."""
        update = MultiFileUpdate(files={"domain.yml": "version: '3.1'"})
        assert len(update.files) == 1

    def test_valid_multiple_files(self) -> None:
        """Test that multiple file updates are valid."""
        files = {
            "domain.yml": "version: '3.1'",
            "config.yml": "pipeline: []",
            "data/nlu.yml": "nlu: []",
        }
        update = MultiFileUpdate(files=files)
        assert len(update.files) == 3

    def test_empty_files_rejected(self) -> None:
        """Test that empty files dict is rejected."""
        with pytest.raises(ValidationError, match="At least one file"):
            MultiFileUpdate(files={})

    def test_too_many_files_rejected(self) -> None:
        """Test that more than 50 files is rejected."""
        # Create 51 files
        files = {f"file{i}.yml": "content" for i in range(51)}
        with pytest.raises(ValidationError, match="Cannot update more than 50"):
            MultiFileUpdate(files=files)

    def test_invalid_path_in_batch_rejected(self) -> None:
        """Test that invalid paths in batch are rejected."""
        files = {
            "valid.yml": "content",
            "../invalid.yml": "content",  # Path traversal
        }
        with pytest.raises(ValidationError, match="Path traversal"):
            MultiFileUpdate(files=files)


class TestSearchQueryValidator:
    """Test SearchQuery model validation."""

    def test_valid_query(self) -> None:
        """Test that valid queries are accepted."""
        query = SearchQuery(query="How do I create a flow?")
        assert query.query == "How do I create a flow?"

    def test_query_trimmed(self) -> None:
        """Test that query whitespace is trimmed."""
        query = SearchQuery(query="  How do I create a flow?  ")
        assert query.query == "How do I create a flow?"

    def test_short_query_rejected(self) -> None:
        """Test that queries shorter than 2 chars are rejected."""
        with pytest.raises(ValidationError, match="at least 2"):
            SearchQuery(query="a")

    def test_whitespace_only_query_rejected(self) -> None:
        """Test that whitespace-only queries are rejected after trimming."""
        with pytest.raises(ValidationError, match="at least 2"):
            SearchQuery(query="   ")


class TestFilePathInputValidator:
    """Test FilePathInput model validation."""

    def test_valid_path(self) -> None:
        """Test that valid file paths are accepted."""
        input_model = FilePathInput(file_path="data/nlu.yml")
        assert input_model.file_path == "data/nlu.yml"

    def test_path_traversal_rejected(self) -> None:
        """Test that path traversal is rejected."""
        with pytest.raises(ValidationError, match="Path traversal"):
            FilePathInput(file_path="../outside.yml")

    def test_absolute_path_rejected(self) -> None:
        """Test that absolute paths are rejected."""
        with pytest.raises(ValidationError, match="Absolute paths"):
            FilePathInput(file_path="/etc/passwd")


class TestDocumentSearchModels:
    """Test DocumentSearch related models."""

    def test_document_search_result_creation(self) -> None:
        """Test DocumentSearchResult model creation."""
        result = DocumentSearchResult(
            index=1,
            title="Test Document",
            url="https://rasa.com/docs",
            content="Sample content",
        )
        assert result.index == 1
        assert result.title == "Test Document"
        assert result.url == "https://rasa.com/docs"
        assert result.content == "Sample content"

    def test_document_search_response_empty(self) -> None:
        """Test DocumentSearchResponse with no documents."""
        response = DocumentSearchResponse()
        assert response.documents == []
        assert response.error is None

    def test_document_search_response_with_documents(self) -> None:
        """Test DocumentSearchResponse with documents."""
        docs = [
            DocumentSearchResult(
                index=1, title="Doc 1", url="url1", content="content1"
            ),
            DocumentSearchResult(
                index=2, title="Doc 2", url="url2", content="content2"
            ),
        ]
        response = DocumentSearchResponse(documents=docs)
        assert len(response.documents) == 2

    def test_document_search_response_with_error(self) -> None:
        """Test DocumentSearchResponse with error."""
        response = DocumentSearchResponse(error="Search failed")
        assert response.documents == []
        assert response.error == "Search failed"


class TestFileOperationResponseModels:
    """Test file operation response models."""

    def test_file_list_response(self) -> None:
        """Test FileListResponse model."""
        response = FileListResponse(
            success=True,
            tree=".\n├── domain.yml",
            files=["domain.yml", "config.yml"],
            count=2,
            directories=["data"],
        )
        assert response.success is True
        assert response.count == 2
        assert "domain.yml" in response.files

    def test_file_content_response_exists(self) -> None:
        """Test FileContentResponse when file exists."""
        response = FileContentResponse(
            file_path="domain.yml",
            content="version: '3.1'",
            exists=True,
        )
        assert response.exists is True
        assert response.content == "version: '3.1'"

    def test_file_content_response_not_exists(self) -> None:
        """Test FileContentResponse when file does not exist."""
        response = FileContentResponse(
            file_path="missing.yml",
            content=None,
            exists=False,
            error="File not found",
        )
        assert response.exists is False
        assert response.content is None
        assert response.error == "File not found"

    def test_read_files_response(self) -> None:
        """Test ReadFilesResponse model."""
        response = ReadFilesResponse(
            files={"domain.yml": "content", "config.yml": "pipeline: []"},
            count=2,
        )
        assert response.count == 2
        assert "domain.yml" in response.files

    def test_write_file_response(self) -> None:
        """Test WriteFileResponse model."""
        response = WriteFileResponse(
            success=True,
            file_path="domain.yml",
            message="File written successfully",
        )
        assert response.success is True

    def test_update_files_response(self) -> None:
        """Test UpdateFilesResponse model."""
        response = UpdateFilesResponse(
            success=True,
            updated=["domain.yml", "config.yml"],
            failed=[],
            message="Updated 2 files",
        )
        assert response.success is True
        assert len(response.updated) == 2


class TestValidationAndTrainingResponseModels:
    """Test validation and training response models."""

    def test_validation_error_detail(self) -> None:
        """Test ValidationErrorDetail model."""
        error = ValidationErrorDetail(
            level="error",
            message="Missing intent definition",
            details={"line": 10, "file": "nlu.yml"},
        )
        assert error.level == "error"
        assert error.message == "Missing intent definition"
        assert error.details["line"] == 10

    def test_validation_response_success(self) -> None:
        """Test ValidationResponse for successful validation."""
        response = ValidationResponse(
            success=True,
            errors=None,
            message="Validation passed",
        )
        assert response.success is True
        assert response.errors is None

    def test_validation_response_failure(self) -> None:
        """Test ValidationResponse for failed validation."""
        errors = [
            ValidationErrorDetail(message="Error 1"),
            ValidationErrorDetail(message="Error 2"),
        ]
        response = ValidationResponse(
            success=False,
            errors=errors,
            message="Validation failed",
        )
        assert response.success is False
        assert len(response.errors) == 2

    def test_training_response_success(self) -> None:
        """Test TrainingResponse for successful training."""
        response = TrainingResponse(
            success=True,
            model_path="/path/to/model.tar.gz",
            message="Training completed",
            agent_reloaded=True,
        )
        assert response.success is True
        assert response.model_path is not None
        assert response.agent_reloaded is True

    def test_training_response_failure(self) -> None:
        """Test TrainingResponse for failed training."""
        response = TrainingResponse(
            success=False,
            model_path=None,
            message="Training failed: Invalid config",
        )
        assert response.success is False
        assert response.model_path is None
