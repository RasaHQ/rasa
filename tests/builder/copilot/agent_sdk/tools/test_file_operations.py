"""Tests for file operation supplementary tools."""

from pathlib import Path

import pytest
from agents.tool_context import ToolContext

from rasa.builder.copilot.agent_sdk.tools.file_operations import (
    get_file_content,
    get_project_file,
    list_files,
    list_project_files,
    read_assistant_files,
    read_project_files,
    update_files,
    update_multiple_files,
    write_file,
    write_project_file,
)
from rasa.builder.copilot.mcp_server.models import (
    FileContentResponse,
    FileListResponse,
    ReadFilesResponse,
    UpdateFilesResponse,
    WriteFileResponse,
)


def _make_tool_context(tool_name: str = "test_tool") -> ToolContext:
    """Create a minimal ToolContext for testing on_invoke_tool calls."""
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id="test_call_id",
        tool_arguments="{}",
    )


class TestListFiles:
    """Test list_files function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with test files."""
        # Create some files
        (tmp_path / "domain.yml").write_text("version: '3.1'")
        (tmp_path / "config.yml").write_text("pipeline: []")

        # Create a subdirectory with files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "nlu.yml").write_text("nlu: []")
        (data_dir / "stories.yml").write_text("stories: []")

        return tmp_path

    @pytest.mark.asyncio
    async def test_list_files_success(self, project_folder: Path):
        """Test successful file listing."""
        result = await list_files(str(project_folder))

        assert isinstance(result, FileListResponse)
        assert result.success is True
        assert result.count >= 4  # At least our created files
        assert "domain.yml" in result.files
        assert "config.yml" in result.files
        assert "data/nlu.yml" in result.files or "data\\nlu.yml" in result.files
        assert "data" in result.directories

    @pytest.mark.asyncio
    async def test_list_files_includes_tree_structure(self, project_folder: Path):
        """Test that list_files includes a tree structure."""
        result = await list_files(str(project_folder))

        assert result.tree is not None
        assert "Project Structure" in result.tree
        assert "domain.yml" in result.tree

    @pytest.mark.asyncio
    async def test_list_files_excludes_hidden_files(self, project_folder: Path):
        """Test that hidden files are excluded."""
        # Create a hidden file
        (project_folder / ".hidden").write_text("hidden content")
        hidden_dir = project_folder / ".git"
        hidden_dir.mkdir()
        (hidden_dir / "config").write_text("git config")

        result = await list_files(str(project_folder))

        assert ".hidden" not in result.files
        assert ".git/config" not in result.files

    @pytest.mark.asyncio
    async def test_list_files_excludes_pycache(self, project_folder: Path):
        """Test that __pycache__ is excluded."""
        pycache_dir = project_folder / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "module.pyc").write_text("bytecode")

        result = await list_files(str(project_folder))

        assert "__pycache__" not in result.directories
        assert any("__pycache__" in f for f in result.files) is False

    @pytest.mark.asyncio
    async def test_list_files_nonexistent_folder(self, tmp_path: Path):
        """Test listing files in nonexistent folder returns empty results."""
        result = await list_files(str(tmp_path / "nonexistent"))

        # Non-existent folder returns empty results (no exception
        # raised by path iteration)
        assert result.count == 0
        assert result.files == []


class TestReadAssistantFiles:
    """Test read_assistant_files function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with test files."""
        (tmp_path / "domain.yml").write_text("version: '3.1'")
        (tmp_path / "config.yml").write_text("pipeline: []")
        (tmp_path / "readme.txt").write_text("readme content")

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide")

        return tmp_path

    @pytest.mark.asyncio
    async def test_read_assistant_files_all(self, project_folder: Path):
        """Test reading all files."""
        result = await read_assistant_files(str(project_folder), exclude_docs=False)

        assert isinstance(result, ReadFilesResponse)
        assert result.error is None
        assert "domain.yml" in result.files
        assert result.files["domain.yml"] == "version: '3.1'"

    @pytest.mark.asyncio
    async def test_read_assistant_files_excludes_docs(self, project_folder: Path):
        """Test that docs directory is excluded by default."""
        result = await read_assistant_files(str(project_folder), exclude_docs=True)

        assert result.error is None
        # Docs should be excluded
        assert any("docs/" in key for key in result.files.keys()) is False

    @pytest.mark.asyncio
    async def test_read_assistant_files_with_extension_filter(
        self, project_folder: Path
    ):
        """Test filtering by file extension."""
        result = await read_assistant_files(
            str(project_folder),
            exclude_docs=False,
            allowed_extensions="yml,yaml",
        )

        assert result.error is None
        # Should have yaml files
        assert "domain.yml" in result.files
        assert "config.yml" in result.files
        # Should not have txt files
        assert "readme.txt" not in result.files


class TestGetFileContent:
    """Test get_file_content function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with test files."""
        (tmp_path / "domain.yml").write_text("version: '3.1'\nintents: []")
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_file_content_success(self, project_folder: Path):
        """Test reading file content successfully."""
        result = await get_file_content(str(project_folder), "domain.yml")

        assert isinstance(result, FileContentResponse)
        assert result.error is None
        assert result.exists is True
        assert result.content == "version: '3.1'\nintents: []"
        assert result.file_path == "domain.yml"

    @pytest.mark.asyncio
    async def test_get_file_content_not_found(self, project_folder: Path):
        """Test reading nonexistent file."""
        result = await get_file_content(str(project_folder), "nonexistent.yml")

        assert result.exists is False
        assert result.content is None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_file_content_restricted_path(self, project_folder: Path):
        """Test reading restricted (hidden) file."""
        # Create a hidden file
        hidden_file = project_folder / ".hidden"
        hidden_file.write_text("secret content")

        result = await get_file_content(str(project_folder), ".hidden")

        # Should fail due to restricted path
        assert result.error is not None
        assert result.content is None

    @pytest.mark.asyncio
    async def test_get_file_content_subdirectory(self, project_folder: Path):
        """Test reading file from subdirectory."""
        data_dir = project_folder / "data"
        data_dir.mkdir()
        (data_dir / "nlu.yml").write_text("nlu content")

        result = await get_file_content(str(project_folder), "data/nlu.yml")

        assert result.exists is True
        assert result.content == "nlu content"


class TestWriteFile:
    """Test write_file function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create an empty project folder."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_write_file_new(self, project_folder: Path):
        """Test writing a new file."""
        content = "version: '3.1'\nslots:\n  name:\n    type: text"

        result = await write_file(str(project_folder), "domain.yml", content)

        assert isinstance(result, WriteFileResponse)
        assert result.success is True
        assert (project_folder / "domain.yml").read_text() == content

    @pytest.mark.asyncio
    async def test_write_file_overwrite(self, project_folder: Path):
        """Test overwriting existing file."""
        # Create existing file
        (project_folder / "domain.yml").write_text("old content")

        new_content = "new content"
        result = await write_file(str(project_folder), "domain.yml", new_content)

        assert result.success is True
        assert (project_folder / "domain.yml").read_text() == new_content

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, project_folder: Path):
        """Test that write_file creates parent directories."""
        content = "nlu content"

        result = await write_file(str(project_folder), "data/nlu/training.yml", content)

        assert result.success is True
        assert (project_folder / "data" / "nlu" / "training.yml").exists()
        assert (project_folder / "data" / "nlu" / "training.yml").read_text() == content

    @pytest.mark.asyncio
    async def test_write_file_restricted_path(self, project_folder: Path):
        """Test writing to restricted path fails."""
        result = await write_file(str(project_folder), ".hidden/file.txt", "content")

        assert result.success is False


class TestUpdateFiles:
    """Test update_files function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create an empty project folder."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_update_files_success(self, project_folder: Path):
        """Test updating multiple files."""
        files = {
            "domain.yml": "version: '3.1'\nintents: []",
            "config.yml": "pipeline: []",
            "data/nlu.yml": "nlu:\n  - intent: greet",
        }

        result = await update_files(str(project_folder), files)

        assert isinstance(result, UpdateFilesResponse)
        assert result.success is True
        assert len(result.updated) == 3
        assert len(result.failed) == 0

        # Verify file contents
        assert (
            project_folder / "domain.yml"
        ).read_text() == "version: '3.1'\nintents: []"
        assert (project_folder / "config.yml").read_text() == "pipeline: []"
        assert (
            project_folder / "data" / "nlu.yml"
        ).read_text() == "nlu:\n  - intent: greet"

    @pytest.mark.asyncio
    async def test_update_files_partial_failure(self, project_folder: Path):
        """Test updating files with some failures."""
        files = {
            "domain.yml": "valid content",
            ".hidden/secret.txt": "should fail",  # Restricted path
        }

        result = await update_files(str(project_folder), files)

        assert result.success is False  # At least one failure
        assert "domain.yml" in result.updated
        assert len(result.failed) == 1

    @pytest.mark.asyncio
    async def test_update_files_empty(self, project_folder: Path):
        """Test updating with empty files dict."""
        result = await update_files(str(project_folder), {})

        assert result.success is True
        assert len(result.updated) == 0
        assert len(result.failed) == 0


class TestFunctionToolWrappers:
    @pytest.fixture
    def project_folder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Create a project folder and configure the server-level global."""
        # Create some test files
        (tmp_path / "domain.yml").write_text("version: '3.1'")
        (tmp_path / "config.yml").write_text("pipeline: []")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "nlu.yml").write_text("nlu: []")

        monkeypatch.setattr(
            "rasa.builder.copilot.mcp_server.server._project_folder_path",
            str(tmp_path),
        )

        return tmp_path

    @pytest.mark.asyncio
    async def test_list_project_files(self, project_folder: Path):
        """Test list_project_files wrapper gets project folder from env."""
        # FunctionTool.on_invoke_tool takes (ctx, input) where input is JSON string
        ctx = _make_tool_context("list_project_files")
        result = await list_project_files.on_invoke_tool(ctx, "{}")

        assert result.success is True
        assert "domain.yml" in result.files

    @pytest.mark.asyncio
    async def test_read_project_files(self, project_folder: Path):
        """Test read_project_files wrapper."""
        ctx = _make_tool_context("read_project_files")
        result = await read_project_files.on_invoke_tool(
            ctx, '{"exclude_docs": true, "allowed_extensions": "yml"}'
        )

        assert result.count >= 1
        assert "domain.yml" in result.files

    @pytest.mark.asyncio
    async def test_get_project_file(self, project_folder: Path):
        """Test get_project_file wrapper."""
        ctx = _make_tool_context("get_project_file")
        result = await get_project_file.on_invoke_tool(
            ctx, '{"file_path": "domain.yml"}'
        )

        assert result.exists is True
        assert result.content == "version: '3.1'"

    @pytest.mark.asyncio
    async def test_write_project_file(self, project_folder: Path):
        """Test write_project_file wrapper."""
        ctx = _make_tool_context("write_project_file")
        result = await write_project_file.on_invoke_tool(
            ctx, '{"file_path": "new_file.yml", "content": "test content"}'
        )

        assert result.success is True
        assert (project_folder / "new_file.yml").read_text() == "test content"

    @pytest.mark.asyncio
    async def test_update_multiple_files_success(self, project_folder: Path):
        """Test update_multiple_files wrapper with valid JSON."""
        # The files_json parameter itself is a JSON string
        ctx = _make_tool_context("update_multiple_files")
        input_json = (
            '{"files_json": "{\\"file1.yml\\": \\"content1\\", '
            '\\"file2.yml\\": \\"content2\\"}"}'
        )
        result = await update_multiple_files.on_invoke_tool(ctx, input_json)

        assert result.success is True
        assert "file1.yml" in result.updated
        assert "file2.yml" in result.updated

    @pytest.mark.asyncio
    async def test_update_multiple_files_invalid_json(self, project_folder: Path):
        """Test update_multiple_files wrapper with invalid JSON."""
        ctx = _make_tool_context("update_multiple_files")
        result = await update_multiple_files.on_invoke_tool(
            ctx, '{"files_json": "not valid json"}'
        )

        assert result.success is False
        assert "Invalid JSON" in result.failed[0].error

    @pytest.mark.asyncio
    async def test_update_multiple_files_invalid_format(self, project_folder: Path):
        """Test update_multiple_files wrapper with wrong data structure."""
        # Valid JSON but wrong format (array instead of dict)
        ctx = _make_tool_context("update_multiple_files")
        result = await update_multiple_files.on_invoke_tool(
            ctx, '{"files_json": "[\\"file1.yml\\", \\"file2.yml\\"]"}'
        )

        assert result.success is False
        assert "Invalid input format" in result.failed[0].error

    @pytest.mark.asyncio
    async def test_wrapper_without_project_folder(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that wrappers return error when project folder is not set."""
        monkeypatch.setattr(
            "rasa.builder.copilot.mcp_server.server._project_folder_path",
            None,
        )

        # on_invoke_tool catches exceptions and returns error as string
        ctx = _make_tool_context("list_project_files")
        result = await list_project_files.on_invoke_tool(ctx, "{}")

        assert isinstance(result, str)
        assert "project folder" in result.lower()
        assert "error" in result.lower()
