"""Tests for MCP server implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.shared.exceptions import RasaException


class TestGetProjectFolder:
    """Test _get_project_folder function."""

    def test_get_project_folder_returns_env_value(self, monkeypatch):
        """Test that _get_project_folder returns RASA_PROJECT_FOLDER env var."""
        from rasa.builder.copilot.mcp_server.server import _get_project_folder

        monkeypatch.setenv("RASA_PROJECT_FOLDER", "/test/project/path")
        result = _get_project_folder()
        assert result == "/test/project/path"

    def test_get_project_folder_raises_when_not_set(self, monkeypatch):
        """Test that _get_project_folder raises when env var not set."""
        from rasa.builder.copilot.mcp_server.server import _get_project_folder

        monkeypatch.delenv("RASA_PROJECT_FOLDER", raising=False)
        with pytest.raises(RasaException, match="Project folder not configured"):
            _get_project_folder()


class TestDummyProgressReporter:
    """Test dummy_progress_reporter context manager."""

    @pytest.mark.asyncio
    async def test_dummy_progress_reporter_reports_progress(self):
        """Test that dummy_progress_reporter reports progress."""
        from rasa.builder.copilot.mcp_server.server import dummy_progress_reporter

        mock_ctx = MagicMock()
        mock_ctx.report_progress = AsyncMock()
        mock_ctx.info = AsyncMock()

        # Use a very short interval for testing
        async with dummy_progress_reporter(mock_ctx, interval_seconds=0.1):
            # Wait a bit for at least one progress report
            await asyncio.sleep(0.25)

        # Should have reported progress at least once
        assert mock_ctx.report_progress.call_count >= 1

    @pytest.mark.asyncio
    async def test_dummy_progress_reporter_stops_on_exit(self):
        """Test that progress reporting stops when context exits."""
        from rasa.builder.copilot.mcp_server.server import dummy_progress_reporter

        mock_ctx = MagicMock()
        mock_ctx.report_progress = AsyncMock()
        mock_ctx.info = AsyncMock()

        async with dummy_progress_reporter(mock_ctx, interval_seconds=0.1):
            await asyncio.sleep(0.15)

        call_count_after_exit = mock_ctx.report_progress.call_count
        await asyncio.sleep(0.2)

        # No more calls after exiting the context
        assert mock_ctx.report_progress.call_count == call_count_after_exit


class TestMCPServerTools:
    """Test MCP server tool functions."""

    @pytest.fixture
    def mock_project_folder(self, monkeypatch, tmp_path):
        """Set up mock project folder."""
        monkeypatch.setenv("RASA_PROJECT_FOLDER", str(tmp_path))
        return tmp_path

    @pytest.mark.asyncio
    async def test_list_project_files_tool(self, mock_project_folder):
        """Test list_project_files tool."""
        from rasa.builder.copilot.mcp_server.server import list_project_files

        # Create some test files
        (mock_project_folder / "domain.yml").write_text("version: '3.1'")
        (mock_project_folder / "config.yml").write_text("pipeline: []")

        result = await list_project_files()

        assert result.success is True
        assert "domain.yml" in result.tree or "config.yml" in result.tree

    @pytest.mark.asyncio
    async def test_get_project_file_tool(self, mock_project_folder):
        """Test get_project_file tool."""
        from rasa.builder.copilot.mcp_server.server import get_project_file

        # Create a test file
        (mock_project_folder / "domain.yml").write_text("version: '3.1'\nintents: []")

        result = await get_project_file("domain.yml")

        assert result.error is None
        assert result.content == "version: '3.1'\nintents: []"

    @pytest.mark.asyncio
    async def test_get_project_file_tool_not_found(self, mock_project_folder):
        """Test get_project_file tool with non-existent file."""
        from rasa.builder.copilot.mcp_server.server import get_project_file

        result = await get_project_file("nonexistent.yml")

        assert result.exists is False
        assert "not found" in result.error.lower() or result.content is None

    @pytest.mark.asyncio
    async def test_write_project_file_tool(self, mock_project_folder):
        """Test write_project_file tool."""
        from rasa.builder.copilot.mcp_server.server import write_project_file

        content = "version: '3.1'\nslots:\n  name:\n    type: text"
        result = await write_project_file("domain/slots.yml", content)

        assert result.success is True
        assert (mock_project_folder / "domain" / "slots.yml").exists()
        assert (mock_project_folder / "domain" / "slots.yml").read_text() == content

    @pytest.mark.asyncio
    async def test_update_multiple_files_tool(self, mock_project_folder):
        """Test update_multiple_files tool."""
        from rasa.builder.copilot.mcp_server.server import update_multiple_files

        files = {
            "domain.yml": "version: '3.1'\nintents: []",
            "config.yml": "pipeline: []",
        }

        result = await update_multiple_files(files)

        assert result.success is True
        assert (
            mock_project_folder / "domain.yml"
        ).read_text() == "version: '3.1'\nintents: []"
        assert (mock_project_folder / "config.yml").read_text() == "pipeline: []"

    @pytest.mark.asyncio
    async def test_read_project_files_tool(self, mock_project_folder):
        """Test read_project_files tool."""
        from rasa.builder.copilot.mcp_server.server import read_project_files

        # Create some test files
        (mock_project_folder / "domain.yml").write_text("version: '3.1'")
        (mock_project_folder / "config.yml").write_text("pipeline: []")
        # Create a non-yaml file that should be filtered
        (mock_project_folder / "readme.txt").write_text("This is readme")

        result = await read_project_files(
            exclude_docs=True,
            allowed_extensions="yaml,yml",
        )

        assert result.error is None
        assert "domain.yml" in result.files
        assert "config.yml" in result.files
        # txt file should be filtered out
        assert "readme.txt" not in result.files


class TestMCPServerDocumentSearch:
    """Test MCP server document search tool."""

    @pytest.fixture
    def mock_project_folder(self, monkeypatch, tmp_path):
        """Set up mock project folder."""
        monkeypatch.setenv("RASA_PROJECT_FOLDER", str(tmp_path))
        return tmp_path

    @pytest.mark.asyncio
    async def test_search_docs_tool(self, mock_project_folder):
        """Test search_docs tool."""
        from rasa.builder.copilot.mcp_server.server import search_docs

        with patch(
            "rasa.builder.copilot.mcp_server.tools.document_search.search_rasa_documentation"
        ) as mock_search:
            mock_search.return_value = MagicMock(
                success=True,
                documents=[
                    {
                        "title": "Flows Guide",
                        "url": "https://docs.rasa.com/flows",
                        "content": "Flow documentation",
                    }
                ],
            )

            result = await search_docs("How do I create a flow?")

            # Result depends on the mock implementation
            assert result is not None


class TestMCPServerValidation:
    """Test MCP server validation and training tools."""

    @pytest.fixture
    def mock_project_folder(self, monkeypatch, tmp_path):
        """Set up mock project folder."""
        monkeypatch.setenv("RASA_PROJECT_FOLDER", str(tmp_path))
        return tmp_path

    @pytest.mark.asyncio
    async def test_validate_project_tool(self, mock_project_folder):
        """Test validate_project tool."""
        from rasa.builder.copilot.mcp_server.server import validate_project

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.report_progress = AsyncMock()

        with patch(
            "rasa.builder.copilot.mcp_server.tools.validation_training.validate_assistant_project"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(
                success=True,
                valid=True,
                errors=[],
                warnings=[],
            )

            await validate_project(mock_ctx)

            mock_validate.assert_called_once_with(str(mock_project_folder))
            mock_ctx.info.assert_called()


class TestMCPServerBotInteraction:
    """Test MCP server bot interaction tools."""

    @pytest.fixture
    def mock_project_folder(self, monkeypatch, tmp_path):
        """Set up mock project folder."""
        monkeypatch.setenv("RASA_PROJECT_FOLDER", str(tmp_path))
        return tmp_path

    @pytest.mark.asyncio
    async def test_talk_to_assistant_empty_messages(self, mock_project_folder):
        """Test talk_to_assistant with empty messages."""
        from rasa.builder.copilot.mcp_server.server import talk_to_assistant

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        result = await talk_to_assistant(mock_ctx, [])

        assert result.success is False
        assert "No messages provided" in result.error

    @pytest.mark.asyncio
    async def test_talk_to_assistant_with_messages(self, mock_project_folder):
        """Test talk_to_assistant with messages."""
        from rasa.builder.copilot.mcp_server.server import talk_to_assistant

        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.report_progress = AsyncMock()

        with patch(
            "rasa.builder.copilot.mcp_server.tools.bot_interaction.talk_to_assistant"
        ) as mock_talk:
            mock_talk.return_value = MagicMock(
                success=True,
                session_id="test-session",
                message_count=1,
                conversation=[{"user": "Hello", "bot": "Hi!"}],
                tracker_context=None,
                error=None,
            )

            await talk_to_assistant(mock_ctx, ["Hello"])

            mock_talk.assert_called_once_with(["Hello"])


class TestMCPServerResources:
    """Test MCP server resources."""

    @pytest.fixture
    def mock_project_folder(self, monkeypatch, tmp_path):
        """Set up mock project folder."""
        monkeypatch.setenv("RASA_PROJECT_FOLDER", str(tmp_path))
        return tmp_path

    @pytest.mark.asyncio
    async def test_project_files_resource(self, mock_project_folder):
        """Test project_files_resource."""
        from rasa.builder.copilot.mcp_server.server import project_files_resource

        # Create some test files
        (mock_project_folder / "domain.yml").write_text("version: '3.1'")

        with patch(
            "rasa.builder.copilot.mcp_server.tools.file_operations.read_assistant_files"
        ) as mock_read:
            mock_read.return_value = MagicMock(
                error=None,
                files={"domain.yml": "version: '3.1'"},
                model_dump_json=lambda: '{"files": {"domain.yml": "version: \'3.1\'"}}',
            )

            result = await project_files_resource()

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_project_tree_resource(self, mock_project_folder):
        """Test project_tree_resource."""
        from rasa.builder.copilot.mcp_server.server import project_tree_resource

        # Create some test files
        (mock_project_folder / "domain.yml").write_text("version: '3.1'")

        with patch(
            "rasa.builder.copilot.mcp_server.tools.file_operations.list_files"
        ) as mock_list:
            mock_list.return_value = MagicMock(
                success=True,
                tree="domain.yml",
                model_dump_json=lambda: '{"success": true, "tree": "domain.yml"}',
            )

            result = await project_tree_resource()

            assert isinstance(result, str)


class TestMCPServerPrompts:
    """Test MCP server prompts."""

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        """Test system_prompt returns valid prompt structure."""
        from rasa.builder.copilot.mcp_server.server import system_prompt

        with patch(
            "rasa.builder.copilot.mcp_server.prompts.prompt_loader.get_copilot_system_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = "Test system prompt content"

            result = await system_prompt()

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["role"] == "user"
            assert result[0]["content"]["type"] == "text"
            assert result[0]["content"]["text"] == "Test system prompt content"

    @pytest.mark.asyncio
    async def test_training_error_analysis_prompt(self):
        """Test training_error_analysis returns valid prompt structure."""
        from rasa.builder.copilot.mcp_server.server import training_error_analysis

        with patch(
            "rasa.builder.copilot.mcp_server.prompts.prompt_loader.get_training_error_handler_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = "Test training error prompt"

            result = await training_error_analysis()

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["role"] == "user"


class TestRunServer:
    """Test run_server function."""

    def test_run_server_logs_startup(self, monkeypatch, tmp_path):
        """Test that run_server logs startup information."""
        from rasa.builder.copilot.mcp_server.server import run_server

        monkeypatch.setenv("RASA_PROJECT_FOLDER", str(tmp_path))

        # Mock the mcp.run to prevent actually starting the server
        with patch("rasa.builder.copilot.mcp_server.server.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()

            # This would normally start the server
            run_server(host="127.0.0.1", port=5051)

            mock_mcp.run.assert_called_once_with(transport="streamable-http")
            assert mock_mcp.settings.host == "127.0.0.1"
            assert mock_mcp.settings.port == 5051
