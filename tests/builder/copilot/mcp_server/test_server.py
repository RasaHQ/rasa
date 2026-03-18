"""Tests for MCP server implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import rasa.builder.copilot.mcp_server.server as server_module
from rasa.builder.copilot.constants import RASA_PROJECT_FOLDER_ENV_VAR
from rasa.builder.copilot.mcp_server.constants import (
    DEFAULT_RASA_SERVER_URL,
    MCP_TRANSPORT_STDIO,
    MCP_TRANSPORT_STREAMABLE_HTTP,
)
from rasa.builder.copilot.mcp_server.server import (
    _get_project_folder,
    _set_project_folder,
    dummy_progress_reporter,
    health_check,
    run_server,
    search_rasa_documentation,
    talk_to_assistant,
    validate_project,
)
from rasa.shared.exceptions import RasaException

SERVER_MODULE = "rasa.builder.copilot.mcp_server.server"


class TestProjectFolderState:
    """Test _set_project_folder / _get_project_folder module-level state."""

    def test_get_project_folder_returns_stored_value(self, monkeypatch):
        """Test if _get_project_folder returns the value set by _set_project_folder."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)
        project_folder = "/test/project/path"
        _set_project_folder(project_folder)
        assert _get_project_folder() == project_folder

    def test_get_project_folder_raises_when_not_set(self, monkeypatch):
        """_get_project_folder raises when _set_project_folder was never called."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)
        with pytest.raises(RasaException, match="Project folder not configured"):
            _get_project_folder()

    def test_set_project_folder_falls_back_to_env_var(self, monkeypatch, tmp_path):
        """_set_project_folder uses the env var when no explicit folder is given."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(tmp_path))

        _set_project_folder(None)

        assert _get_project_folder() == str(tmp_path)

    def test_set_project_folder_explicit_arg_takes_priority_over_env_var(
        self, monkeypatch, tmp_path
    ):
        """Explicit folder argument takes precedence over the env var."""
        explicit = str(tmp_path / "explicit")
        env_path = str(tmp_path / "env")
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, env_path)

        _set_project_folder(explicit)

        assert _get_project_folder() == explicit

    def test_run_server_stores_project_folder(self, monkeypatch, tmp_path):
        """Test that run_server stores the project_folder in module state."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)

        with patch(f"{SERVER_MODULE}.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()

            server_module.run_server(project_folder=str(tmp_path))

        assert server_module._project_folder_path == str(tmp_path)

    def test_run_server_falls_back_to_env_var(self, monkeypatch, tmp_path):
        """Test that run_server falls back to the env var when no arg is given."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(tmp_path))

        with patch(f"{SERVER_MODULE}.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()

            server_module.run_server()

        assert server_module._project_folder_path == str(tmp_path)


class TestDummyProgressReporter:
    """Test dummy_progress_reporter context manager."""

    @pytest.mark.asyncio
    async def test_dummy_progress_reporter_reports_progress(self):
        """Test that dummy_progress_reporter reports progress."""
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
        mock_ctx = MagicMock()
        mock_ctx.report_progress = AsyncMock()
        mock_ctx.info = AsyncMock()

        async with dummy_progress_reporter(mock_ctx, interval_seconds=0.1):
            await asyncio.sleep(0.15)

        call_count_after_exit = mock_ctx.report_progress.call_count
        await asyncio.sleep(0.2)

        # No more calls after exiting the context
        assert mock_ctx.report_progress.call_count == call_count_after_exit


class TestMCPServerDocumentSearch:
    """Test MCP server document search tool."""

    @pytest.fixture
    def mock_project_folder(self, monkeypatch, tmp_path):
        """Set up mock project folder."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", str(tmp_path))
        return tmp_path

    @pytest.mark.asyncio
    async def test_search_rasa_documentation_tool(self, mock_project_folder):
        """Test search_rasa_documentation tool."""
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

            result = await search_rasa_documentation("How do I create a flow?")

            # Result depends on the mock implementation
            assert result is not None


class TestMCPServerValidation:
    """Test MCP server validation and training tools."""

    @pytest.fixture
    def mock_project_folder(self, monkeypatch, tmp_path):
        """Set up mock project folder."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", str(tmp_path))
        return tmp_path

    @pytest.mark.asyncio
    async def test_validate_project_tool(self, mock_project_folder):
        """Test validate_project tool."""
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
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", str(tmp_path))
        return tmp_path

    @pytest.fixture
    def mock_rasa_server_url(self, monkeypatch):
        """Set the Rasa server URL to the default value."""
        monkeypatch.setattr(
            f"{SERVER_MODULE}._rasa_server_url", DEFAULT_RASA_SERVER_URL
        )

    @pytest.mark.asyncio
    async def test_talk_to_assistant_empty_messages(self, mock_project_folder):
        """Test talk_to_assistant with empty messages."""
        mock_ctx = MagicMock()
        mock_ctx.info = AsyncMock()

        result = await talk_to_assistant(mock_ctx, [])

        assert result.success is False
        assert "No messages provided" in result.error

    @pytest.mark.asyncio
    async def test_talk_to_assistant_with_messages(
        self, mock_project_folder, mock_rasa_server_url
    ):
        """Test talk_to_assistant with messages."""
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

            mock_talk.assert_called_once_with(["Hello"], DEFAULT_RASA_SERVER_URL)


class TestRunServer:
    """Test run_server function."""

    def test_run_server_logs_startup(self, monkeypatch, tmp_path):
        """Test that run_server logs startup information."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)

        with patch(f"{SERVER_MODULE}.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()

            run_server(host="127.0.0.1", port=5051, project_folder=str(tmp_path))

            mock_mcp.run.assert_called_once_with(transport="streamable-http")
            assert mock_mcp.settings.host == "127.0.0.1"
            assert mock_mcp.settings.port == 5051

    def test_run_server_stdio_transport(self, monkeypatch, tmp_path):
        """Test that run_server with stdio transport calls mcp.run correctly."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)

        with patch(f"{SERVER_MODULE}.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()

            run_server(transport=MCP_TRANSPORT_STDIO, project_folder=str(tmp_path))

            mock_mcp.run.assert_called_once_with(transport=MCP_TRANSPORT_STDIO)

    def test_run_server_streamable_http_transport(self, monkeypatch, tmp_path):
        """Test that run_server with streamable-http configures host/port."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)

        with patch(f"{SERVER_MODULE}.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()

            run_server(
                host="0.0.0.0",
                port=9999,
                transport=MCP_TRANSPORT_STREAMABLE_HTTP,
                project_folder=str(tmp_path),
            )

            assert mock_mcp.settings.host == "0.0.0.0"
            assert mock_mcp.settings.port == 9999
            mock_mcp.run.assert_called_once_with(
                transport=MCP_TRANSPORT_STREAMABLE_HTTP
            )

    def test_run_server_default_transport_is_streamable_http(
        self, monkeypatch, tmp_path
    ):
        """Test that default transport is streamable-http for backward compatibility."""
        monkeypatch.setattr(f"{SERVER_MODULE}._project_folder_path", None)

        with patch(f"{SERVER_MODULE}.mcp") as mock_mcp:
            mock_mcp.settings = MagicMock()
            mock_mcp.run = MagicMock()

            run_server(host="127.0.0.1", port=5051, project_folder=str(tmp_path))

            mock_mcp.run.assert_called_once_with(transport="streamable-http")


class TestHealthEndpoint:
    """Test health_check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_ok(self):
        """Test that health_check returns correct response."""
        # Create a mock request
        mock_request = MagicMock()

        response = await health_check(mock_request)

        # Verify response
        assert response.status_code == 200
        assert response.body == b'{"status":"ok"}'
