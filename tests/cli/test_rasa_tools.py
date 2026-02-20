import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rasa.builder.copilot.constants import RASA_PROJECT_FOLDER_ENV_VAR
from rasa.builder.copilot.mcp_server.constants import (
    MCP_DEFAULT_HOST,
    MCP_TOOLS_DEFAULT_PORT,
    MCP_TRANSPORT_STDIO,
    MCP_TRANSPORT_STREAMABLE_HTTP,
)
from rasa.cli.tools import _resolve_project_path, run_tools
from rasa.shared.exceptions import RasaException

tools_module_path = "rasa.cli.tools"


class TestResolveProjectPath:
    """Test _resolve_project_path function."""

    def test_cli_arg_takes_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that CLI arg takes priority over env var and cwd."""
        env_path = tmp_path / "env_project"
        env_path.mkdir()
        cli_path = tmp_path / "cli_project"
        cli_path.mkdir()

        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(env_path))

        result = _resolve_project_path(str(cli_path))
        assert result == cli_path.resolve()

    def test_env_var_used_when_cli_arg_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that env var is used when CLI arg is None."""
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(tmp_path))

        result = _resolve_project_path(None)
        assert result == tmp_path.resolve()

    def test_falls_back_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that it falls back to cwd when no CLI arg and no env var."""
        monkeypatch.delenv(RASA_PROJECT_FOLDER_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)

        result = _resolve_project_path(None)
        assert result == tmp_path.resolve()

    def test_resolves_to_absolute_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that relative paths are resolved to absolute."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(tmp_path)

        result = _resolve_project_path("subdir")
        assert result.is_absolute()
        assert result == subdir.resolve()

    def test_raises_for_non_existent_path(self) -> None:
        """Test that RasaException is raised for non-existent path."""
        with pytest.raises(RasaException, match="does not exist"):
            _resolve_project_path("/path/that/does/not/exist")

    def test_raises_when_path_is_file(self, tmp_path: Path) -> None:
        """Test that RasaException is raised when path is a file, not directory."""
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("content")

        with pytest.raises(RasaException, match="not a directory"):
            _resolve_project_path(str(file_path))


class TestRunTools:
    """Test run_tools function."""

    @pytest.fixture
    def mock_run_server(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Mock the run_server function."""
        mock = MagicMock()
        # Mock at the location where it's imported
        monkeypatch.setattr(
            "rasa.builder.copilot.mcp_server.server.run_server",
            mock,
        )
        return mock

    def test_stdio_mode_redirects_logging_to_stderr(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that stdio mode redirects logging from stdout to stderr."""
        import logging
        import sys

        # Set up a handler that writes to stdout (simulating default config)
        root_logger = logging.getLogger()
        stdout_handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(stdout_handler)

        args = argparse.Namespace(
            mode="stdio",
            port=MCP_TOOLS_DEFAULT_PORT,
            project=str(tmp_path),
        )

        run_tools(args)

        # Verify that the handler's stream was changed to stderr
        assert stdout_handler.stream == sys.stderr

        # Clean up
        root_logger.removeHandler(stdout_handler)

    def test_stdio_mode_calls_run_server_correctly(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that stdio mode calls run_server with correct transport."""
        args = argparse.Namespace(
            mode="stdio",
            port=MCP_TOOLS_DEFAULT_PORT,
            project=str(tmp_path),
        )

        run_tools(args)

        mock_run_server.assert_called_once_with(
            transport=MCP_TRANSPORT_STDIO,
            project_folder=str(tmp_path.resolve()),
        )

    def test_http_mode_calls_run_server_correctly(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that http mode calls run_server with correct parameters."""
        args = argparse.Namespace(
            mode="http",
            port=9999,
            project=str(tmp_path),
        )

        run_tools(args)

        mock_run_server.assert_called_once_with(
            host=MCP_DEFAULT_HOST,
            port=9999,
            transport=MCP_TRANSPORT_STREAMABLE_HTTP,
            project_folder=str(tmp_path.resolve()),
        )

    def test_passes_project_folder_to_run_server(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that resolved project folder is passed to run_server."""
        monkeypatch.delenv(RASA_PROJECT_FOLDER_ENV_VAR, raising=False)

        args = argparse.Namespace(
            mode="stdio",
            port=MCP_TOOLS_DEFAULT_PORT,
            project=str(tmp_path),
        )

        run_tools(args)

        mock_run_server.assert_called_once_with(
            transport=MCP_TRANSPORT_STDIO,
            project_folder=str(tmp_path.resolve()),
        )

    def test_invalid_project_path_raises(
        self,
        mock_run_server: MagicMock,
    ) -> None:
        """Test that invalid project path raises before calling run_server."""
        args = argparse.Namespace(
            mode="stdio",
            port=MCP_TOOLS_DEFAULT_PORT,
            project="/nonexistent/path",
        )

        with pytest.raises(RasaException):
            run_tools(args)

        mock_run_server.assert_not_called()

    def test_project_path_resolution_with_env_var(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that project path is resolved using env var when CLI arg is None."""
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(tmp_path))

        args = argparse.Namespace(
            mode="stdio",
            port=MCP_TOOLS_DEFAULT_PORT,
            project=None,
        )

        run_tools(args)

        mock_run_server.assert_called_once_with(
            transport=MCP_TRANSPORT_STDIO,
            project_folder=str(tmp_path.resolve()),
        )
