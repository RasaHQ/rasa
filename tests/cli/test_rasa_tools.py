import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rasa.builder.copilot.constants import RASA_PROJECT_FOLDER_ENV_VAR
from rasa.builder.copilot.mcp_server.constants import (
    MCP_TRANSPORT_STDIO,
    MCP_TRANSPORT_STREAMABLE_HTTP,
)
from rasa.cli.arguments.tools import MCP_TOOLS_DEFAULT_HOST
from rasa.cli.tools import (
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
    RunConfig,
    _resolve_project_path,
    _resolve_tools_run_config,
    run_tools,
)
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


class TestResolveRunConfig:
    """Tests for _resolve_tools_run_config."""

    @pytest.fixture()
    def config_file(
        self,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ) -> Path | None:
        """Pre-save a RunConfig. Param controls save location."""
        location = request.param
        if not location:
            return None
        cfg = RunConfig(mode="http", port=5555)
        if location == "default":
            path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
            cfg.save(path)
            return path
        if location == "custom_dir":
            custom_dir = tmp_path / "other"
            custom_dir.mkdir()
            cfg.save(
                custom_dir / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME,
            )
            return custom_dir
        if location == "custom_file":
            path = tmp_path / "my_config.yaml"
            cfg.save(path)
            return path
        if location == "dot_rasa_dir":
            dot_rasa = tmp_path / TOOLS_CONFIG_DIR
            cfg.save(dot_rasa / TOOLS_CONFIG_FILENAME)
            return dot_rasa
        return None

    @pytest.mark.parametrize(
        "cli_mode, cli_port, use_cli_config, config_file,"
        " expected_mode, expected_port, has_source",
        [
            pytest.param(
                None,
                None,
                False,
                None,
                "stdio",
                7331,
                False,
                id="no_config_no_cli_returns_defaults",
            ),
            pytest.param(
                "http",
                9000,
                False,
                None,
                "http",
                9000,
                False,
                id="cli_values_used_when_no_config",
            ),
            pytest.param(
                "http",
                None,
                False,
                None,
                "http",
                7331,
                False,
                id="cli_partial_args_fill_defaults",
            ),
            pytest.param(
                None,
                None,
                False,
                "default",
                "http",
                5555,
                True,
                id="existing_config_loaded",
            ),
            pytest.param(
                "stdio",
                None,
                False,
                "default",
                "stdio",
                7331,
                False,
                id="cli_args_skip_existing_config",
            ),
            pytest.param(
                None,
                None,
                True,
                "custom_dir",
                "http",
                5555,
                True,
                id="custom_config_dir_via_cli",
            ),
            pytest.param(
                None,
                None,
                True,
                "custom_file",
                "http",
                5555,
                True,
                id="custom_config_file_via_cli",
            ),
            pytest.param(
                None,
                None,
                True,
                "dot_rasa_dir",
                "http",
                5555,
                True,
                id="config_path_ending_with_dot_rasa",
            ),
        ],
        indirect=["config_file"],
    )
    def test_resolve(
        self,
        tmp_path: Path,
        config_file: Path | None,
        cli_mode: str | None,
        cli_port: int | None,
        use_cli_config: bool,
        expected_mode: str,
        expected_port: int,
        has_source: bool,
    ) -> None:
        cfg, source = _resolve_tools_run_config(
            cli_mode=cli_mode,
            cli_port=cli_port,
            cli_config=(str(config_file) if use_cli_config else None),
            project_dir=tmp_path,
        )
        assert cfg.mode == expected_mode
        assert cfg.port == expected_port
        assert (source is not None) == has_source


class TestRunTools:
    """Test run_tools function."""

    @pytest.fixture
    def mock_run_server(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Mock the run_server function."""
        mock = MagicMock()
        monkeypatch.setattr(
            "rasa.builder.copilot.mcp_server.server.run_server",
            mock,
        )
        return mock

    # -- server dispatch -----------------------------------------------------------

    def test_stdio_mode_calls_run_server_correctly(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
    ) -> None:
        """Test that stdio mode calls run_server with correct transport."""
        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project=str(tmp_path),
            config=None,
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
    ) -> None:
        """Test that http mode calls run_server with correct parameters."""
        args = argparse.Namespace(
            mode="http",
            port=9999,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        mock_run_server.assert_called_once_with(
            host=MCP_TOOLS_DEFAULT_HOST,
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
            mode=None,
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        mock_run_server.assert_called_once_with(
            transport=MCP_TRANSPORT_STDIO,
            project_folder=str(tmp_path.resolve()),
        )

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
            port=None,
            project=None,
            config=None,
        )

        run_tools(args)

        mock_run_server.assert_called_once()

    def test_invalid_project_path_raises(
        self,
        mock_run_server: MagicMock,
    ) -> None:
        """Test that invalid project path raises before calling run_server."""
        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project="/nonexistent/path",
            config=None,
        )

        with pytest.raises(RasaException):
            run_tools(args)

        mock_run_server.assert_not_called()

    # -- logging redirect ----------------------------------------------------------

    def test_stdio_mode_redirects_logging_to_stderr(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
    ) -> None:
        """Test that stdio mode redirects logging from stdout to stderr."""
        import logging
        import sys

        root_logger = logging.getLogger()
        stdout_handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(stdout_handler)

        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        assert stdout_handler.stream == sys.stderr

        root_logger.removeHandler(stdout_handler)

    # -- stdio stdout must stay clean ----------------------------------------------

    def test_stdio_mode_nothing_written_to_stdout(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """In stdio mode, stdout must stay clean for JSON-RPC messages."""
        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        captured = capsys.readouterr()
        assert (
            captured.out == ""
        ), f"Unexpected stdout output in stdio mode: {captured.out!r}"

    def test_stdio_mode_nothing_written_to_stdout_with_existing_config(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Stdio stays clean even when loading an existing config file."""
        existing = RunConfig(mode="stdio", port=7331)
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        existing.save(config_path)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        captured = capsys.readouterr()
        assert (
            captured.out == ""
        ), f"Unexpected stdout output in stdio mode: {captured.out!r}"

    def test_stdio_mode_structlog_does_not_write_to_stdout(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """structlog calls after redirect must not leak to stdout."""
        import structlog

        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        # Emit a structlog message after the redirect has been applied
        structlog.get_logger().info("test.structlog.after_redirect", key="value")

        captured = capsys.readouterr()
        assert (
            captured.out == ""
        ), f"structlog leaked to stdout in stdio mode: {captured.out!r}"

    # -- auto-save -----------------------------------------------------------------

    def test_auto_saves_config_when_no_file(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
    ) -> None:
        """Config is auto-saved to the default location on first run."""
        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        expected = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        assert expected.is_file()

        loaded = RunConfig.load(expected)
        assert loaded.mode == "stdio"

    def test_auto_saves_http_config_when_no_file(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
    ) -> None:
        """HTTP mode also auto-saves config on first run."""
        args = argparse.Namespace(
            mode="http",
            port=8080,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        expected = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        assert expected.is_file()

        loaded = RunConfig.load(expected)
        assert loaded.mode == "http"
        assert loaded.port == 8080

    def test_does_not_overwrite_existing_config(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
    ) -> None:
        """If a config file already exists, auto-save is skipped."""
        existing = RunConfig(mode="http", port=5555)
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        existing.save(config_path)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        loaded = RunConfig.load(config_path)
        assert loaded.mode == "http"
        assert loaded.port == 5555

    def test_server_starts_even_if_read_text_raises_after_save(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
    ) -> None:
        """read_text() failure after save must not propagate."""
        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project=str(tmp_path),
            config=None,
        )

        with patch("pathlib.Path.read_text", side_effect=OSError("disk error")):
            run_tools(args)

        mock_run_server.assert_called_once()

    # -- stderr guidance messages --------------------------------------------------

    def test_stdio_stderr_on_auto_save(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Stdio auto-save prints settings and hint to stderr."""
        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        captured = capsys.readouterr()
        assert "Saved run config to" in captured.err
        assert "mode: stdio" in captured.err
        assert "--mode" in captured.err

    def test_stdio_stderr_on_loaded_config(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Stdio mode logs loaded config path and values to stderr."""
        existing = RunConfig(mode="stdio", port=7331)
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        existing.save(config_path)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        captured = capsys.readouterr()
        assert "Loaded run config from" in captured.err
        assert str(config_path) in captured.err
        assert "mode: stdio" in captured.err

    def test_http_stdout_on_auto_save(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """HTTP auto-save prints settings to stdout with colors."""
        args = argparse.Namespace(
            mode="http",
            port=9999,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        captured = capsys.readouterr()
        assert "Saved run config to" in captured.out
        assert "mode: http" in captured.out

    def test_http_stdout_on_loaded_config(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """HTTP mode prints loaded config info to stdout with colors."""
        existing = RunConfig(mode="http", port=5555)
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        existing.save(config_path)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project=str(tmp_path),
            config=None,
        )

        run_tools(args)

        captured = capsys.readouterr()
        assert "Loaded run config from" in captured.out
        assert "mode: http" in captured.out

    def test_server_starts_even_if_loaded_config_read_text_raises(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Failure when displaying a loaded config must not prevent startup."""
        existing = RunConfig(mode="http", port=5555)
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        existing.save(config_path)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project=str(tmp_path),
            config=None,
        )

        with patch("pathlib.Path.read_text", side_effect=OSError("disk error")):
            run_tools(args)

        mock_run_server.assert_called_once()
        captured = capsys.readouterr()
        assert "Loaded run config from" in captured.out
        assert "<could not read file>" in captured.out
