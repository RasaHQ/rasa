import argparse
import logging
import sys
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
    _precheck,
    _redirect_logging_to_stderr,
    _resolve_config_path,
    _resolve_project_dir,
    _resolve_tools_run_config,
    _validate_config_exclusivity,
    run_tools,
)
from rasa.shared.exceptions import RasaException

tools_module_path = "rasa.cli.tools"


class TestResolveProjectDir:
    """Tests for _resolve_project_dir."""

    def test_cli_arg_takes_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI arg wins over env var and CWD."""
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(env_dir))

        assert _resolve_project_dir(str(cli_dir)) == cli_dir.resolve()

    def test_env_var_used_when_no_cli_arg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var is used when no CLI arg is given."""
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(tmp_path))

        assert _resolve_project_dir() == tmp_path.resolve()

    def test_config_project_path_used_when_no_cli_or_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """config_project_path is consulted when neither CLI arg nor env var is set."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.delenv(RASA_PROJECT_FOLDER_ENV_VAR, raising=False)

        assert (
            _resolve_project_dir(config_project_path=str(config_dir))
            == config_dir.resolve()
        )

    def test_cli_arg_beats_config_project_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI arg takes priority over config_project_path."""
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.delenv(RASA_PROJECT_FOLDER_ENV_VAR, raising=False)

        assert (
            _resolve_project_dir(str(cli_dir), config_project_path=str(config_dir))
            == cli_dir.resolve()
        )

    def test_env_var_beats_config_project_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var takes priority over config_project_path."""
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(env_dir))

        assert (
            _resolve_project_dir(config_project_path=str(config_dir))
            == env_dir.resolve()
        )

    def test_falls_back_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to CWD when no CLI arg, env var, or config path is given."""
        monkeypatch.delenv(RASA_PROJECT_FOLDER_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)

        assert _resolve_project_dir() == tmp_path.resolve()

    def test_resolves_relative_path_to_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Relative paths are resolved to absolute."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(tmp_path)

        result = _resolve_project_dir("subdir")
        assert result.is_absolute()
        assert result == subdir.resolve()

    def test_no_existence_check_by_default(self) -> None:
        """Missing paths are allowed when validate_exists is not set."""
        result = _resolve_project_dir("/path/that/does/not/exist")
        assert str(result) == "/path/that/does/not/exist"

    def test_validate_exists_raises_for_missing_path(self) -> None:
        """validate_exists=True raises RasaException for a non-existent path."""
        with pytest.raises(RasaException, match="does not exist"):
            _resolve_project_dir("/path/that/does/not/exist", validate_exists=True)

    def test_validate_exists_raises_when_path_is_file(self, tmp_path: Path) -> None:
        """validate_exists=True raises RasaException when path is a file."""
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("content")

        with pytest.raises(RasaException, match="not a directory"):
            _resolve_project_dir(str(file_path), validate_exists=True)


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
            cli_project_path=str(tmp_path),
        )
        assert cfg.mode == expected_mode
        assert cfg.port == expected_port
        assert (source is not None) == has_source


class TestRunTools:
    """Test run_tools function."""

    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rasa.utils.licensing.validate_license_from_env", MagicMock()
        )

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
            project_path=str(tmp_path),
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
            project_path=str(tmp_path),
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

        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        RunConfig(mode="stdio").save(config_path)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project_path=str(tmp_path),
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
            project_path=None,
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
            project_path="/nonexistent/path",
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
        root_logger = logging.getLogger()
        stdout_handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(stdout_handler)

        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project_path=str(tmp_path),
            config=None,
        )

        run_tools(args)

        assert stdout_handler.stream == sys.stderr
        root_logger.removeHandler(stdout_handler)

    def test_logging_redirect_happens_before_precheck(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_run_server: MagicMock,
    ) -> None:
        """Logging must be redirected to stderr before _precheck runs.

        If _precheck (or validate_license_from_env) emits log messages before
        the redirect, they would reach stdout in stdio mode and corrupt the
        JSON-RPC stream.
        """
        call_order: list[str] = []

        original_redirect = _redirect_logging_to_stderr

        def tracking_redirect() -> None:
            call_order.append("redirect")
            original_redirect()

        monkeypatch.setattr(
            "rasa.cli.tools._redirect_logging_to_stderr", tracking_redirect
        )
        monkeypatch.setattr(
            "rasa.utils.licensing.validate_license_from_env",
            lambda: call_order.append("license"),
        )

        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project_path=str(tmp_path),
            config=None,
        )
        run_tools(args)

        assert call_order.index("redirect") < call_order.index("license")

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
            project_path=str(tmp_path),
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
            project_path=str(tmp_path),
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
            project_path=str(tmp_path),
            config=None,
        )

        run_tools(args)

        # Emit a structlog message after the redirect has been applied
        structlog.get_logger().info("test.structlog.after_redirect", key="value")

        captured = capsys.readouterr()
        assert (
            captured.out == ""
        ), f"structlog leaked to stdout in stdio mode: {captured.out!r}"

    # -- config display messages ---------------------------------------------------

    def test_stdio_stderr_on_loaded_config(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Stdio mode prints loaded config path and contents to stderr."""
        existing = RunConfig(mode="stdio", port=7331)
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        existing.save(config_path)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project_path=str(tmp_path),
            config=None,
        )

        run_tools(args)

        captured = capsys.readouterr()
        assert "Config loaded from:" in captured.err
        assert TOOLS_CONFIG_FILENAME in captured.err
        assert "mode: stdio" in captured.err

    def test_http_stdout_on_loaded_config(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """HTTP mode prints loaded config path and contents to stdout."""
        existing = RunConfig(mode="http", port=5555)
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        existing.save(config_path)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project_path=str(tmp_path),
            config=None,
        )

        run_tools(args)

        captured = capsys.readouterr()
        assert "Config loaded from:" in captured.out
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
            project_path=str(tmp_path),
            config=None,
        )

        with patch("pathlib.Path.read_text", side_effect=OSError("disk error")):
            run_tools(args)

        mock_run_server.assert_called_once()
        captured = capsys.readouterr()
        assert "Config loaded from:" in captured.out
        assert "<could not read file>" in captured.out


class TestRunConfigLoad:
    def test_raises_when_file_does_not_exist(self, tmp_path: Path) -> None:
        with pytest.raises(RasaException, match="does not exist"):
            RunConfig.load(tmp_path / "missing.yaml")

    def test_raises_on_invalid_config_content(self, tmp_path: Path) -> None:
        path = tmp_path / "tools.yaml"
        path.write_text("ide_integrations: [not-a-valid-ide]\n")
        with pytest.raises(RasaException, match="Invalid config"):
            RunConfig.load(path)

    def test_raises_on_unparseable_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "tools.yaml"
        path.write_text("key: [unclosed bracket\n")
        with pytest.raises(RasaException, match="Failed to read config file"):
            RunConfig.load(path)


class TestRunConfigSerializer:
    def test_ide_integrations_omitted_when_empty(self) -> None:
        data = RunConfig(mode="stdio", ide_integrations=[]).model_dump()
        assert "ide_integrations" not in data

    def test_ide_integrations_present_when_non_empty(self) -> None:
        data = RunConfig(mode="stdio", ide_integrations=["cursor"]).model_dump()
        assert data["ide_integrations"] == ["cursor"]

    def test_extra_fields_ignored_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "tools.yaml"
        path.write_text("mode: stdio\nunknown_future_field: some_value\n")
        cfg = RunConfig.load(path)
        assert cfg.mode == "stdio"
        assert not hasattr(cfg, "unknown_future_field")

    def test_port_omitted_for_stdio(self) -> None:
        data = RunConfig(mode="stdio").model_dump()
        assert "port" not in data

    def test_port_included_for_http(self) -> None:
        data = RunConfig(mode="http", port=9000).model_dump()
        assert data["port"] == 9000


class TestValidateConfigExclusivity:
    def _args(self, **kwargs: object) -> argparse.Namespace:
        defaults = dict(config=None, mode=None, port=None, project_path=None)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_no_config_always_passes(self) -> None:
        _validate_config_exclusivity(self._args(mode="stdio", port=9000))

    def test_config_alone_passes(self) -> None:
        _validate_config_exclusivity(self._args(config="/some/path"))

    def test_config_with_mode_raises(self) -> None:
        with pytest.raises(RasaException, match="--mode"):
            _validate_config_exclusivity(self._args(config="/p", mode="stdio"))

    def test_config_with_port_raises(self) -> None:
        with pytest.raises(RasaException, match="--port"):
            _validate_config_exclusivity(self._args(config="/p", port=9000))

    def test_config_with_project_path_raises(self) -> None:
        with pytest.raises(RasaException, match="--project-path"):
            _validate_config_exclusivity(self._args(config="/p", project_path="/x"))

    def test_config_with_all_three_raises_and_lists_all(self) -> None:
        with pytest.raises(RasaException, match="--mode") as exc_info:
            _validate_config_exclusivity(
                self._args(config="/p", mode="stdio", port=9000, project_path="/x")
            )
        msg = str(exc_info.value)
        assert "--port" in msg
        assert "--project-path" in msg


class TestRunToolsNoConfigGuard:
    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rasa.utils.licensing.validate_license_from_env", MagicMock()
        )

    def test_raises_when_no_config_and_no_explicit_args(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RasaException raised when there is no config file and no --mode/--port."""
        monkeypatch.delenv(RASA_PROJECT_FOLDER_ENV_VAR, raising=False)

        args = argparse.Namespace(
            mode=None,
            port=None,
            project_path=str(tmp_path),
            config=None,
        )

        with pytest.raises(RasaException, match="No configuration found"):
            run_tools(args)


class TestResolveConfigPath:
    def test_explicit_file_that_exists_is_returned(self, tmp_path: Path) -> None:
        cfg = tmp_path / "tools.yaml"
        cfg.write_text("mode: stdio\n")
        assert _resolve_config_path(str(cfg), tmp_path) == cfg.resolve()

    def test_explicit_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RasaException, match="does not exist"):
            _resolve_config_path(str(tmp_path / "ghost.yaml"), tmp_path)

    def test_explicit_dir_resolves_to_tools_yaml(self, tmp_path: Path) -> None:
        cfg = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        cfg.parent.mkdir()
        cfg.write_text("mode: stdio\n")
        assert _resolve_config_path(str(tmp_path), tmp_path) == cfg.resolve()

    def test_explicit_dot_rasa_dir_resolves_to_tools_yaml(self, tmp_path: Path) -> None:
        dot_rasa = tmp_path / TOOLS_CONFIG_DIR
        cfg = dot_rasa / TOOLS_CONFIG_FILENAME
        dot_rasa.mkdir()
        cfg.write_text("mode: stdio\n")
        assert _resolve_config_path(str(dot_rasa), tmp_path) == cfg.resolve()

    def test_no_cli_config_returns_default_when_file_exists(
        self, tmp_path: Path
    ) -> None:
        cfg = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        cfg.parent.mkdir()
        cfg.write_text("mode: stdio\n")
        assert _resolve_config_path(None, tmp_path) == cfg.resolve()

    def test_no_cli_config_returns_none_when_no_file(self, tmp_path: Path) -> None:
        assert _resolve_config_path(None, tmp_path) is None


class TestPrecheckStreamRouting:
    _VALIDATE = "rasa.utils.licensing.validate_license_from_env"

    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(self._VALIDATE, MagicMock())

    def test_banner_written_to_stderr_when_file_is_stderr(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import sys

        _precheck(file=sys.stderr)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert len(captured.err) > 0

    def test_banner_written_to_stdout_by_default(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _precheck()

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_run_tools_banner_never_reaches_stdout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """_precheck called from run_tools must not write the banner to stdout."""
        monkeypatch.setattr(
            "rasa.builder.copilot.mcp_server.server.run_server", MagicMock()
        )
        args = argparse.Namespace(
            mode="stdio",
            port=None,
            project_path=str(tmp_path),
            config=None,
        )

        run_tools(args)

        assert capsys.readouterr().out == ""


class TestRedirectLoggingToStderr:
    def test_handler_without_stream_attr_is_skipped(self) -> None:
        import logging

        handler = logging.Handler()
        assert not hasattr(handler, "stream")

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _redirect_logging_to_stderr()  # must not raise
        finally:
            root.removeHandler(handler)

    def test_non_stdout_handler_stream_is_not_changed(self) -> None:
        import logging
        import sys

        handler = logging.StreamHandler(sys.stderr)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _redirect_logging_to_stderr()
            assert handler.stream is sys.stderr
        finally:
            root.removeHandler(handler)
