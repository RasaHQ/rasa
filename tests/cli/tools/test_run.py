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
from rasa.cli.tools.constants import (
    DEFAULT_RASA_SERVER_URL,
    MCP_TOOLS_DEFAULT_HOST,
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.run import (
    _resolve_config_path,
    _resolve_tools_run_config,
    _validate_config_exclusivity,
    run_tools,
)
from rasa.cli.tools.utils import (
    RunConfig,
    _redirect_logging_to_stderr,
)
from rasa.shared.exceptions import RasaException


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
            rasa_server_url=DEFAULT_RASA_SERVER_URL,
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
            rasa_server_url=DEFAULT_RASA_SERVER_URL,
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
            rasa_server_url=DEFAULT_RASA_SERVER_URL,
        )

    def test_passes_rasa_server_url_to_run_server(
        self,
        tmp_path: Path,
        mock_run_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a custom rasa_server_url is passed through to run_server."""
        monkeypatch.delenv(RASA_PROJECT_FOLDER_ENV_VAR, raising=False)

        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        RunConfig(mode="stdio").save(config_path)

        custom_url = "http://my-server:9999"
        args = argparse.Namespace(
            mode=None,
            port=None,
            project_path=str(tmp_path),
            config=None,
            rasa_server_url=custom_url,
        )

        run_tools(args)

        mock_run_server.assert_called_once_with(
            transport=MCP_TRANSPORT_STDIO,
            project_folder=str(tmp_path.resolve()),
            rasa_server_url=custom_url,
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
            "rasa.cli.tools.run._redirect_logging_to_stderr", tracking_redirect
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


class TestValidateConfigExclusivity:
    def _args(self, **kwargs: object) -> argparse.Namespace:
        defaults = dict(
            config=None, mode=None, port=None, project_path=None, rasa_server_url=None
        )
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


class TestRunToolsBannerRouting:
    _VALIDATE = "rasa.utils.licensing.validate_license_from_env"

    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(self._VALIDATE, MagicMock())

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
