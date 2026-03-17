"""Tests for ``rasa.cli.tools.utils``."""

import logging
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rasa.builder.copilot.constants import RASA_PROJECT_FOLDER_ENV_VAR
from rasa.cli.tools.constants import TOOLS_CONFIG_DIR, TOOLS_CONFIG_FILENAME
from rasa.cli.tools.models import RunConfig
from rasa.cli.tools.utils import (
    _load_project_dotenv,
    _precheck,
    _redirect_logging_to_stderr,
    _resolve_ides,
    _resolve_project_dir,
    restore_blocking_io,
)
from rasa.shared.exceptions import RasaException

_VALIDATE_LICENSE = "rasa.utils.licensing.validate_license_from_env"


class TestResolveProjectDir:
    """Tests for _resolve_project_dir."""

    def test_cli_arg_takes_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(env_dir))

        assert _resolve_project_dir(str(cli_dir)) == cli_dir.resolve()

    def test_env_var_used_when_no_cli_arg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(RASA_PROJECT_FOLDER_ENV_VAR, str(tmp_path))

        assert _resolve_project_dir() == tmp_path.resolve()

    def test_config_project_path_used_when_no_cli_or_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        monkeypatch.delenv(RASA_PROJECT_FOLDER_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)

        assert _resolve_project_dir() == tmp_path.resolve()

    def test_resolves_relative_path_to_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(tmp_path)

        result = _resolve_project_dir("subdir")
        assert result.is_absolute()
        assert result == subdir.resolve()

    def test_no_existence_check_by_default(self) -> None:
        result = _resolve_project_dir("/path/that/does/not/exist")
        assert str(result) == "/path/that/does/not/exist"

    def test_validate_exists_raises_for_missing_path(self) -> None:
        with pytest.raises(RasaException, match="does not exist"):
            _resolve_project_dir("/path/that/does/not/exist", validate_exists=True)

    def test_validate_exists_raises_when_path_is_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("content")

        with pytest.raises(RasaException, match="not a directory"):
            _resolve_project_dir(str(file_path), validate_exists=True)


class TestResolveIdes:
    def test_cli_ides_take_priority(self, tmp_path: Path) -> None:
        """--ides CLI arg should win over saved config."""
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config_path.parent.mkdir(parents=True)
        RunConfig(ide_integrations=["vscode"]).save(config_path)

        result = _resolve_ides("cursor", tmp_path)
        assert result == ["cursor"]

    def test_falls_back_to_saved_config(self, tmp_path: Path) -> None:
        """When --ides is absent, reads from saved config."""
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config_path.parent.mkdir(parents=True)
        RunConfig(ide_integrations=["cursor", "vscode"]).save(config_path)

        result = _resolve_ides(None, tmp_path)
        assert result == ["cursor", "vscode"]

    def test_returns_empty_when_no_config(self, tmp_path: Path) -> None:
        """When no --ides and no config file, returns empty list."""
        result = _resolve_ides(None, tmp_path)
        assert result == []

    def test_handles_comma_separated_ides(self, tmp_path: Path) -> None:
        result = _resolve_ides(" cursor , vscode , claude ", tmp_path)
        assert result == ["cursor", "vscode", "claude"]

    def test_normalises_case(self, tmp_path: Path) -> None:
        result = _resolve_ides("Cursor,VSCODE", tmp_path)
        assert result == ["cursor", "vscode"]


class TestPrecheckLicenseEnforcement:
    def test_passes_with_valid_license(self, monkeypatch: Any) -> None:
        """_precheck should complete without error when the license is valid."""
        mock_validate = MagicMock()
        monkeypatch.setattr(_VALIDATE_LICENSE, mock_validate)
        _precheck()
        mock_validate.assert_called_once()

    @pytest.mark.parametrize(
        "message",
        [
            "A Rasa license is required.",
            "Failed to validate Rasa license.",
        ],
        ids=["missing", "invalid"],
    )
    def test_exits_on_license_error(self, monkeypatch: Any, message: str) -> None:
        """_precheck should propagate SystemExit for any license failure."""
        monkeypatch.setattr(
            _VALIDATE_LICENSE,
            MagicMock(side_effect=SystemExit(message)),
        )
        with pytest.raises(SystemExit):
            _precheck()


class TestPrecheckStreamRouting:
    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_VALIDATE_LICENSE, MagicMock())

    def test_banner_written_to_stderr_when_file_is_stderr(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
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


class TestRestoreBlockingIo:
    def test_restores_blocking_after_nonblocking(self) -> None:
        """Stdout must be blocking after restore, even if set non-blocking."""
        fd = sys.stdout.fileno()
        original = os.get_blocking(fd)
        try:
            os.set_blocking(fd, False)
            assert not os.get_blocking(fd)

            restore_blocking_io()
            assert os.get_blocking(fd)
        finally:
            os.set_blocking(fd, original)

    def test_noop_when_already_blocking(self) -> None:
        """Calling restore when fds are already blocking must not raise."""
        assert os.get_blocking(sys.stdout.fileno())
        restore_blocking_io()
        assert os.get_blocking(sys.stdout.fileno())

    def test_handles_stream_without_fileno(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Must not raise when stdout lacks a real file descriptor."""
        monkeypatch.setattr(sys, "stdout", StringIO())
        restore_blocking_io()


class TestLoadProjectDotenv:
    """_load_project_dotenv must load the project's .env into os.environ."""

    def test_loads_license_from_dotenv(self, tmp_path: Path, monkeypatch: Any) -> None:
        """License in .env must become available in os.environ."""
        env_file = tmp_path / ".env"
        env_file.write_text("RASA_LICENSE=license-from-dotenv\n")
        monkeypatch.delenv("RASA_LICENSE", raising=False)
        monkeypatch.delenv("RASA_PRO_LICENSE", raising=False)

        _load_project_dotenv(str(tmp_path))

        assert os.environ.get("RASA_LICENSE") == "license-from-dotenv"

    def test_does_not_override_existing_env(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """An explicit env var must take precedence over .env."""
        env_file = tmp_path / ".env"
        env_file.write_text("RASA_LICENSE=from-dotenv\n")
        monkeypatch.setenv("RASA_LICENSE", "from-environment")

        _load_project_dotenv(str(tmp_path))

        assert os.environ.get("RASA_LICENSE") == "from-environment"

    def test_no_dotenv_file_is_harmless(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Missing .env must not raise."""
        monkeypatch.delenv("RASA_LICENSE", raising=False)
        _load_project_dotenv(str(tmp_path))


class TestRedirectLoggingToStderr:
    def test_handler_without_stream_attr_is_skipped(self) -> None:
        handler = logging.Handler()
        assert not hasattr(handler, "stream")

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _redirect_logging_to_stderr()
        finally:
            root.removeHandler(handler)

    def test_non_stdout_handler_stream_is_not_changed(self) -> None:
        handler = logging.StreamHandler(sys.stderr)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            _redirect_logging_to_stderr()
            assert handler.stream is sys.stderr
        finally:
            root.removeHandler(handler)
