import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rasa.cli.tools.constants import (
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.status import (
    _check_reachability,
    _print_no_config,
    _print_status,
    status_tools,
)
from rasa.cli.tools.utils import RunConfig


def _strip_panel_wrapping(text: str) -> str:
    """Collapse Rich panel borders, line wraps, and extra whitespace."""
    import re

    text = text.replace("│", "").replace("\n", " ")
    return re.sub(r"\s+", " ", text)


class TestStatusTools:
    """Tests for the `rasa tools status` entrypoint."""

    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "rasa.utils.licensing.validate_license_from_env", MagicMock()
        )

    def test_status_with_existing_config(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """All config fields appear in output when config file exists."""
        config = RunConfig(
            mode="http",
            port=9999,
            project_path=str(tmp_path),
            docs_mode="online",
            ide_integrations=["cursor", "vscode"],
            rasa_server_url="http://my-server:5005",
        )
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config.save(config_path)

        args = argparse.Namespace(project_path=str(tmp_path))

        with patch("rasa.cli.tools.status._check_reachability", return_value=False):
            status_tools(args)

        captured = capsys.readouterr()
        flat = _strip_panel_wrapping(captured.out)
        assert tmp_path.name in flat
        assert "http" in flat
        assert "9999" in flat
        assert "online" in flat
        assert "http://my-server:5005" in flat
        assert "Cursor" in flat
        assert "VS Code" in flat

    def test_status_without_config(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Warning message shown when config file is missing."""
        args = argparse.Namespace(project_path=str(tmp_path))

        status_tools(args)

        captured = capsys.readouterr()
        flat = _strip_panel_wrapping(captured.out)
        assert tmp_path.name in flat
        assert "NOT FOUND" in flat
        assert "rasa tools init" in flat

    def test_status_shows_reachable_server(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Reachability check result is displayed when server is reachable."""
        config = RunConfig(mode="stdio", project_path=str(tmp_path))
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config.save(config_path)

        args = argparse.Namespace(project_path=str(tmp_path))

        with patch("rasa.cli.tools.status._check_reachability", return_value=True):
            status_tools(args)

        captured = capsys.readouterr()
        assert "reachable" in captured.out

    def test_status_shows_unreachable_server(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Reachability check result is displayed when server is unreachable."""
        config = RunConfig(mode="stdio", project_path=str(tmp_path))
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config.save(config_path)

        args = argparse.Namespace(project_path=str(tmp_path))

        with patch("rasa.cli.tools.status._check_reachability", return_value=False):
            status_tools(args)

        captured = capsys.readouterr()
        assert "unreachable" in captured.out

    def test_status_no_ide_integrations(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When no IDEs are configured, output says 'none configured'."""
        config = RunConfig(
            mode="stdio",
            project_path=str(tmp_path),
            ide_integrations=[],
        )
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config.save(config_path)

        args = argparse.Namespace(project_path=str(tmp_path))

        with patch("rasa.cli.tools.status._check_reachability", return_value=False):
            status_tools(args)

        captured = capsys.readouterr()
        assert "none configured" in captured.out

    def test_status_port_only_shown_for_http(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Port line is omitted in stdio mode."""
        config = RunConfig(mode="stdio", project_path=str(tmp_path))
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config.save(config_path)

        args = argparse.Namespace(project_path=str(tmp_path))

        with patch("rasa.cli.tools.status._check_reachability", return_value=False):
            status_tools(args)

        captured = capsys.readouterr()
        assert "Port" not in captured.out


class TestPrintStatus:
    """Unit tests for _print_status output formatting."""

    def test_http_mode_includes_port(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = RunConfig(mode="http", port=8080, project_path=str(tmp_path))
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME

        with patch("rasa.cli.tools.status._check_reachability", return_value=False):
            _print_status(tmp_path, config_path, config)

        captured = capsys.readouterr()
        assert "8080" in captured.out
        assert "Port" in captured.out

    def test_default_rasa_server_url_shows_default_annotation(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When rasa_server_url is the default, '(default)' is shown."""
        config = RunConfig(mode="stdio", project_path=str(tmp_path))
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME

        with patch("rasa.cli.tools.status._check_reachability", return_value=False):
            _print_status(tmp_path, config_path, config)

        captured = capsys.readouterr()
        assert "(default)" in captured.out

    def test_custom_rasa_server_url_omits_default_annotation(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When rasa_server_url differs from the default, no '(default)' shown."""
        config = RunConfig(
            mode="stdio",
            project_path=str(tmp_path),
            rasa_server_url="http://custom:9999",
        )
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME

        with patch("rasa.cli.tools.status._check_reachability", return_value=False):
            _print_status(tmp_path, config_path, config)

        captured = capsys.readouterr()
        assert "(default)" not in captured.out
        assert "http://custom:9999" in captured.out


class TestPrintNoConfig:
    """Unit tests for _print_no_config output."""

    def test_includes_project_path_and_guidance(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        _print_no_config(tmp_path, config_path)

        captured = capsys.readouterr()
        flat = _strip_panel_wrapping(captured.out)
        assert tmp_path.name in flat
        assert "NOT FOUND" in flat
        assert "rasa tools init" in flat


class TestCheckReachability:
    """Tests for the best-effort HTTP reachability check."""

    def test_reachable(self) -> None:
        with patch("rasa.cli.tools.status.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            assert _check_reachability("http://localhost:5005") is True

    def test_unreachable_url_error(self) -> None:
        from urllib.error import URLError

        with patch(
            "rasa.cli.tools.status.urlopen",
            side_effect=URLError("Connection refused"),
        ):
            assert _check_reachability("http://localhost:5005") is False

    def test_unreachable_os_error(self) -> None:
        with patch(
            "rasa.cli.tools.status.urlopen",
            side_effect=OSError("Network error"),
        ):
            assert _check_reachability("http://localhost:5005") is False

    def test_unreachable_value_error(self) -> None:
        with patch(
            "rasa.cli.tools.status.urlopen",
            side_effect=ValueError("Invalid URL"),
        ):
            assert _check_reachability("not-a-url") is False

    def test_reachable_on_http_error(self) -> None:
        """Server returning 404/500 is still reachable."""
        from urllib.error import HTTPError

        with patch(
            "rasa.cli.tools.status.urlopen",
            side_effect=HTTPError("http://localhost:5005", 404, "Not Found", {}, None),
        ):
            assert _check_reachability("http://localhost:5005") is True
