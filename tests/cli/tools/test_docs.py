"""Tests for ``rasa tools init docs``."""

import argparse
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from rasa.cli.tools.constants import (
    LLMS_TXT_BASE_URL_ENV_VAR,
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.docs import (
    _confirm_overwrite_docs,
    _is_online_docs_configured,
    docs_tools,
    fetch_offline_docs,
    resolve_llms_txt_base_url,
    warn_if_offline_docs_exist,
    warn_if_online_docs_configured,
)


class TestResolveLlmsTxtBaseUrl:
    def test_uses_env_var_when_set(self, monkeypatch: Any) -> None:
        monkeypatch.setenv(LLMS_TXT_BASE_URL_ENV_VAR, "https://custom.example.com")
        assert resolve_llms_txt_base_url() == "https://custom.example.com"

    def test_falls_back_to_default_when_env_var_absent(self, monkeypatch: Any) -> None:
        monkeypatch.delenv(LLMS_TXT_BASE_URL_ENV_VAR, raising=False)
        assert resolve_llms_txt_base_url() == "https://rasa.com/docs"

    def test_strips_trailing_slash(self, monkeypatch: Any) -> None:
        monkeypatch.setenv(LLMS_TXT_BASE_URL_ENV_VAR, "https://custom.example.com/")
        assert resolve_llms_txt_base_url() == "https://custom.example.com"


class TestFetchOfflineDocs:
    @pytest.fixture()
    def fake_urlopen(self, monkeypatch: Any) -> None:
        """Patch urlopen to return ``b"content of docs"`` for every request."""

        def _fake(url: str, timeout: int) -> Any:
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(return_value=b"content of docs")
            return mock

        monkeypatch.setattr("urllib.request.urlopen", _fake)

    def test_creates_dest_dir_and_writes_files(
        self, tmp_path: Path, fake_urlopen: None
    ) -> None:
        fetch_offline_docs(tmp_path, non_interactive=True)

        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        assert (dest_dir / "llms.txt").exists()
        assert (dest_dir / "llms-full.txt").exists()
        assert "content of docs" in (dest_dir / "llms.txt").read_bytes().decode()

    def test_graceful_failure_does_not_raise(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            MagicMock(side_effect=OSError("network error")),
        )
        fetch_offline_docs(tmp_path, non_interactive=True)

    def test_partial_failure_continues_remaining_files(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        fetched_urls: list[str] = []

        def _fake(url: str, timeout: int) -> Any:
            fetched_urls.append(url)
            if "llms-full.txt" in url:
                raise OSError("timeout")
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(return_value=b"ok")
            return mock

        monkeypatch.setattr("urllib.request.urlopen", _fake)
        fetch_offline_docs(tmp_path, non_interactive=True)

        assert len(fetched_urls) == 2
        assert (tmp_path / TOOLS_CONFIG_DIR / "llms.txt").exists()
        assert not (tmp_path / TOOLS_CONFIG_DIR / "llms-full.txt").exists()

    def test_uses_custom_base_url(self, tmp_path: Path, monkeypatch: Any) -> None:
        fetched_urls: list[str] = []

        def _fake(url: str, timeout: int) -> Any:
            fetched_urls.append(url)
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(return_value=b"ok")
            return mock

        monkeypatch.setattr("urllib.request.urlopen", _fake)
        monkeypatch.setenv(LLMS_TXT_BASE_URL_ENV_VAR, "https://custom.example.com")
        fetch_offline_docs(tmp_path, non_interactive=True)

        assert all(u.startswith("https://custom.example.com/") for u in fetched_urls)

    def test_skips_download_when_user_declines_overwrite(
        self, tmp_path: Path, fake_urlopen: None, monkeypatch: Any
    ) -> None:
        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        dest_dir.mkdir(parents=True)
        (dest_dir / "llms.txt").write_text("old content")

        monkeypatch.setattr(
            "rasa.cli.tools.docs._confirm_overwrite_docs", MagicMock(return_value=False)
        )
        fetch_offline_docs(tmp_path)

        assert (dest_dir / "llms.txt").read_text() == "old content"

    def test_overwrites_when_user_confirms(
        self, tmp_path: Path, fake_urlopen: None, monkeypatch: Any
    ) -> None:
        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        dest_dir.mkdir(parents=True)
        (dest_dir / "llms.txt").write_text("old content")

        monkeypatch.setattr(
            "rasa.cli.tools.docs._confirm_overwrite_docs", MagicMock(return_value=True)
        )
        fetch_offline_docs(tmp_path)

        assert (dest_dir / "llms.txt").read_text() != "old content"

    def test_non_interactive_skips_prompt_and_overwrites(
        self, tmp_path: Path, fake_urlopen: None, monkeypatch: Any
    ) -> None:
        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        dest_dir.mkdir(parents=True)
        (dest_dir / "llms.txt").write_text("old content")

        mock_confirm = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.docs._confirm_overwrite_docs", mock_confirm)
        fetch_offline_docs(tmp_path, non_interactive=True)

        mock_confirm.assert_not_called()
        assert (dest_dir / "llms.txt").read_text() != "old content"


class TestConfirmOverwriteDocs:
    def _mock_confirm(self, monkeypatch: Any, answer: Any) -> None:
        monkeypatch.setattr(
            "questionary.confirm",
            MagicMock(return_value=MagicMock(ask=MagicMock(return_value=answer))),
        )

    def test_returns_true_when_user_confirms(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        self._mock_confirm(monkeypatch, True)
        assert _confirm_overwrite_docs(["llms.txt"], tmp_path) is True

    def test_returns_false_when_user_declines(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        self._mock_confirm(monkeypatch, False)
        assert _confirm_overwrite_docs(["llms.txt"], tmp_path) is False

    def test_exits_when_user_interrupts(self, tmp_path: Path, monkeypatch: Any) -> None:
        self._mock_confirm(monkeypatch, None)
        with pytest.raises(SystemExit):
            _confirm_overwrite_docs(["llms.txt"], tmp_path)


class TestIsOnlineDocsConfigured:
    def _write_config(self, tmp_path: Path, content: str) -> None:
        config_dir = tmp_path / TOOLS_CONFIG_DIR
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / TOOLS_CONFIG_FILENAME).write_text(content)

    def test_returns_false_when_no_config_file(self, tmp_path: Path) -> None:
        assert _is_online_docs_configured(tmp_path) is False

    def test_returns_false_when_offline_mode(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "rasa.cli.tools.models.RunConfig.load",
            MagicMock(return_value=MagicMock(docs_mode="offline")),
        )
        self._write_config(tmp_path, "docs_mode: offline\n")
        assert _is_online_docs_configured(tmp_path) is False

    def test_returns_true_when_online_mode(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "rasa.cli.tools.models.RunConfig.load",
            MagicMock(return_value=MagicMock(docs_mode="online")),
        )
        self._write_config(tmp_path, "docs_mode: online\n")
        assert _is_online_docs_configured(tmp_path) is True

    def test_returns_false_on_corrupt_config(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, "not: valid: yaml: [")
        assert _is_online_docs_configured(tmp_path) is False


class TestWarnIfOnlineDocsConfigured:
    def test_no_output_when_offline_mode(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "rasa.cli.tools.docs._is_online_docs_configured",
            MagicMock(return_value=False),
        )
        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools.docs.console", wide)
        warn_if_online_docs_configured(tmp_path)

        assert buf.getvalue() == ""

    def test_warns_when_online_mode(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "rasa.cli.tools.docs._is_online_docs_configured",
            MagicMock(return_value=True),
        )
        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools.docs.console", wide)
        warn_if_online_docs_configured(tmp_path)

        output = buf.getvalue()
        assert "Online documentation mode" in output
        assert "Disable the MCP tool" in output


class TestWarnIfOfflineDocsExist:
    def test_no_warning_when_no_files_present(self, tmp_path: Path) -> None:
        warn_if_offline_docs_exist(tmp_path)

    def test_warns_when_llms_txt_exists(self, tmp_path: Path, monkeypatch: Any) -> None:
        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        dest_dir.mkdir(parents=True)
        (dest_dir / "llms.txt").write_text("docs")

        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools.docs.console", wide)
        warn_if_offline_docs_exist(tmp_path)

        assert "llms.txt" in buf.getvalue()

    def test_warns_for_each_present_file(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        dest_dir.mkdir(parents=True)
        (dest_dir / "llms.txt").write_text("docs")
        (dest_dir / "llms-full.txt").write_text("full docs")

        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools.docs.console", wide)
        warn_if_offline_docs_exist(tmp_path)

        output = buf.getvalue()
        assert "llms.txt" in output
        assert "llms-full.txt" in output


class TestDocsToolsEntrypoint:
    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "rasa.utils.licensing.validate_license_from_env", MagicMock()
        )

    @pytest.fixture()
    def mock_fetch(self, monkeypatch: Any) -> MagicMock:
        mock = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.docs.fetch_offline_docs", mock)
        return mock

    def test_fetches_docs_for_project(
        self, tmp_path: Path, mock_fetch: MagicMock
    ) -> None:
        args = argparse.Namespace(project_path=str(tmp_path), yes=False)
        docs_tools(args)
        mock_fetch.assert_called_once_with(tmp_path, non_interactive=False)

    def test_defaults_to_cwd_when_no_project_path(
        self, tmp_path: Path, monkeypatch: Any, mock_fetch: MagicMock
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RASA_PROJECT_FOLDER", raising=False)
        args = argparse.Namespace(project_path=None, yes=False)
        docs_tools(args)
        mock_fetch.assert_called_once_with(tmp_path, non_interactive=False)

    def test_yes_flag_passes_non_interactive(
        self, tmp_path: Path, mock_fetch: MagicMock
    ) -> None:
        args = argparse.Namespace(project_path=str(tmp_path), yes=True)
        docs_tools(args)
        mock_fetch.assert_called_once_with(tmp_path, non_interactive=True)

    def test_calls_warn_if_online_docs_configured(
        self, tmp_path: Path, mock_fetch: MagicMock, monkeypatch: Any
    ) -> None:
        mock_warn = MagicMock()
        monkeypatch.setattr(
            "rasa.cli.tools.docs.warn_if_online_docs_configured",
            mock_warn,
        )
        args = argparse.Namespace(project_path=str(tmp_path), yes=False)
        docs_tools(args)

        mock_fetch.assert_called_once()
        mock_warn.assert_called_once_with(tmp_path)
