import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from rich.console import Console

from rasa.cli.arguments.tools import (
    DOCS_MODE_OFFLINE,
    DOCS_MODE_ONLINE,
    MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR,
    MCP_TOOLS_TRANSPORT_HTTP,
    MCP_TOOLS_TRANSPORT_STDIO,
)
from rasa.cli.tools import (
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
    RunConfig,
    _precheck,
    init_tools,
    run_tools,
)
from rasa.cli.tools_wizard import (
    _HELLO_LLM_PROXY_URL,
    LLMS_TXT_BASE_URL_ENV_VAR,
    _build_http_entry,
    _build_stdio_entry,
    _confirm_overwrite,
    _fetch_offline_docs,
    _fetch_skill_names,
    _install_agent_skills,
    _merge_mcp_json,
    _print_summary,
    _resolve_llms_txt_base_url,
    _run_non_interactive,
    _warn_if_agent_skills_exist,
    _warn_if_offline_docs_exist,
    _write_ide_configs,
    run_wizard,
)

_FAKE_LICENSE = "fake-license-token-for-tests"
_RETRIEVE_LICENSE = "rasa.utils.licensing.retrieve_license_from_env"


def _make_args(**overrides: Any) -> argparse.Namespace:
    defaults = dict(
        yes=True,
        project_path=None,
        config=None,
        mode=None,
        port=None,
        docs=None,
        ides=None,
        skills=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestResolveLlmsTxtBaseUrl:
    def test_uses_env_var_when_set(self, monkeypatch: Any) -> None:
        monkeypatch.setenv(LLMS_TXT_BASE_URL_ENV_VAR, "https://custom.example.com")
        assert _resolve_llms_txt_base_url() == "https://custom.example.com"

    def test_falls_back_to_default_when_env_var_absent(self, monkeypatch: Any) -> None:
        monkeypatch.delenv(LLMS_TXT_BASE_URL_ENV_VAR, raising=False)
        assert _resolve_llms_txt_base_url() == "https://rasa.com/docs"

    def test_strips_trailing_slash(self, monkeypatch: Any) -> None:
        monkeypatch.setenv(LLMS_TXT_BASE_URL_ENV_VAR, "https://custom.example.com/")
        assert _resolve_llms_txt_base_url() == "https://custom.example.com"


class TestNonInteractive:
    def test_all_defaults(self, tmp_path: Path) -> None:
        args = _make_args()
        cfg = _run_non_interactive(args, tmp_path)

        assert cfg.mode == MCP_TOOLS_TRANSPORT_STDIO
        assert cfg.port == 7331
        assert cfg.project_path == str(tmp_path)
        assert cfg.docs_mode == DOCS_MODE_OFFLINE
        assert cfg.ide_integrations == []

    def test_explicit_values(self, tmp_path: Path) -> None:
        args = _make_args(
            mode=MCP_TOOLS_TRANSPORT_HTTP,
            port=9000,
            docs=DOCS_MODE_ONLINE,
            ides="cursor,vscode,claude",
        )
        cfg = _run_non_interactive(args, tmp_path)

        assert cfg.mode == MCP_TOOLS_TRANSPORT_HTTP
        assert cfg.port == 9000
        assert cfg.docs_mode == DOCS_MODE_ONLINE
        assert cfg.ide_integrations == ["cursor", "vscode", "claude"]

    def test_ides_whitespace_handling(self, tmp_path: Path) -> None:
        args = _make_args(ides=" cursor , vscode , ")
        cfg = _run_non_interactive(args, tmp_path)

        assert cfg.ide_integrations == ["cursor", "vscode"]

    def test_ides_case_normalisation(self, tmp_path: Path) -> None:
        args = _make_args(ides="Cursor,VSCODE")
        cfg = _run_non_interactive(args, tmp_path)

        assert cfg.ide_integrations == ["cursor", "vscode"]


class TestMcpEntryBuilders:
    def test_stdio_entry(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))
        entry = _build_stdio_entry(tmp_path)
        assert entry["command"] == sys.executable
        assert entry["args"] == [
            "-m",
            "rasa",
            "tools",
            "run",
            "--mode",
            "stdio",
            "--project-path",
            str(tmp_path),
        ]

    def test_stdio_entry_includes_license_env(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))
        entry = _build_stdio_entry(tmp_path)
        assert entry["env"]["RASA_LICENSE"] == _FAKE_LICENSE

    def test_stdio_entry_includes_hello_llm_proxy_url(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))
        entry = _build_stdio_entry(tmp_path)
        assert entry["env"]["HELLO_LLM_PROXY_BASE_URL"] == _HELLO_LLM_PROXY_URL

    def test_http_entry(self) -> None:
        entry = _build_http_entry(9000)
        assert "9000" in entry["url"]
        assert entry["url"].startswith("http://")

    def test_http_entry_has_no_env(self) -> None:
        entry = _build_http_entry(9000)
        assert "env" not in entry


class TestMergeMcpJson:
    def test_creates_new_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "mcp.json"
        _merge_mcp_json(path, {"command": "rasa"}, wrapper_key="mcpServers")

        data = json.loads(path.read_text())
        assert data == {"mcpServers": {"rasa-tools": {"command": "rasa"}}}

    def test_preserves_existing_servers(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        existing = {"mcpServers": {"other-tool": {"command": "other"}}}
        path.write_text(json.dumps(existing))

        _merge_mcp_json(path, {"command": "rasa"}, wrapper_key="mcpServers")

        data = json.loads(path.read_text())
        assert "other-tool" in data["mcpServers"]
        assert data["mcpServers"]["rasa-tools"] == {"command": "rasa"}

    def test_overwrites_existing_rasa_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        existing = {"mcpServers": {"rasa-tools": {"command": "old"}}}
        path.write_text(json.dumps(existing))

        _merge_mcp_json(path, {"command": "new"}, wrapper_key="mcpServers")

        data = json.loads(path.read_text())
        assert data["mcpServers"]["rasa-tools"] == {"command": "new"}

    def test_handles_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        path.write_text("NOT JSON {{{")

        _merge_mcp_json(path, {"command": "rasa"}, wrapper_key="mcpServers")

        data = json.loads(path.read_text())
        assert data == {"mcpServers": {"rasa-tools": {"command": "rasa"}}}

    def test_handles_non_dict_wrapper_key(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"mcpServers": ["not", "a", "dict"]}))

        _merge_mcp_json(path, {"command": "rasa"}, wrapper_key="mcpServers")

        data = json.loads(path.read_text())
        assert data == {"mcpServers": {"rasa-tools": {"command": "rasa"}}}

    def test_handles_top_level_json_array(self, tmp_path: Path) -> None:
        """A file whose top-level JSON value is an array must not raise."""
        path = tmp_path / "mcp.json"
        path.write_text("[]")

        _merge_mcp_json(path, {"command": "rasa"}, wrapper_key="mcpServers")

        data = json.loads(path.read_text())
        assert data == {"mcpServers": {"rasa-tools": {"command": "rasa"}}}

    def test_handles_top_level_json_scalar(self, tmp_path: Path) -> None:
        """A file whose top-level JSON value is a scalar must not raise."""
        path = tmp_path / "mcp.json"
        path.write_text("42")

        _merge_mcp_json(path, {"command": "rasa"}, wrapper_key="mcpServers")

        data = json.loads(path.read_text())
        assert data == {"mcpServers": {"rasa-tools": {"command": "rasa"}}}

    def test_vscode_uses_servers_key(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        _merge_mcp_json(
            path, {"type": "stdio", "command": "rasa"}, wrapper_key="servers"
        )

        data = json.loads(path.read_text())
        assert "servers" in data
        assert "rasa-tools" in data["servers"]


class TestRunWizardNonInteractive:
    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))

    def test_creates_config_file(self, tmp_path: Path) -> None:
        args = _make_args(project_path=str(tmp_path))
        run_wizard(args)

        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        assert config_path.exists()

        loaded = RunConfig.load(config_path)
        assert loaded.mode == MCP_TOOLS_TRANSPORT_STDIO

    def test_creates_cursor_config(self, tmp_path: Path) -> None:
        args = _make_args(project_path=str(tmp_path), ides="cursor")
        run_wizard(args)

        cursor_cfg = tmp_path / ".cursor" / "mcp.json"
        assert cursor_cfg.exists()
        data = json.loads(cursor_cfg.read_text())
        assert "rasa-tools" in data["mcpServers"]

    def test_creates_vscode_config(self, tmp_path: Path) -> None:
        args = _make_args(project_path=str(tmp_path), ides="vscode")
        run_wizard(args)

        vscode_cfg = tmp_path / ".vscode" / "mcp.json"
        assert vscode_cfg.exists()
        data = json.loads(vscode_cfg.read_text())
        assert "rasa-tools" in data["servers"]
        assert data["servers"]["rasa-tools"]["type"] == MCP_TOOLS_TRANSPORT_STDIO

    def test_creates_claude_config(self, tmp_path: Path) -> None:
        args = _make_args(project_path=str(tmp_path), ides="claude")
        run_wizard(args)

        claude_cfg = tmp_path / ".mcp.json"
        assert claude_cfg.exists()
        data = json.loads(claude_cfg.read_text())
        assert "rasa-tools" in data["mcpServers"]

    def test_creates_multiple_ide_configs(self, tmp_path: Path) -> None:
        args = _make_args(project_path=str(tmp_path), ides="cursor,vscode,claude")
        run_wizard(args)

        assert (tmp_path / ".cursor" / "mcp.json").exists()
        assert (tmp_path / ".vscode" / "mcp.json").exists()
        assert (tmp_path / ".mcp.json").exists()

    def test_http_mode_config(self, tmp_path: Path) -> None:
        args = _make_args(
            project_path=str(tmp_path),
            mode=MCP_TOOLS_TRANSPORT_HTTP,
            port=9000,
            ides="cursor",
        )
        run_wizard(args)

        cursor_cfg = tmp_path / ".cursor" / "mcp.json"
        data = json.loads(cursor_cfg.read_text())
        entry = data["mcpServers"]["rasa-tools"]
        assert "url" in entry
        assert "9000" in entry["url"]

    def test_vscode_http_mode_config(self, tmp_path: Path) -> None:
        """VS Code HTTP entry must include a 'type' field alongside the url."""
        args = _make_args(
            project_path=str(tmp_path),
            mode=MCP_TOOLS_TRANSPORT_HTTP,
            port=9000,
            ides="vscode",
        )
        run_wizard(args)

        data = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        entry = data["servers"]["rasa-tools"]
        assert entry["type"] == "http"
        assert "9000" in entry["url"]

    def test_claude_http_mode_config(self, tmp_path: Path) -> None:
        """Claude Code HTTP entry must include a 'type' field alongside the url."""
        args = _make_args(
            project_path=str(tmp_path),
            mode=MCP_TOOLS_TRANSPORT_HTTP,
            port=9000,
            ides="claude",
        )
        run_wizard(args)

        data = json.loads((tmp_path / ".mcp.json").read_text())
        entry = data["mcpServers"]["rasa-tools"]
        assert entry["type"] == "http"
        assert "9000" in entry["url"]

    def test_overwrite_existing_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config_path.parent.mkdir(parents=True)
        config_path.write_text(f"mode: {MCP_TOOLS_TRANSPORT_HTTP}\nport: 1111\n")

        args = _make_args(project_path=str(tmp_path), mode=MCP_TOOLS_TRANSPORT_STDIO)
        run_wizard(args)

        loaded = RunConfig.load(config_path)
        assert loaded.mode == MCP_TOOLS_TRANSPORT_STDIO

    def test_offline_docs_called(self, tmp_path: Path, monkeypatch: Any) -> None:
        mock_fetch = MagicMock()
        monkeypatch.setattr("rasa.cli.tools_wizard._fetch_offline_docs", mock_fetch)
        args = _make_args(project_path=str(tmp_path), docs=DOCS_MODE_OFFLINE)
        run_wizard(args)

        mock_fetch.assert_called_once_with(tmp_path)

    def test_online_docs_skips_fetch(self, tmp_path: Path, monkeypatch: Any) -> None:
        mock_fetch = MagicMock()
        monkeypatch.setattr("rasa.cli.tools_wizard._fetch_offline_docs", mock_fetch)
        args = _make_args(project_path=str(tmp_path), docs=DOCS_MODE_ONLINE)
        run_wizard(args)

        mock_fetch.assert_not_called()

    def test_stdio_mode_excludes_port_from_config(self, tmp_path: Path) -> None:
        """Port should not be saved in config when using stdio mode."""
        args = _make_args(project_path=str(tmp_path), mode=MCP_TOOLS_TRANSPORT_STDIO)
        run_wizard(args)

        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config_text = config_path.read_text()
        assert "port:" not in config_text

    def test_http_mode_includes_port_in_config(self, tmp_path: Path) -> None:
        """Port should be saved in config when using http mode."""
        args = _make_args(
            project_path=str(tmp_path), mode=MCP_TOOLS_TRANSPORT_HTTP, port=9000
        )
        run_wizard(args)

        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config_text = config_path.read_text()
        assert "port: 9000" in config_text

    def test_online_mode_warns_if_offline_files_exist(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Switching to online mode should warn if offline files exist."""
        mock_warn = MagicMock()
        monkeypatch.setattr(
            "rasa.cli.tools_wizard._warn_if_offline_docs_exist", mock_warn
        )
        args = _make_args(project_path=str(tmp_path), docs=DOCS_MODE_ONLINE)
        run_wizard(args)

        mock_warn.assert_called_once_with(tmp_path)

    def test_creates_jetbrains_config(self, tmp_path: Path) -> None:
        """JetBrains IDE config should be created (prints instructions)."""
        args = _make_args(project_path=str(tmp_path), ides="jetbrains")
        run_wizard(args)

        # JetBrains doesn't create a file, just prints instructions
        # This test mainly ensures no errors occur
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        assert config_path.exists()

    def test_stdio_cursor_config_includes_env(self, tmp_path: Path) -> None:
        """Cursor stdio config must include license and proxy env vars."""
        args = _make_args(project_path=str(tmp_path), ides="cursor")
        run_wizard(args)

        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        env = data["mcpServers"]["rasa-tools"]["env"]
        assert env["RASA_LICENSE"] == _FAKE_LICENSE
        assert env["HELLO_LLM_PROXY_BASE_URL"] == _HELLO_LLM_PROXY_URL

    def test_stdio_vscode_config_includes_env(self, tmp_path: Path) -> None:
        """VS Code stdio config must include license and proxy env vars."""
        args = _make_args(project_path=str(tmp_path), ides="vscode")
        run_wizard(args)

        data = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        env = data["servers"]["rasa-tools"]["env"]
        assert env["RASA_LICENSE"] == _FAKE_LICENSE
        assert env["HELLO_LLM_PROXY_BASE_URL"] == _HELLO_LLM_PROXY_URL

    def test_stdio_claude_config_includes_env(self, tmp_path: Path) -> None:
        """Claude stdio config must include license and proxy env vars."""
        args = _make_args(project_path=str(tmp_path), ides="claude")
        run_wizard(args)

        data = json.loads((tmp_path / ".mcp.json").read_text())
        env = data["mcpServers"]["rasa-tools"]["env"]
        assert env["RASA_LICENSE"] == _FAKE_LICENSE
        assert env["HELLO_LLM_PROXY_BASE_URL"] == _HELLO_LLM_PROXY_URL

    def test_http_config_has_no_env(self, tmp_path: Path) -> None:
        """HTTP mode config entries should not contain an env block."""
        args = _make_args(
            project_path=str(tmp_path),
            mode=MCP_TOOLS_TRANSPORT_HTTP,
            port=9000,
            ides="cursor",
        )
        run_wizard(args)

        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert "env" not in data["mcpServers"]["rasa-tools"]

    def test_env_var_project_path_used_when_no_cli_arg(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """RASA_PROJECT_FOLDER env var must be used when --project-path is absent."""
        monkeypatch.setenv(MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR, str(tmp_path))
        args = _make_args(project_path=None)
        run_wizard(args)

        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        assert config_path.exists()

    def test_cli_project_path_overrides_env_var(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """--project-path must win over RASA_PROJECT_FOLDER."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        monkeypatch.setenv(MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR, str(other_dir))

        args = _make_args(project_path=str(tmp_path))
        run_wizard(args)

        assert (tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME).exists()
        assert not (other_dir / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME).exists()


class TestAgentSkills:
    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))

    FAKE_SKILLS: ClassVar[list[str]] = [
        "rasa-building-flows",
        "rasa-writing-custom-actions",
    ]

    def _fake_urlopen(self, url: str, timeout: int = 30) -> MagicMock:
        if "api.github.com" in url:
            entries = [{"name": s, "type": "dir"} for s in self.FAKE_SKILLS]
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(return_value=json.dumps(entries).encode())
            return mock
        # Raw SKILL.md fetch
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(
            return_value=b"---\nname: rasa-skill\n---\n# Skill content"
        )
        return mock

    def test_cursor_skill_files_created(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Each skill should be written as SKILL.md under .cursor/skills/<skill>/."""
        mock_urlopen = MagicMock(side_effect=self._fake_urlopen)
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        _install_agent_skills(tmp_path, ["cursor"])

        for skill in self.FAKE_SKILLS:
            dest = tmp_path / ".cursor" / "skills" / skill / "SKILL.md"
            assert dest.exists(), f"Expected {dest}"
            assert "Skill content" in dest.read_text()

    def test_vscode_skill_files_created(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Each skill should be written as SKILL.md under .github/skills/<skill>/."""
        mock_urlopen = MagicMock(side_effect=self._fake_urlopen)
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        _install_agent_skills(tmp_path, ["vscode"])

        for skill in self.FAKE_SKILLS:
            dest = tmp_path / ".github" / "skills" / skill / "SKILL.md"
            assert dest.exists(), f"Expected {dest}"

    def test_claude_skill_files_created(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Each skill should be written as SKILL.md under .claude/skills/<skill>/."""
        mock_urlopen = MagicMock(side_effect=self._fake_urlopen)
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        _install_agent_skills(tmp_path, ["claude"])

        for skill in self.FAKE_SKILLS:
            dest = tmp_path / ".claude" / "skills" / skill / "SKILL.md"
            assert dest.exists(), f"Expected {dest}"

    def test_jetbrains_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """JetBrains IDE has no skills location; nothing should be written."""
        mock_urlopen = MagicMock(side_effect=self._fake_urlopen)
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        _install_agent_skills(tmp_path, ["jetbrains"])

        assert not (tmp_path / ".cursor").exists()
        assert not (tmp_path / ".github").exists()
        assert not (tmp_path / ".claude").exists()

    def test_multiple_ides_all_receive_skills(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """All three supported IDEs should receive the full skill set in one call."""
        mock_urlopen = MagicMock(side_effect=self._fake_urlopen)
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
        _install_agent_skills(tmp_path, ["cursor", "vscode", "claude"])

        for skill in self.FAKE_SKILLS:
            assert (tmp_path / ".cursor" / "skills" / skill / "SKILL.md").exists()
            assert (tmp_path / ".github" / "skills" / skill / "SKILL.md").exists()
            assert (tmp_path / ".claude" / "skills" / skill / "SKILL.md").exists()

    def test_network_failure_writes_nothing(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """When the GitHub API call fails, no skill directories should be created."""
        monkeypatch.setattr(
            "urllib.request.urlopen", MagicMock(side_effect=OSError("network error"))
        )
        _install_agent_skills(tmp_path, ["cursor", "vscode", "claude"])

        assert not (tmp_path / ".cursor").exists()
        assert not (tmp_path / ".github").exists()
        assert not (tmp_path / ".claude").exists()

    def test_skills_flag_triggers_install(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Passing --skills in non-interactive mode calls _install_agent_skills."""
        mock_install = MagicMock()
        monkeypatch.setattr("rasa.cli.tools_wizard._install_agent_skills", mock_install)
        args = _make_args(project_path=str(tmp_path), ides="cursor", skills=True)
        run_wizard(args)

        mock_install.assert_called_once()

    def test_no_skills_flag_skips_install(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Without --skills, _install_agent_skills should not be called."""
        mock_install = MagicMock()
        monkeypatch.setattr("rasa.cli.tools_wizard._install_agent_skills", mock_install)
        args = _make_args(project_path=str(tmp_path), ides="cursor", skills=False)
        run_wizard(args)

        mock_install.assert_not_called()

    def test_no_ides_skips_install(self, tmp_path: Path, monkeypatch: Any) -> None:
        """When no IDEs are selected, skills install and warn should never run."""
        mock_install = MagicMock()
        mock_warn = MagicMock()
        monkeypatch.setattr("rasa.cli.tools_wizard._install_agent_skills", mock_install)
        monkeypatch.setattr(
            "rasa.cli.tools_wizard._warn_if_agent_skills_exist", mock_warn
        )
        args = _make_args(project_path=str(tmp_path), ides=None, skills=True)
        run_wizard(args)

        mock_install.assert_not_called()
        mock_warn.assert_not_called()


_VALIDATE_LICENSE = "rasa.utils.licensing.validate_license_from_env"


class TestLicenseEnforcement:
    def test_precheck_passes_with_valid_license(self, monkeypatch: Any) -> None:
        """_precheck should complete without error when the license is valid."""
        mock_validate = MagicMock()
        monkeypatch.setattr(_VALIDATE_LICENSE, mock_validate)
        _precheck()
        mock_validate.assert_called_once()

    def test_precheck_exits_when_license_missing(self, monkeypatch: Any) -> None:
        """_precheck should propagate SystemExit when no license is found."""
        monkeypatch.setattr(
            _VALIDATE_LICENSE,
            MagicMock(side_effect=SystemExit("A Rasa license is required.")),
        )
        with pytest.raises(SystemExit):
            _precheck()

    def test_precheck_exits_when_license_invalid(self, monkeypatch: Any) -> None:
        """_precheck should propagate SystemExit when the license is invalid."""
        monkeypatch.setattr(
            _VALIDATE_LICENSE,
            MagicMock(side_effect=SystemExit("Failed to validate Rasa license.")),
        )
        with pytest.raises(SystemExit):
            _precheck()

    def test_init_tools_exits_before_wizard_without_license(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """init_tools must exit before reaching the wizard if no license is set."""
        mock_wizard = MagicMock()
        monkeypatch.setattr("rasa.cli.tools_wizard.run_wizard", mock_wizard)
        monkeypatch.setattr(_VALIDATE_LICENSE, MagicMock(side_effect=SystemExit(1)))

        with pytest.raises(SystemExit):
            init_tools(_make_args(project_path=str(tmp_path)))

        mock_wizard.assert_not_called()

    def test_run_tools_exits_before_server_without_license(
        self, monkeypatch: Any
    ) -> None:
        """run_tools must exit before starting the MCP server if no license is set."""
        monkeypatch.setattr(_VALIDATE_LICENSE, MagicMock(side_effect=SystemExit(1)))

        with pytest.raises(SystemExit):
            run_tools(_make_args(mode=MCP_TOOLS_TRANSPORT_STDIO))


class TestRunConfigValidation:
    def test_invalid_ide_rejected(self) -> None:
        """Invalid IDE values should raise ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported IDE"):
            RunConfig(ide_integrations=["cursor", "invalid-ide"])

    def test_valid_ides_accepted(self) -> None:
        """All supported IDEs should be accepted."""
        cfg = RunConfig(ide_integrations=["cursor", "vscode", "claude", "jetbrains"])
        assert len(cfg.ide_integrations) == 4

    def test_empty_ide_list_accepted(self) -> None:
        """Empty IDE list should be valid."""
        cfg = RunConfig(ide_integrations=[])
        assert cfg.ide_integrations == []

    def test_bare_string_coerced_to_single_element_list(self) -> None:
        """A scalar string (as written in YAML without brackets) is accepted."""
        cfg = RunConfig(ide_integrations="cursor")  # type: ignore[arg-type]
        assert cfg.ide_integrations == ["cursor"]

    def test_bare_invalid_string_names_ide_not_characters(self) -> None:
        """A bare invalid string should report the IDE name, not its characters."""
        with pytest.raises(ValidationError, match="not-an-ide"):
            RunConfig(ide_integrations="not-an-ide")  # type: ignore[arg-type]

    def test_bare_string_loaded_from_yaml(self, tmp_path: Path) -> None:
        """tools.yaml with a scalar ide_integrations value is loaded correctly."""
        path = tmp_path / "tools.yaml"
        path.write_text("mode: stdio\nide_integrations: cursor\n")
        cfg = RunConfig.load(path)
        assert cfg.ide_integrations == ["cursor"]


class TestRunToolsProjectPathResolution:
    """run_tools must honour config.project_path when no explicit override exists."""

    def _patch_server(self, monkeypatch: Any) -> MagicMock:
        mock_server = MagicMock()
        monkeypatch.setattr(
            "rasa.builder.copilot.mcp_server.server.run_server", mock_server
        )
        return mock_server

    def test_config_project_path_used_when_no_cli_arg(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """project_path from a loaded config file must be passed to run_server."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config_dir = tmp_path / ".rasa"
        config_dir.mkdir()
        config_file = config_dir / "tools.yaml"
        config_file.write_text(f"mode: stdio\nproject_path: {project_dir}\n")

        monkeypatch.setattr(_VALIDATE_LICENSE, MagicMock())
        mock_server = self._patch_server(monkeypatch)
        monkeypatch.setenv(MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR, "")

        args = _make_args(config=str(config_dir))
        run_tools(args)

        call_kwargs = mock_server.call_args.kwargs
        assert call_kwargs["project_folder"] == str(project_dir.resolve())

    def test_cli_project_path_overrides_config(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """An explicit --project-path CLI arg must win over config.project_path."""
        config_project = tmp_path / "config_project"
        config_project.mkdir()
        cli_project = tmp_path / "cli_project"
        cli_project.mkdir()

        config_dir = tmp_path / ".rasa"
        config_dir.mkdir()
        config_file = config_dir / "tools.yaml"
        config_file.write_text(f"mode: stdio\nproject_path: {config_project}\n")

        monkeypatch.setattr(_VALIDATE_LICENSE, MagicMock())
        mock_server = self._patch_server(monkeypatch)
        monkeypatch.delenv(MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR, raising=False)

        # --config and --project-path are mutually exclusive, so we exercise
        # the case where the config file lives at the default location inside
        # cli_project and we pass --project-path directly.
        cli_config_dir = cli_project / ".rasa"
        cli_config_dir.mkdir()
        (cli_config_dir / "tools.yaml").write_text(
            f"mode: stdio\nproject_path: {config_project}\n"
        )

        args = _make_args(project_path=str(cli_project))
        run_tools(args)

        call_kwargs = mock_server.call_args.kwargs
        assert call_kwargs["project_folder"] == str(cli_project.resolve())

    def test_env_var_project_path_overrides_config(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """RASA_PROJECT_FOLDER env var must win over config.project_path."""
        config_project = tmp_path / "config_project"
        config_project.mkdir()
        env_project = tmp_path / "env_project"
        env_project.mkdir()

        config_dir = env_project / ".rasa"
        config_dir.mkdir()
        config_file = config_dir / "tools.yaml"
        config_file.write_text(f"mode: stdio\nproject_path: {config_project}\n")

        monkeypatch.setattr(_VALIDATE_LICENSE, MagicMock())
        mock_server = self._patch_server(monkeypatch)
        monkeypatch.setenv(MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR, str(env_project))

        args = _make_args()
        run_tools(args)

        call_kwargs = mock_server.call_args.kwargs
        assert call_kwargs["project_folder"] == str(env_project.resolve())


class TestConfirmOverwrite:
    def test_non_interactive_always_returns_true(self) -> None:
        assert _confirm_overwrite(non_interactive=True) is True

    def test_interactive_proceeds_when_user_confirms(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "questionary.confirm", lambda *a, **kw: MagicMock(ask=lambda: True)
        )
        assert _confirm_overwrite(non_interactive=False) is True

    def test_interactive_aborts_when_user_declines(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "questionary.confirm", lambda *a, **kw: MagicMock(ask=lambda: False)
        )
        assert _confirm_overwrite(non_interactive=False) is False

    def test_interactive_aborts_on_ctrl_c(self, monkeypatch: Any) -> None:
        """Ctrl+C (questionary returns None) must exit with code 1, not silently."""
        monkeypatch.setattr(
            "questionary.confirm", lambda *a, **kw: MagicMock(ask=lambda: None)
        )
        with pytest.raises(SystemExit):
            _confirm_overwrite(non_interactive=False)


class TestFetchOfflineDocs:
    def _make_urlopen(self, content: bytes = b"ok") -> Any:
        """Return a fake urlopen that yields *content* for every request."""

        def fake_urlopen(url: str, timeout: int) -> Any:
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(return_value=content)
            return mock

        return fake_urlopen

    def test_creates_dest_dir_and_writes_files(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Successful fetch writes each llms.txt file under .rasa/."""
        monkeypatch.setattr(
            "urllib.request.urlopen",
            self._make_urlopen(b"content of docs"),
        )
        _fetch_offline_docs(tmp_path)

        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        assert (dest_dir / "llms.txt").exists()
        assert (dest_dir / "llms-full.txt").exists()
        assert "content of docs" in (dest_dir / "llms.txt").read_bytes().decode()

    def test_graceful_failure_does_not_raise(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """A network error must not propagate — the wizard continues."""
        monkeypatch.setattr(
            "urllib.request.urlopen",
            MagicMock(side_effect=OSError("network error")),
        )
        _fetch_offline_docs(tmp_path)  # must not raise

    def test_partial_failure_continues_remaining_files(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """If one file fails, the other is still attempted."""
        fetched_urls: list[str] = []

        def fake_urlopen(url: str, timeout: int) -> Any:
            fetched_urls.append(url)
            if "llms-full.txt" in url:
                raise OSError("timeout")
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(return_value=b"ok")
            return mock

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        _fetch_offline_docs(tmp_path)

        assert len(fetched_urls) == 2
        assert (tmp_path / TOOLS_CONFIG_DIR / "llms.txt").exists()
        assert not (tmp_path / TOOLS_CONFIG_DIR / "llms-full.txt").exists()

    def test_uses_custom_base_url(self, tmp_path: Path, monkeypatch: Any) -> None:
        """The fetched URL must use the resolved base URL."""
        fetched_urls: list[str] = []

        def fake_urlopen(url: str, timeout: int) -> Any:
            fetched_urls.append(url)
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(return_value=b"ok")
            return mock

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        monkeypatch.setenv(LLMS_TXT_BASE_URL_ENV_VAR, "https://custom.example.com")
        _fetch_offline_docs(tmp_path)

        assert all(u.startswith("https://custom.example.com/") for u in fetched_urls)


class TestWarnIfOfflineDocsExist:
    def test_no_warning_when_no_files_present(self, tmp_path: Path) -> None:
        """No output expected when the .rasa/ dir has no llms.txt files."""
        _warn_if_offline_docs_exist(tmp_path)  # must not raise; nothing to assert

    def test_warns_when_llms_txt_exists(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Warning panel is printed when at least one llms.txt file is present."""
        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        dest_dir.mkdir(parents=True)
        (dest_dir / "llms.txt").write_text("docs")

        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools_wizard.console", wide)
        _warn_if_offline_docs_exist(tmp_path)

        assert "llms.txt" in buf.getvalue()

    def test_warns_for_each_present_file(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Both llms.txt files are listed in the warning when both are present."""
        dest_dir = tmp_path / TOOLS_CONFIG_DIR
        dest_dir.mkdir(parents=True)
        (dest_dir / "llms.txt").write_text("docs")
        (dest_dir / "llms-full.txt").write_text("full docs")

        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools_wizard.console", wide)
        _warn_if_offline_docs_exist(tmp_path)

        output = buf.getvalue()
        assert "llms.txt" in output
        assert "llms-full.txt" in output


class TestWarnIfAgentSkillsExist:
    def test_no_warning_when_no_skills_installed(self, tmp_path: Path) -> None:
        """No output when no skill directories exist."""
        _warn_if_agent_skills_exist(tmp_path, ["cursor"])  # must not raise

    def test_no_warning_when_skills_dir_is_empty(self, tmp_path: Path) -> None:
        """No output when the skills directory exists but is empty."""
        (tmp_path / ".cursor" / "skills").mkdir(parents=True)
        _warn_if_agent_skills_exist(tmp_path, ["cursor"])  # must not raise

    def test_warns_when_skills_exist_for_selected_ide(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Warning panel is shown when a non-empty skills directory is found."""
        skill_dir = tmp_path / ".cursor" / "skills" / "rasa-flows"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("content")

        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools_wizard.console", wide)
        _warn_if_agent_skills_exist(tmp_path, ["cursor"])

        assert ".cursor/skills" in buf.getvalue()

    def test_no_warning_for_unselected_ide(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """No warning for an IDE not in the selected list, even if files exist."""
        skill_dir = tmp_path / ".cursor" / "skills" / "rasa-flows"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("content")

        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools_wizard.console", wide)
        _warn_if_agent_skills_exist(tmp_path, ["vscode"])

        assert buf.getvalue() == ""

    def test_warns_for_multiple_ides(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Warning lists all IDEs that have existing skills."""
        for base in (".cursor/skills", ".github/skills"):
            skill_dir = tmp_path / base / "rasa-flows"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("content")

        buf = StringIO()
        wide = Console(file=buf, width=1000)
        monkeypatch.setattr("rasa.cli.tools_wizard.console", wide)
        _warn_if_agent_skills_exist(tmp_path, ["cursor", "vscode"])

        output = buf.getvalue()
        assert ".cursor/skills" in output
        assert ".github/skills" in output

    def test_wizard_warns_when_skills_exist_and_install_declined(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_wizard calls _warn_if_agent_skills_exist when skills are declined."""
        mock_warn = MagicMock()
        monkeypatch.setattr(
            "rasa.cli.tools_wizard._warn_if_agent_skills_exist", mock_warn
        )
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))
        args = _make_args(project_path=str(tmp_path), ides="cursor", skills=False)
        run_wizard(args)

        mock_warn.assert_called_once_with(tmp_path, ["cursor"])

    def test_wizard_skips_warn_when_skills_installed(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """_warn_if_agent_skills_exist is not called when skills are installed."""
        mock_warn = MagicMock()
        mock_install = MagicMock()
        monkeypatch.setattr(
            "rasa.cli.tools_wizard._warn_if_agent_skills_exist", mock_warn
        )
        monkeypatch.setattr("rasa.cli.tools_wizard._install_agent_skills", mock_install)
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))
        args = _make_args(project_path=str(tmp_path), ides="cursor", skills=True)
        run_wizard(args)

        mock_install.assert_called_once()
        mock_warn.assert_not_called()

    def test_wizard_skips_warn_when_no_ides_configured(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """_warn_if_agent_skills_exist is never reached when no IDEs are selected."""
        mock_warn = MagicMock()
        monkeypatch.setattr(
            "rasa.cli.tools_wizard._warn_if_agent_skills_exist", mock_warn
        )
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))
        args = _make_args(project_path=str(tmp_path), ides=None, skills=False)
        run_wizard(args)

        mock_warn.assert_not_called()


class TestFetchSkillNames:
    def test_returns_only_directory_entries(self, monkeypatch: Any) -> None:
        """Only entries with type=='dir' should be returned; files are filtered out."""
        entries = [
            {"name": "rasa-flows", "type": "dir"},
            {"name": "README.md", "type": "file"},
            {"name": "rasa-actions", "type": "dir"},
        ]
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(return_value=json.dumps(entries).encode())
        monkeypatch.setattr("urllib.request.urlopen", MagicMock(return_value=mock))

        names = _fetch_skill_names()
        assert names == ["rasa-flows", "rasa-actions"]

    def test_returns_empty_list_on_network_error(self, monkeypatch: Any) -> None:
        """Any network failure must return [] without raising."""
        monkeypatch.setattr(
            "urllib.request.urlopen", MagicMock(side_effect=OSError("timeout"))
        )
        assert _fetch_skill_names() == []

    def test_returns_empty_list_on_invalid_json(self, monkeypatch: Any) -> None:
        """Malformed JSON from the API must return [] without raising."""
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(return_value=b"NOT JSON {{{")
        monkeypatch.setattr("urllib.request.urlopen", MagicMock(return_value=mock))

        assert _fetch_skill_names() == []

    def test_returns_empty_list_when_no_dirs(self, monkeypatch: Any) -> None:
        """When the API returns only non-dir entries, result must be empty."""
        entries = [{"name": "README.md", "type": "file"}]
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(return_value=json.dumps(entries).encode())
        monkeypatch.setattr("urllib.request.urlopen", MagicMock(return_value=mock))

        assert _fetch_skill_names() == []


class TestWriteIdeConfigs:
    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))

    def test_no_files_written_when_no_ides_selected(self, tmp_path: Path) -> None:
        """_write_ide_configs must short-circuit when ide_integrations is empty."""
        config = RunConfig(mode=MCP_TOOLS_TRANSPORT_STDIO, ide_integrations=[])
        _write_ide_configs(tmp_path, config)

        assert not (tmp_path / ".cursor").exists()
        assert not (tmp_path / ".vscode").exists()
        assert not (tmp_path / ".mcp.json").exists()

    def test_writes_all_file_based_ides(self, tmp_path: Path) -> None:
        """All three file-based IDEs produce their config files."""
        config = RunConfig(
            mode=MCP_TOOLS_TRANSPORT_STDIO,
            ide_integrations=["cursor", "vscode", "claude"],
        )
        _write_ide_configs(tmp_path, config)

        assert (tmp_path / ".cursor" / "mcp.json").exists()
        assert (tmp_path / ".vscode" / "mcp.json").exists()
        assert (tmp_path / ".mcp.json").exists()


class TestPrintSummary:
    def _render(self, config: RunConfig, tmp_path: Path, monkeypatch: Any) -> str:
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        buf = StringIO()
        monkeypatch.setattr("rasa.cli.tools_wizard.console", Console(file=buf))
        _print_summary(config, config_path)
        return buf.getvalue()

    def test_stdio_mode_omits_port(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Summary must not include a port line for stdio mode."""
        output = self._render(
            RunConfig(mode=MCP_TOOLS_TRANSPORT_STDIO), tmp_path, monkeypatch
        )
        assert "stdio" in output
        assert "Port" not in output

    def test_http_mode_includes_port(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Summary must include the port value for HTTP mode."""
        output = self._render(
            RunConfig(mode=MCP_TOOLS_TRANSPORT_HTTP, port=9000), tmp_path, monkeypatch
        )
        assert "9000" in output

    def test_includes_ide_names_when_configured(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Summary must list human-readable IDE names when IDEs are set."""
        config = RunConfig(
            mode=MCP_TOOLS_TRANSPORT_STDIO, ide_integrations=["cursor", "vscode"]
        )
        output = self._render(config, tmp_path, monkeypatch)
        assert "Cursor" in output
        assert "VS Code" in output
