import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from rasa.cli.tools.constants import (
    DEFAULT_RASA_SERVER_URL,
    DOCS_MODE_OFFLINE,
    DOCS_MODE_ONLINE,
    MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR,
    MCP_TOOLS_TRANSPORT_HTTP,
    MCP_TOOLS_TRANSPORT_STDIO,
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.init import (
    _ask_ides,
    _build_http_entry,
    _build_stdio_entry,
    _confirm_overwrite,
    _merge_mcp_json,
    _print_summary,
    _run_non_interactive,
    _write_ide_configs,
    init_tools,
    run_wizard,
)
from rasa.cli.tools.run import run_tools
from rasa.cli.tools.utils import RunConfig

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
        rasa_server_url=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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

    def test_default_rasa_server_url(self, tmp_path: Path) -> None:
        args = _make_args()
        cfg = _run_non_interactive(args, tmp_path)
        assert cfg.rasa_server_url == DEFAULT_RASA_SERVER_URL

    def test_explicit_rasa_server_url(self, tmp_path: Path) -> None:
        args = _make_args(rasa_server_url="http://my-server:9999")
        cfg = _run_non_interactive(args, tmp_path)
        assert cfg.rasa_server_url == "http://my-server:9999"

    def test_ides_whitespace_handling(self, tmp_path: Path) -> None:
        args = _make_args(ides=" cursor , vscode , ")
        cfg = _run_non_interactive(args, tmp_path)

        assert cfg.ide_integrations == ["cursor", "vscode"]

    def test_ides_case_normalisation(self, tmp_path: Path) -> None:
        args = _make_args(ides="Cursor,VSCODE")
        cfg = _run_non_interactive(args, tmp_path)

        assert cfg.ide_integrations == ["cursor", "vscode"]


class TestMcpEntryBuilders:
    def test_stdio_entry(self, tmp_path: Path) -> None:
        config = RunConfig(mode=MCP_TOOLS_TRANSPORT_STDIO)
        entry = _build_stdio_entry(tmp_path, config)
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

    def test_stdio_entry_has_no_env(self, tmp_path: Path) -> None:
        config = RunConfig(mode=MCP_TOOLS_TRANSPORT_STDIO)
        entry = _build_stdio_entry(tmp_path, config)
        assert "env" not in entry

    def test_stdio_entry_includes_custom_rasa_server_url(self, tmp_path: Path) -> None:
        config = RunConfig(
            mode=MCP_TOOLS_TRANSPORT_STDIO,
            rasa_server_url="http://my-server:9999",
        )
        entry = _build_stdio_entry(tmp_path, config)
        assert "--rasa-server-url" in entry["args"]
        idx = entry["args"].index("--rasa-server-url")
        assert entry["args"][idx + 1] == "http://my-server:9999"

    def test_stdio_entry_omits_default_rasa_server_url(self, tmp_path: Path) -> None:
        config = RunConfig(
            mode=MCP_TOOLS_TRANSPORT_STDIO,
            rasa_server_url=DEFAULT_RASA_SERVER_URL,
        )
        entry = _build_stdio_entry(tmp_path, config)
        assert "--rasa-server-url" not in entry["args"]

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
        monkeypatch.setattr("rasa.cli.tools.init.fetch_offline_docs", mock_fetch)
        args = _make_args(project_path=str(tmp_path), docs=DOCS_MODE_OFFLINE)
        run_wizard(args)

        mock_fetch.assert_called_once_with(tmp_path, non_interactive=True)

    def test_online_docs_skips_fetch(self, tmp_path: Path, monkeypatch: Any) -> None:
        mock_fetch = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.init.fetch_offline_docs", mock_fetch)
        args = _make_args(project_path=str(tmp_path), docs=DOCS_MODE_ONLINE)
        run_wizard(args)

        mock_fetch.assert_not_called()

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
        monkeypatch.setattr("rasa.cli.tools.init.warn_if_offline_docs_exist", mock_warn)
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

    @pytest.mark.parametrize(
        "ide, cfg_path, wrapper_key",
        [
            ("cursor", ".cursor/mcp.json", "mcpServers"),
            ("vscode", ".vscode/mcp.json", "servers"),
            ("claude", ".mcp.json", "mcpServers"),
        ],
        ids=["cursor", "vscode", "claude"],
    )
    def test_stdio_config_has_no_env_block(
        self, tmp_path: Path, ide: str, cfg_path: str, wrapper_key: str
    ) -> None:
        """Stdio config must not embed credentials.

        License and proxy URL are loaded at runtime from the project's
        .env file and .rasa/tools.yaml respectively.
        """
        args = _make_args(project_path=str(tmp_path), ides=ide)
        run_wizard(args)

        data = json.loads((tmp_path / cfg_path).read_text())
        assert "env" not in data[wrapper_key]["rasa-tools"]

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


class TestWizardSkillsIntegration:
    """Tests that run_wizard correctly delegates to skills functions."""

    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_RETRIEVE_LICENSE, lambda: (_FAKE_LICENSE, "RASA_LICENSE"))

    def test_skills_flag_triggers_install(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Passing --skills in non-interactive mode calls install_agent_skills."""
        mock_install = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.init.install_agent_skills", mock_install)
        args = _make_args(project_path=str(tmp_path), ides="cursor", skills=True)
        run_wizard(args)

        mock_install.assert_called_once()

    def test_no_skills_flag_skips_install(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Without --skills, install_agent_skills should not be called."""
        mock_install = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.init.install_agent_skills", mock_install)
        args = _make_args(project_path=str(tmp_path), ides="cursor", skills=False)
        run_wizard(args)

        mock_install.assert_not_called()

    def test_no_ides_skips_install(self, tmp_path: Path, monkeypatch: Any) -> None:
        """When no IDEs are selected, skills install and warn should never run."""
        mock_install = MagicMock()
        mock_warn = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.init.install_agent_skills", mock_install)
        monkeypatch.setattr("rasa.cli.tools.init.warn_if_agent_skills_exist", mock_warn)
        args = _make_args(project_path=str(tmp_path), ides=None, skills=True)
        run_wizard(args)

        mock_install.assert_not_called()
        mock_warn.assert_not_called()

    def test_wizard_warns_when_skills_declined(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_wizard calls warn_if_agent_skills_exist when skills are declined."""
        mock_warn = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.init.warn_if_agent_skills_exist", mock_warn)
        args = _make_args(project_path=str(tmp_path), ides="cursor", skills=False)
        run_wizard(args)

        mock_warn.assert_called_once_with(tmp_path, ["cursor"])

    def test_wizard_skips_warn_when_skills_installed(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """warn_if_agent_skills_exist is not called when skills are installed."""
        mock_warn = MagicMock()
        mock_install = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.init.warn_if_agent_skills_exist", mock_warn)
        monkeypatch.setattr("rasa.cli.tools.init.install_agent_skills", mock_install)
        args = _make_args(project_path=str(tmp_path), ides="cursor", skills=True)
        run_wizard(args)

        mock_install.assert_called_once()
        mock_warn.assert_not_called()

    def test_wizard_skips_warn_when_no_ides_configured(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """warn_if_agent_skills_exist is never reached when no IDEs are selected."""
        mock_warn = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.init.warn_if_agent_skills_exist", mock_warn)
        args = _make_args(project_path=str(tmp_path), ides=None, skills=False)
        run_wizard(args)

        mock_warn.assert_not_called()


_VALIDATE_LICENSE = "rasa.utils.licensing.validate_license_from_env"


class TestLicenseEnforcement:
    def test_init_tools_exits_before_wizard_without_license(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """init_tools must exit before reaching the wizard if no license is set."""
        mock_wizard = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.init.run_wizard", mock_wizard)
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


class TestAskIdes:
    """Tests for the _ask_ides prompt, focusing on the validation callback."""

    def _capture_validate(self, monkeypatch: Any) -> Any:
        """Patch questionary.checkbox, return the captured validate kwarg."""
        captured: dict = {}

        def fake_checkbox(*args: Any, **kwargs: Any) -> MagicMock:
            captured["validate"] = kwargs.get("validate")
            return MagicMock(ask=lambda: ["cursor"])

        monkeypatch.setattr("rasa.cli.tools.init.questionary.checkbox", fake_checkbox)
        monkeypatch.setattr("rasa.cli.tools.init.restore_blocking_io", MagicMock())
        _ask_ides()
        return captured["validate"]

    def test_validate_accepts_non_empty_selection(self, monkeypatch: Any) -> None:
        validate = self._capture_validate(monkeypatch)
        assert validate(["cursor"]) is True

    def test_validate_accepts_multiple_selections(self, monkeypatch: Any) -> None:
        validate = self._capture_validate(monkeypatch)
        assert validate(["cursor", "vscode"]) is True

    def test_validate_rejects_empty_selection(self, monkeypatch: Any) -> None:
        validate = self._capture_validate(monkeypatch)
        result = validate([])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_validate_error_message_mentions_space_key(self, monkeypatch: Any) -> None:
        validate = self._capture_validate(monkeypatch)
        result = validate([])
        assert "Space" in result or "space" in result.lower()

    def test_returns_selected_ides(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "rasa.cli.tools.init.questionary.checkbox",
            lambda *a, **kw: MagicMock(ask=lambda: ["cursor", "vscode"]),
        )
        monkeypatch.setattr("rasa.cli.tools.init.restore_blocking_io", MagicMock())
        assert _ask_ides() == ["cursor", "vscode"]

    def test_aborts_on_ctrl_c(self, monkeypatch: Any) -> None:
        """Ctrl+C (questionary returns None) must exit the process."""
        monkeypatch.setattr(
            "rasa.cli.tools.init.questionary.checkbox",
            lambda *a, **kw: MagicMock(ask=lambda: None),
        )
        monkeypatch.setattr("rasa.cli.tools.init.restore_blocking_io", MagicMock())
        with pytest.raises(SystemExit):
            _ask_ides()


class TestRestoreBlockingIoCalledAfterPrompts:
    """Every questionary .ask() in init.py must be followed by restore_blocking_io."""

    _RESTORE = "rasa.cli.tools.init.restore_blocking_io"

    def test_confirm_overwrite_calls_restore(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "questionary.confirm", lambda *a, **kw: MagicMock(ask=lambda: True)
        )
        mock_restore = MagicMock()
        monkeypatch.setattr(self._RESTORE, mock_restore)

        _confirm_overwrite(non_interactive=False)
        mock_restore.assert_called()

    def test_run_interactive_calls_restore(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from rasa.cli.tools.init import _run_interactive

        select_answers = iter([MCP_TOOLS_TRANSPORT_STDIO, DOCS_MODE_OFFLINE])
        monkeypatch.setattr(
            "rasa.cli.tools.init.questionary.select",
            lambda *a, **kw: MagicMock(ask=lambda: next(select_answers)),
        )
        monkeypatch.setattr(
            "rasa.cli.tools.init.questionary.text",
            lambda *a, **kw: MagicMock(ask=lambda: DEFAULT_RASA_SERVER_URL),
        )
        monkeypatch.setattr(
            "rasa.cli.tools.init.questionary.checkbox",
            lambda *a, **kw: MagicMock(ask=lambda: ["cursor"]),
        )
        mock_restore = MagicMock()
        monkeypatch.setattr(self._RESTORE, mock_restore)

        _run_interactive(tmp_path)
        assert mock_restore.call_count >= 4

    def test_ask_install_agent_skills_calls_restore(self, monkeypatch: Any) -> None:
        from rasa.cli.tools.init import _ask_install_agent_skills

        monkeypatch.setattr(
            "rasa.cli.tools.init.questionary.confirm",
            lambda *a, **kw: MagicMock(ask=lambda: True),
        )
        mock_restore = MagicMock()
        monkeypatch.setattr(self._RESTORE, mock_restore)

        _ask_install_agent_skills()
        mock_restore.assert_called()


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
        monkeypatch.setattr("rasa.cli.tools.init.console", Console(file=buf))
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

    def test_http_mode_shows_dotenv_instructions(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """HTTP summary must guide users to put the license in .env."""
        output = self._render(
            RunConfig(mode=MCP_TOOLS_TRANSPORT_HTTP, port=9000),
            tmp_path,
            monkeypatch,
        )
        assert "RASA_LICENSE" in output
        assert ".env" in output
        assert "rasa tools run" in output

    def test_includes_rasa_server_url(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Summary must include the configured Rasa server URL."""
        config = RunConfig(
            mode=MCP_TOOLS_TRANSPORT_STDIO,
            rasa_server_url="http://my-server:9999",
        )
        output = self._render(config, tmp_path, monkeypatch)
        assert "http://my-server:9999" in output

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
