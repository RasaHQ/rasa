"""Tests for ``rasa tools init skills``."""

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rasa.cli.tools.constants import (
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.models import RunConfig
from rasa.cli.tools.skills import (
    _fetch_skill_names_from_repo,
    install_agent_skills,
    skills_tools,
    warn_if_agent_skills_exist,
)

_FAKE_LICENSE = "fake-license-token-for-tests"
_VALIDATE_LICENSE = "rasa.utils.licensing.validate_license_from_env"


def _make_args(**overrides: Any) -> argparse.Namespace:
    defaults = dict(project_path=None, ides=None, yes=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture()
def fake_skills() -> list[str]:
    return [
        "rasa-building-flows",
        "rasa-writing-custom-actions",
    ]


@pytest.fixture()
def compatible_frontmatter() -> str:
    return (
        '---\nname: {skill}\nmetadata:\n  rasa_version: ">=3.9.0"\n---\n'
        "# Skill content"
    )


@pytest.fixture()
def fake_github(
    monkeypatch: Any, fake_skills: list[str], compatible_frontmatter: str
) -> None:
    """Patch urllib.request.urlopen to return fake compatible skill data."""

    def _urlopen(url: str, timeout: int = 30) -> MagicMock:
        if "api.github.com" in url:
            entries = [{"name": s, "type": "dir"} for s in fake_skills]
            body = json.dumps(entries).encode()
        else:
            skill_name = url.rstrip("/").split("/")[-2]
            body = compatible_frontmatter.format(skill=skill_name).encode()
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(return_value=body)
        return mock

    monkeypatch.setattr(
        "urllib.request.urlopen",
        MagicMock(side_effect=_urlopen),
    )


class TestFetchSkillNames:
    def test_returns_only_directory_entries(self, monkeypatch: Any) -> None:
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

        names = _fetch_skill_names_from_repo()
        assert names == ["rasa-flows", "rasa-actions"]

    def test_returns_empty_list_on_network_error(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            MagicMock(side_effect=OSError("timeout")),
        )
        assert _fetch_skill_names_from_repo() == []

    def test_returns_empty_list_on_invalid_json(self, monkeypatch: Any) -> None:
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(return_value=b"NOT JSON {{{")
        monkeypatch.setattr("urllib.request.urlopen", MagicMock(return_value=mock))

        assert _fetch_skill_names_from_repo() == []

    def test_returns_empty_list_when_no_dirs(self, monkeypatch: Any) -> None:
        entries = [{"name": "README.md", "type": "file"}]
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read = MagicMock(return_value=json.dumps(entries).encode())
        monkeypatch.setattr("urllib.request.urlopen", MagicMock(return_value=mock))

        assert _fetch_skill_names_from_repo() == []


class TestInstallAgentSkills:
    def test_cursor_skill_files_created(
        self, tmp_path: Path, fake_skills: list[str], fake_github: None
    ) -> None:
        install_agent_skills(tmp_path, ["cursor"], non_interactive=True)

        for skill in fake_skills:
            dest = tmp_path / ".cursor" / "skills" / skill / "SKILL.md"
            assert dest.exists(), f"Expected {dest}"
            assert "Skill content" in dest.read_text()

    def test_vscode_skill_files_created(
        self, tmp_path: Path, fake_skills: list[str], fake_github: None
    ) -> None:
        install_agent_skills(tmp_path, ["vscode"], non_interactive=True)

        for skill in fake_skills:
            dest = tmp_path / ".github" / "skills" / skill / "SKILL.md"
            assert dest.exists(), f"Expected {dest}"

    def test_jetbrains_skipped(self, tmp_path: Path, fake_github: None) -> None:
        install_agent_skills(tmp_path, ["jetbrains"], non_interactive=True)

        assert not (tmp_path / ".cursor").exists()
        assert not (tmp_path / ".github").exists()
        assert not (tmp_path / ".claude").exists()

    def test_network_failure_writes_nothing(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            MagicMock(side_effect=OSError("network error")),
        )
        install_agent_skills(tmp_path, ["cursor"], non_interactive=True)

        assert not (tmp_path / ".cursor").exists()


class TestExistenceCheck:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, fake_github: None) -> None:
        self.tmp_path = tmp_path

    def test_non_interactive_overwrites_without_asking(
        self, fake_skills: list[str]
    ) -> None:
        (self.tmp_path / ".cursor" / "skills" / "old-skill").mkdir(parents=True)
        (self.tmp_path / ".cursor" / "skills" / "old-skill" / "SKILL.md").write_text(
            "old"
        )

        install_agent_skills(self.tmp_path, ["cursor"], non_interactive=True)

        for skill in fake_skills:
            dest = self.tmp_path / ".cursor" / "skills" / skill / "SKILL.md"
            assert dest.exists()

    def test_interactive_no_aborts(
        self, fake_skills: list[str], monkeypatch: Any
    ) -> None:
        (self.tmp_path / ".cursor" / "skills" / "old-skill").mkdir(parents=True)
        (self.tmp_path / ".cursor" / "skills" / "old-skill" / "SKILL.md").write_text(
            "old"
        )

        monkeypatch.setattr(
            "rasa.cli.tools.skills.questionary.confirm",
            lambda *a, **kw: MagicMock(ask=lambda: False),
        )

        install_agent_skills(self.tmp_path, ["cursor"], non_interactive=False)

        for skill in fake_skills:
            dest = self.tmp_path / ".cursor" / "skills" / skill / "SKILL.md"
            assert not dest.exists()

    def test_interactive_yes_continues(
        self, fake_skills: list[str], monkeypatch: Any
    ) -> None:
        (self.tmp_path / ".cursor" / "skills" / "old-skill").mkdir(parents=True)
        (self.tmp_path / ".cursor" / "skills" / "old-skill" / "SKILL.md").write_text(
            "old"
        )

        monkeypatch.setattr(
            "rasa.cli.tools.skills.questionary.confirm",
            lambda *a, **kw: MagicMock(ask=lambda: True),
        )

        install_agent_skills(self.tmp_path, ["cursor"], non_interactive=False)

        for skill in fake_skills:
            dest = self.tmp_path / ".cursor" / "skills" / skill / "SKILL.md"
            assert dest.exists()

    def test_no_prompt_when_no_existing_skills(
        self, fake_skills: list[str], monkeypatch: Any
    ) -> None:
        mock_confirm = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.skills.questionary.confirm", mock_confirm)

        install_agent_skills(self.tmp_path, ["cursor"], non_interactive=False)

        mock_confirm.assert_not_called()
        for skill in fake_skills:
            dest = self.tmp_path / ".cursor" / "skills" / skill / "SKILL.md"
            assert dest.exists()


class TestVersionGuard:
    @pytest.fixture()
    def incompatible_frontmatter(self) -> str:
        return (
            '---\nname: {skill}\nmetadata:\n  rasa_version: ">=99.0.0"\n---\n'
            "# Future skill"
        )

    @pytest.fixture()
    def mixed_github(
        self,
        monkeypatch: Any,
        fake_skills: list[str],
        compatible_frontmatter: str,
        incompatible_frontmatter: str,
    ) -> None:
        """Patch urlopen: rasa-building-flows compatible, other incompatible."""

        def _urlopen(url: str, timeout: int = 30) -> MagicMock:
            if "api.github.com" in url:
                entries = [{"name": s, "type": "dir"} for s in fake_skills]
                body = json.dumps(entries).encode()
            elif "rasa-building-flows" in url:
                body = compatible_frontmatter.format(
                    skill="rasa-building-flows"
                ).encode()
            else:
                body = incompatible_frontmatter.format(
                    skill="rasa-writing-custom-actions"
                ).encode()
            mock = MagicMock()
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            mock.read = MagicMock(return_value=body)
            return mock

        monkeypatch.setattr(
            "urllib.request.urlopen",
            MagicMock(side_effect=_urlopen),
        )

    def test_non_interactive_skips_incompatible(
        self, tmp_path: Path, mixed_github: None
    ) -> None:
        install_agent_skills(tmp_path, ["cursor"], non_interactive=True)

        skills_dir = tmp_path / ".cursor" / "skills"
        compatible = skills_dir / "rasa-building-flows" / "SKILL.md"
        incompatible = skills_dir / "rasa-writing-custom-actions" / "SKILL.md"
        assert compatible.exists()
        assert not incompatible.exists()

    def test_interactive_no_skips_incompatible(
        self, tmp_path: Path, mixed_github: None, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "rasa.cli.tools.skills.questionary.confirm",
            lambda *a, **kw: MagicMock(ask=lambda: False),
        )

        install_agent_skills(tmp_path, ["cursor"], non_interactive=False)

        skills_dir = tmp_path / ".cursor" / "skills"
        compatible = skills_dir / "rasa-building-flows" / "SKILL.md"
        incompatible = skills_dir / "rasa-writing-custom-actions" / "SKILL.md"
        assert compatible.exists()
        assert not incompatible.exists()

    def test_interactive_yes_installs_all(
        self,
        tmp_path: Path,
        fake_skills: list[str],
        mixed_github: None,
        monkeypatch: Any,
    ) -> None:
        monkeypatch.setattr(
            "rasa.cli.tools.skills.questionary.confirm",
            lambda *a, **kw: MagicMock(ask=lambda: True),
        )

        install_agent_skills(tmp_path, ["cursor"], non_interactive=False)

        for skill in fake_skills:
            dest = tmp_path / ".cursor" / "skills" / skill / "SKILL.md"
            assert dest.exists()

    def test_all_compatible_no_prompt(
        self,
        tmp_path: Path,
        fake_skills: list[str],
        fake_github: None,
        monkeypatch: Any,
    ) -> None:
        mock_confirm = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.skills.questionary.confirm", mock_confirm)

        install_agent_skills(tmp_path, ["cursor"], non_interactive=False)

        mock_confirm.assert_not_called()
        for skill in fake_skills:
            dest = tmp_path / ".cursor" / "skills" / skill / "SKILL.md"
            assert dest.exists()


class TestWarnIfAgentSkillsExist:
    def test_no_warning_when_no_skills_installed(self, tmp_path: Path) -> None:
        warn_if_agent_skills_exist(tmp_path, ["cursor"])

    def test_no_warning_when_skills_dir_is_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".cursor" / "skills").mkdir(parents=True)
        warn_if_agent_skills_exist(tmp_path, ["cursor"])


class TestSkillsToolsEntrypoint:
    @pytest.fixture(autouse=True)
    def _mock_license(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_VALIDATE_LICENSE, MagicMock())

    @pytest.fixture()
    def mock_install(self, monkeypatch: Any) -> MagicMock:
        mock = MagicMock()
        monkeypatch.setattr("rasa.cli.tools.skills.install_agent_skills", mock)
        return mock

    def test_exits_when_no_ides(self, tmp_path: Path) -> None:
        args = _make_args(project_path=str(tmp_path))
        with pytest.raises(SystemExit):
            skills_tools(args)

    def test_uses_saved_config_ides(
        self, tmp_path: Path, mock_install: MagicMock
    ) -> None:
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config_path.parent.mkdir(parents=True)
        RunConfig(ide_integrations=["cursor"]).save(config_path)

        args = _make_args(project_path=str(tmp_path))
        skills_tools(args)

        mock_install.assert_called_once_with(
            tmp_path, ["cursor"], non_interactive=False
        )

    def test_cli_ides_override_saved_config(
        self, tmp_path: Path, mock_install: MagicMock
    ) -> None:
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config_path.parent.mkdir(parents=True)
        RunConfig(ide_integrations=["vscode"]).save(config_path)

        args = _make_args(project_path=str(tmp_path), ides="cursor,claude")
        skills_tools(args)

        mock_install.assert_called_once_with(
            tmp_path, ["cursor", "claude"], non_interactive=False
        )

    def test_yes_flag_passes_non_interactive(
        self, tmp_path: Path, mock_install: MagicMock
    ) -> None:
        config_path = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        config_path.parent.mkdir(parents=True)
        RunConfig(ide_integrations=["cursor"]).save(config_path)

        args = _make_args(project_path=str(tmp_path), yes=True)
        skills_tools(args)

        mock_install.assert_called_once_with(tmp_path, ["cursor"], non_interactive=True)
