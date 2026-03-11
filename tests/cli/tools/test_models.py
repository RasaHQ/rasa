"""Tests for ``rasa.cli.tools.models``."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rasa.cli.tools.constants import TOOLS_CONFIG_DIR, TOOLS_CONFIG_FILENAME
from rasa.cli.tools.models import AgentSkillInfo, RunConfig
from rasa.shared.exceptions import RasaException


class TestAgentSkillInfo:
    _SKILL_FULL = """\
---
name: rasa-building-flows
metadata:
  author: rasa
  version: "0.1.0"
  rasa_version: ">=3.9.0"
---

# Building Flows
"""

    _SKILL_NO_VER = """\
---
name: rasa-building-flows
metadata:
  author: rasa
---

# Building Flows
"""

    _SKILL_PLAIN = "# Just a markdown file\n"

    @pytest.mark.parametrize(
        "content, expected",
        [
            (_SKILL_FULL, ">=3.9.0"),
            (_SKILL_NO_VER, None),
            (_SKILL_PLAIN, None),
            ('---\nmetadata:\n  rasa_version: ">=3.14.0"\n---\n', ">=3.14.0"),
            ('---\nmetadata:\n  rasa_version: "~=3.12.0"\n---\n', "~=3.12.0"),
            ("---\nmetadata:\n  rasa_version: >=3.7.0\n---\n", ">=3.7.0"),
            ('---\nmetadata:\n  rasa_version: "3.10.0"\n---\n', ">=3.10.0"),
            ('---\nmetadata:\n  rasa_version: "<=3.9.0"\n---\n', "<=3.9.0"),
        ],
        ids=[
            "full_frontmatter",
            "no_rasa_version_field",
            "no_frontmatter",
            "gte_prefix",
            "tilde_prefix",
            "unquoted",
            "bare_version_gets_gte",
            "lte_prefix",
        ],
    )
    def test_parse_rasa_version(self, content: str, expected: str) -> None:
        assert AgentSkillInfo._parse_rasa_version(content) == expected

    @pytest.mark.parametrize(
        "content, exp_version, exp_rasa_version",
        [
            (_SKILL_FULL, "0.1.0", ">=3.9.0"),
            (_SKILL_NO_VER, None, None),
            (_SKILL_PLAIN, None, None),
        ],
        ids=["all_fields", "missing_versions", "no_frontmatter"],
    )
    def test_from_content(
        self,
        content: str,
        exp_version: str,
        exp_rasa_version: str,
    ) -> None:
        skill = AgentSkillInfo.from_content("my-skill", content)
        assert skill.name == "my-skill"
        assert skill.version == exp_version
        assert skill.rasa_version == exp_rasa_version
        assert skill.content == content

    @pytest.mark.parametrize(
        "rasa_version, current, expected",
        [
            (">=3.9.0", "3.16.0", True),
            (">=3.17.0", "3.16.0", False),
            (">=3.16.0", "3.16.0", True),
            (None, "3.16.0", True),
            ("not.a.version", "3.16.0", True),
            (">=3.16.10", "3.16.2", True),
            (">=3.16.2", "3.16.10", True),
            ("3.9.0", "3.16.0", True),
            ("<=3.9.0", "3.16.0", False),
            ("<=3.16.0", "3.16.0", True),
            ("<=3.16.0", "3.9.0", True),
            ("~=3.16.0", "3.16.5", True),
            ("~=3.16.0", "3.17.0", False),
            ("~=3.16", "3.17.0", True),
            ("~=3.16", "3.16.5", True),
            ("~=3.16", "4.0.0", False),
            ("~=3.16", "2.9.0", False),
        ],
        ids=[
            "gte_older_requirement",
            "gte_newer_requirement",
            "gte_equal_versions",
            "no_requirement",
            "invalid_version",
            "gte_skill_higher_patch_still_compatible",
            "gte_skill_lower_patch_still_compatible",
            "bare_version_defaults_to_gte",
            "lte_rejects_higher_minor",
            "lte_equal_versions",
            "lte_allows_lower_minor",
            "compatible_release_same_minor",
            "compatible_release_rejects_next_minor",
            "two_part_tilde_allows_higher_minor",
            "two_part_tilde_allows_same_minor",
            "two_part_tilde_rejects_next_major",
            "two_part_tilde_rejects_older_major",
        ],
    )
    def test_is_compatible_with(
        self,
        rasa_version: str,
        current: str,
        expected: bool,
    ) -> None:
        skill = AgentSkillInfo(name="s", rasa_version=rasa_version, content="")
        assert skill.is_compatible_with(current) is expected


class TestRunConfig:
    # Validation =======================================================================

    @pytest.mark.parametrize(
        "ide_input, expected",
        [
            (
                ["cursor", "vscode", "claude", "jetbrains"],
                ["cursor", "vscode", "claude", "jetbrains"],
            ),
            ([], []),
            ("cursor", ["cursor"]),
        ],
        ids=["all_supported", "empty_list", "bare_string_coerced"],
    )
    def test_valid_ide_integrations(
        self, ide_input: object, expected: list[str]
    ) -> None:
        cfg = RunConfig(ide_integrations=ide_input)  # type: ignore[arg-type]
        assert cfg.ide_integrations == expected

    @pytest.mark.parametrize(
        "ide_input, error_match",
        [
            (["cursor", "invalid-ide"], "Unsupported IDE"),
            ("not-an-ide", "not-an-ide"),
        ],
        ids=["invalid_in_list", "bare_invalid_string"],
    )
    def test_invalid_ide_integrations(
        self, ide_input: object, error_match: str
    ) -> None:
        with pytest.raises(ValidationError, match=error_match):
            RunConfig(ide_integrations=ide_input)  # type: ignore[arg-type]

    def test_bare_string_loaded_from_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "tools.yaml"
        path.write_text("mode: stdio\nide_integrations: cursor\n")
        cfg = RunConfig.load(path)
        assert cfg.ide_integrations == ["cursor"]

    # Load errors ======================================================================

    @pytest.mark.parametrize(
        "content, error_match",
        [
            (None, "does not exist"),
            ("ide_integrations: [not-a-valid-ide]\n", "Invalid config"),
            ("key: [unclosed bracket\n", "Failed to read config file"),
            ("- item1\n- item2\n", "Invalid config"),
        ],
        ids=[
            "missing_file",
            "invalid_content",
            "unparseable_yaml",
            "non_dict_yaml",
        ],
    )
    def test_load_raises_on_bad_input(
        self, tmp_path: Path, content: str | None, error_match: str
    ) -> None:
        path = tmp_path / "tools.yaml"
        if content is not None:
            path.write_text(content)
        with pytest.raises(RasaException, match=error_match):
            RunConfig.load(path)

    # Serialization ====================================================================

    @pytest.mark.parametrize(
        "kwargs, expect_in, expect_not_in",
        [
            (
                dict(mode="stdio", ide_integrations=[]),
                {},
                ["ide_integrations", "port"],
            ),
            (
                dict(mode="stdio", ide_integrations=["cursor"]),
                {"ide_integrations": ["cursor"]},
                ["port"],
            ),
            (
                dict(mode="stdio"),
                {},
                ["port"],
            ),
            (
                dict(mode="http", port=9000),
                {"port": 9000},
                [],
            ),
        ],
        ids=[
            "empty_ide_omitted",
            "non_empty_ide_present",
            "stdio_no_port",
            "http_with_port",
        ],
    )
    def test_model_dump(
        self,
        kwargs: dict,
        expect_in: dict,
        expect_not_in: list[str],
    ) -> None:
        data = RunConfig(**kwargs).model_dump()
        for key, val in expect_in.items():
            assert data[key] == val
        for key in expect_not_in:
            assert key not in data

    def test_extra_fields_ignored_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "tools.yaml"
        path.write_text("mode: http\nport: 5000\nfuture_key: value\n")
        cfg = RunConfig.load(path)
        assert cfg.mode == "http"
        assert cfg.port == 5000
        assert not hasattr(cfg, "future_key")

    # Persistence ======================================================================

    def test_save_creates_directory_and_file(self, tmp_path: Path) -> None:
        cfg = RunConfig(mode="http", port=1234)
        dest = tmp_path / "nested" / "dir" / "tools.yaml"
        cfg.save(dest)
        assert dest.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = RunConfig(mode="http", port=4567)
        dest = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        cfg.save(dest)
        loaded = RunConfig.load(dest)
        assert loaded.mode == "http"
        assert loaded.port == 4567

    def test_roundtrip_with_new_fields(self, tmp_path: Path) -> None:
        cfg = RunConfig(
            mode="stdio",
            port=7331,
            project_path="/my/bot",
            docs_mode="online",
            ide_integrations=["cursor", "vscode"],
        )
        dest = tmp_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        cfg.save(dest)
        loaded = RunConfig.load(dest)
        assert loaded.project_path == "/my/bot"
        assert loaded.docs_mode == "online"
        assert loaded.ide_integrations == ["cursor", "vscode"]

    def test_defaults(self) -> None:
        cfg = RunConfig()
        assert cfg.project_path == "."
        assert cfg.docs_mode == "offline"
        assert cfg.ide_integrations == []
