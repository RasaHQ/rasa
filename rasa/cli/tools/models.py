from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
)

from rasa.cli.tools.constants import (
    AGENT_SKILL_FRONTMATTER_RE,
    AGENT_SKILL_RASA_VERSION_RE,
    AGENT_SKILL_VERSION_RE,
    AGENT_SKILLS_VERSION_SPECIFIER_PREFIX_RE,
    DEFAULT_RASA_SERVER_URL,
    DOCS_MODE_OFFLINE,
    MCP_TOOLS_DEFAULT_PORT,
    MCP_TOOLS_TRANSPORT_HTTP,
    MCP_TOOLS_TRANSPORT_STDIO,
    SUPPORTED_IDES,
)
from rasa.shared.exceptions import RasaException


class AgentSkillInfo(BaseModel):
    """Parsed metadata and content for a single agent skill."""

    name: str
    version: Optional[str] = None
    rasa_version: Optional[str] = None
    content: str

    @classmethod
    def from_content(cls, name: str, content: str) -> "AgentSkillInfo":
        """Create a `AgentSkillInfo` by parsing frontmatter from raw text."""
        return cls(
            name=name,
            version=cls._parse_skill_version(content),
            rasa_version=cls._parse_rasa_version(content),
            content=content,
        )

    def is_compatible_with(self, current_version: str) -> bool:
        """Return `True` if this skill works with *current_version*.

        Only major and minor components are compared so that patch-level differences are
        ignored (e.g. a skill tagged `>=3.16.10` is compatible with `3.16.2`).

        The `rasa_version` field may contain any PEP 440 version specifier prefix
        (`>=`, `<=`, `~=`, etc.).  A bare version is treated as `>=`.
        """
        if not self.rasa_version:
            return True
        try:
            specifier_match = AGENT_SKILLS_VERSION_SPECIFIER_PREFIX_RE.match(
                self.rasa_version.strip()
            )
            if not specifier_match:
                return True
            op = specifier_match.group(1) or ">="
            skill_ver = Version(specifier_match.group(2))
            curr_ver = Version(current_version)
            # Keep 2-part versions as-is so ~= retains correct PEP 440 semantics:
            # ~=3.16 means >=3.16, <4.0  but  ~=3.16.0 means >=3.16.0, <3.17.0
            if len(skill_ver.release) >= 3:
                normalized_skill_ver = f"{skill_ver.major}.{skill_ver.minor}.0"
            else:
                normalized_skill_ver = f"{skill_ver.major}.{skill_ver.minor}"
            normalized_curr_ver = Version(f"{curr_ver.major}.{curr_ver.minor}.0")
            return normalized_curr_ver in SpecifierSet(f"{op}{normalized_skill_ver}")
        except (InvalidVersion, InvalidSpecifier):
            return True

    @staticmethod
    def _parse_rasa_version(content: str) -> Optional[str]:
        """Extract the required Rasa version specifier from SKILL.md frontmatter.

        Looks for a `rasa_version` field inside the YAML frontmatter
        (delimited by `---`) and returns a PEP 440-style version
        specifier (e.g. `">=3.9.0"`).  A bare version with no operator
        prefix is returned with an implicit `>=`.

        Args:
            content: Full text of a SKILL.md file.

        Returns:
            The specifier string, or `None` if the field is missing
            or cannot be parsed.
        """
        fm_match = AGENT_SKILL_FRONTMATTER_RE.match(content)
        if not fm_match:
            return None

        ver_match = AGENT_SKILL_RASA_VERSION_RE.search(fm_match.group(1))
        if not ver_match:
            return None

        op = ver_match.group(1)
        version = ver_match.group(2)
        return f"{op}{version}" if op else f">={version}"

    @staticmethod
    def _parse_skill_version(content: str) -> Optional[str]:
        """Extract the skill version from SKILL.md frontmatter.

        Args:
            content: Full text of a SKILL.md file.

        Returns:
            The version string (e.g. `"0.1.0"`), or `None` if the
            field is missing or cannot be parsed.
        """
        fm_match = AGENT_SKILL_FRONTMATTER_RE.match(content)
        if not fm_match:
            return None

        ver_match = AGENT_SKILL_VERSION_RE.search(fm_match.group(1))
        if not ver_match:
            return None

        return ver_match.group(1)


class RunConfig(BaseModel):
    """Persisted run parameters for `rasa tools run`.

    Add new fields with sensible defaults to extend.  Save, load, and display
    adapt automatically — no other code changes required.
    """

    model_config = {"extra": "ignore"}

    mode: Literal["stdio", "http"] = Field(default=MCP_TOOLS_TRANSPORT_STDIO)
    port: int = Field(default=MCP_TOOLS_DEFAULT_PORT)
    project_path: str = Field(default=".")
    docs_mode: Literal["offline", "online"] = Field(default=DOCS_MODE_OFFLINE)
    ide_integrations: List[str] = Field(default_factory=list)
    rasa_server_url: Optional[str] = Field(default=DEFAULT_RASA_SERVER_URL)

    @field_validator("ide_integrations", mode="before")
    @classmethod
    def _validate_ide_values(cls, value: Any) -> List[str]:
        # A bare YAML scalar (e.g. `ide_integrations: cursor`) arrives as a
        # string instead of a list.
        if isinstance(value, str):
            value = [value]

        invalid = [v for v in value if v not in SUPPORTED_IDES]
        if invalid:
            raise ValueError(
                f"Unsupported IDE(s): {', '.join(invalid)}. "
                f"Accepted values: {', '.join(SUPPORTED_IDES)}"
            )
        return value

    @model_serializer
    def _serialize_model(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "mode": self.mode,
            "project_path": self.project_path,
            "docs_mode": self.docs_mode,
        }
        if self.mode == MCP_TOOLS_TRANSPORT_HTTP:
            data["port"] = self.port
        if self.ide_integrations:
            data["ide_integrations"] = self.ide_integrations
        if self.rasa_server_url and self.rasa_server_url != DEFAULT_RASA_SERVER_URL:
            data["rasa_server_url"] = self.rasa_server_url
        return data

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        """Load config from *path*.

        Raises:
            RasaException: If the file does not exist or contains invalid data.
        """
        if not path.is_file():
            raise RasaException(f"Config file does not exist: {path}")

        try:
            from rasa.shared.utils.yaml import read_yaml_file

            data = read_yaml_file(str(path))
            return cls.model_validate(data)
        # ValidationError is raised if the data is not a valid RunConfig
        except ValidationError as exc:
            raise RasaException(f"Invalid config in {path}: {exc}") from exc
        # Exception is raised if the file is not a valid YAML file
        except Exception as exc:
            raise RasaException(f"Failed to read config file {path}.") from exc

    def save(self, path: Path) -> None:
        """Persist this config to *path*, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        from rasa.shared.utils.yaml import write_yaml

        write_yaml(self.model_dump(), path)
