"""Shared constants for the `rasa tools` CLI command family.

Duplicated from `rasa.builder.copilot.*` to avoid importing heavy
dependencies at CLI startup.  Keep these in sync when updating.
"""

import re
from typing import Dict, Tuple

from prompt_toolkit.styles import Style

# MCP server defaults ==================================================================

MCP_TOOLS_DEFAULT_PORT = 7331
MCP_TOOLS_DEFAULT_HOST = "127.0.0.1"
MCP_TOOLS_TRANSPORT_STDIO = "stdio"
MCP_TOOLS_TRANSPORT_HTTP = "http"
MCP_TOOLS_TRANSPORT_STREAMABLE_HTTP = "streamable-http"
MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR = "RASA_PROJECT_FOLDER"
MCP_TOOLS_HTTP_URL_PATTERN = "http://{host}:{port}/mcp"
MCP_TOOLS_HTTP_HEALTH_URL_PATTERN = "http://{host}:{port}/health"

# Must match the default Rasa server URL (see DEFAULT_SERVER_PORT in
# rasa/core/constants.py).
DEFAULT_RASA_SERVER_URL = "http://localhost:5005"

# Documentation modes ==================================================================

DOCS_MODE_OFFLINE = "offline"
DOCS_MODE_ONLINE = "online"
DOCS_MODES: Tuple[str, ...] = (DOCS_MODE_OFFLINE, DOCS_MODE_ONLINE)

# IDE support ==========================================================================

SUPPORTED_IDES: Tuple[str, ...] = ("cursor", "vscode", "claude", "jetbrains")

# Base directory within the project root where each IDE loads skills from.
# All IDEs use the same layout: <base>/<skill-name>/SKILL.md
IDE_SKILLS_BASE: Dict[str, str] = {
    "cursor": ".cursor/skills",
    "vscode": ".github/skills",
    "claude": ".claude/skills",
    # JetBrains has no standard skills location.
}

IDE_DISPLAY_NAMES: Dict[str, str] = {
    "cursor": "Cursor",
    "vscode": "VS Code (GitHub Copilot)",
    "claude": "Claude Code",
    "jetbrains": "JetBrains IDEs (IntelliJ, WebStorm, PyCharm, etc.)",
}

# Tools config =========================================================================

TOOLS_CONFIG_DIR = ".rasa"
TOOLS_CONFIG_FILENAME = "tools.yaml"

# Offline docs / skills fetching =======================================================

DEFAULT_LLMS_TXT_BASE_URL = "https://rasa.com/docs"
LLMS_TXT_BASE_URL_ENV_VAR = "RASA_LLMS_TXT_BASE_URL"

RASA_TOOLS_PROXY_BASE_URL_ENV_VAR = "RASA_TOOLS_PROXY_URL"

HTTP_TIMEOUT = 30
LLMS_TXT_FILES: Tuple[str, ...] = ("llms.txt", "llms-full.txt")

AGENT_SKILLS_REPO = "RasaHQ/rasa-agent-skills"
AGENT_SKILLS_API_URL = "https://api.github.com/repos/{repo}/contents/skills"
AGENT_SKILLS_RAW_URL = (
    "https://raw.githubusercontent.com/{repo}/main/skills/{skill}/SKILL.md"
)

# Wizard UI style ======================================================================

WIZARD_STYLE = Style(
    [
        ("qmark", "fg:ansicyan bold"),
        ("question", "bold"),
        ("answer", "fg:ansigreen bold"),
        ("pointer", "fg:ansicyan bold"),
        ("highlighted", "fg:ansicyan bold"),
        ("selected", "fg:ansigreen"),
    ]
)

# Agent skills helpers =================================================================

AGENT_SKILL_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
AGENT_SKILL_RASA_VERSION_RE = re.compile(r"rasa_version:\s*[\"']?([><=!~]*)([\d.]+)")
AGENT_SKILL_VERSION_RE = re.compile(r"(?<!rasa_)version:\s*[\"']?([\d.]+)")
AGENT_SKILLS_VERSION_SPECIFIER_PREFIX_RE = re.compile(r"^([><=!~]+)?\s*(.+)$")
