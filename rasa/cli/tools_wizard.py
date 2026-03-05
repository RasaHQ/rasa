"""Interactive setup wizard for `rasa tools init`.

Collects configuration values through a step-by-step terminal UI
(or via CLI flags in non-interactive mode), persists them to
`.rasa/tools.yaml`, optionally fetches the offline docs bundle,
and writes IDE-specific MCP configuration files.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import questionary
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from rasa.cli.arguments.tools import (
    DOCS_MODE_OFFLINE,
    DOCS_MODE_ONLINE,
    MCP_TOOLS_DEFAULT_PORT,
    MCP_TOOLS_TRANSPORT_HTTP,
    MCP_TOOLS_TRANSPORT_STDIO,
    SUPPORTED_IDES,
)
from rasa.cli.tools import (
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
    RunConfig,
    _resolve_project_dir,
)

_DEFAULT_LLMS_TXT_BASE_URL = "https://rasa.com/docs"
LLMS_TXT_BASE_URL_ENV_VAR = "RASA_LLMS_TXT_BASE_URL"

_HELLO_LLM_PROXY_URL = "https://hello-llm-proxy.rasa-e2e.workers.dev"
_HTTP_TIMEOUT = 30
_LLMS_TXT_FILES = ("llms.txt", "llms-full.txt")

_AGENT_SKILLS_REPO = "RasaHQ/rasa-agent-skills"
_AGENT_SKILLS_API_URL = "https://api.github.com/repos/{repo}/contents/skills"
_AGENT_SKILLS_RAW_URL = (
    "https://raw.githubusercontent.com/{repo}/main/skills/{skill}/SKILL.md"
)

# Base directory within the project root where each IDE loads skills from.
# All IDEs use the same layout: <base>/<skill-name>/SKILL.md
_IDE_SKILLS_BASE: Dict[str, str] = {
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

console = Console()


# ── Public entry point ────────────────────────────────────────────────────────


def run_wizard(args: argparse.Namespace) -> None:
    """Run the `rasa tools init` setup wizard.

    Orchestrates the full init flow: resolves paths, collects configuration
    (interactively or from CLI flags), persists it, fetches offline docs if
    requested, writes IDE config files, and prints a summary.

    Args:
        args: Parsed CLI arguments from argparse (includes `yes`,
            `project_path`, `mode`, `port`, `docs`, `ides`).
    """
    console.print(
        Panel(
            "[bold cyan]Rasa Tools Setup Wizard[/bold cyan]\n"
            "Configure your local MCP server for IDE copilots.",
            border_style="cyan",
            expand=False,
        )
    )

    non_interactive = getattr(args, "yes", False)

    project_dir = _resolve_project_dir(getattr(args, "project_path", None))
    config_path = project_dir / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME

    if config_path.exists() and not _confirm_overwrite(non_interactive):
        return

    config = (
        _run_non_interactive(args, project_dir)
        if non_interactive
        else _run_interactive(project_dir)
    )

    config.save(config_path)
    console.print(f"\n[green]✔[/green] Saved config → [bold]{config_path}[/bold]")

    if config.docs_mode == DOCS_MODE_OFFLINE:
        _fetch_offline_docs(project_dir)
    elif config.docs_mode == DOCS_MODE_ONLINE:
        _warn_if_offline_docs_exist(project_dir)

    _write_ide_configs(project_dir, config)

    # Skills install is only offered when at least one IDE was configured.
    if config.ide_integrations:
        install_skills = (
            getattr(args, "skills", False)
            if non_interactive
            else _ask_install_agent_skills()
        )
        if install_skills:
            _install_agent_skills(project_dir, config.ide_integrations)
        else:
            _warn_if_agent_skills_exist(project_dir, config.ide_integrations)

    _print_summary(config, config_path)


# ── Overwrite guard ───────────────────────────────────────────────────────────


def _confirm_overwrite(non_interactive: bool) -> bool:
    """Ask the user whether to overwrite an existing configuration file.

    Args:
        non_interactive: When `True`, automatically confirm the overwrite
            without prompting (`--yes` mode).

    Returns:
        `True` if the wizard should proceed and overwrite, `False` to abort.
    """
    if non_interactive:
        # --yes mode: always overwrite without prompting.
        return True

    proceed = questionary.confirm(
        "Configuration already exists. Overwrite saved values?",
        default=False,
        style=WIZARD_STYLE,
    ).ask()

    if proceed is None:
        _abort()

    if not proceed:
        console.print("[yellow]Aborted.[/yellow] Keeping existing configuration.")

    return bool(proceed)


# ── Interactive flow ──────────────────────────────────────────────────────────


def _run_interactive(project_dir: Path) -> RunConfig:
    """Walk the user through each configuration step interactively.

    Prompts for transport mode, port (if HTTP), documentation mode, and
    IDE selections using the terminal UI.

    Args:
        project_dir: Absolute path to the Rasa project root.

    Returns:
        A fully populated `RunConfig` built from the user's answers.
    """
    console.print(f"\n[green]✔[/green] Project directory: [bold]{project_dir}[/bold]\n")

    mode = _ask_mode()

    # Start with fields that are always required.
    config_kwargs: Dict[str, Any] = {
        "mode": mode,
        "project_path": str(project_dir),
    }

    # Port is only relevant for HTTP mode. Ask immediately after mode selection.
    if mode == MCP_TOOLS_TRANSPORT_HTTP:
        port = _ask_port()
        config_kwargs["port"] = port

    docs_mode = _ask_docs_mode()
    ides = _ask_ides()

    config_kwargs["docs_mode"] = docs_mode
    config_kwargs["ide_integrations"] = ides

    return RunConfig(**config_kwargs)


def _ask_mode() -> str:
    """Prompt the user to select the MCP server transport mode.

    Returns:
        The selected transport mode string (`"stdio"` or `"http"`).
    """
    answer = questionary.select(
        "Transport mode",
        choices=[
            questionary.Choice(
                "stdio   (recommended for IDEs)", value=MCP_TOOLS_TRANSPORT_STDIO
            ),
            questionary.Choice(
                "http    (network listener for debugging)",
                value=MCP_TOOLS_TRANSPORT_HTTP,
            ),
        ],
        style=WIZARD_STYLE,
    ).ask()
    if answer is None:
        _abort()
    return answer


def _ask_port() -> int:
    """Prompt the user to enter the HTTP port number.

    Returns:
        A valid port number.
    """
    answer = questionary.text(
        "Port",
        default=str(MCP_TOOLS_DEFAULT_PORT),
        validate=lambda val: (
            True
            if val.isdigit() and 1 <= int(val) <= 65535
            else "Enter a valid port (1–65535)"
        ),
        style=WIZARD_STYLE,
    ).ask()
    if answer is None:
        _abort()
    return int(answer)


def _ask_docs_mode() -> str:
    """Prompt the user to select the documentation mode.

    Returns:
        The selected docs mode string ("offline" or "online").
    """
    answer = questionary.select(
        "Documentation mode",
        choices=[
            questionary.Choice(
                "Offline  (recommended — local llms.txt bundle)",
                value=DOCS_MODE_OFFLINE,
            ),
            questionary.Choice(
                "Online   (calls Rasa Docs API at runtime)",
                value=DOCS_MODE_ONLINE,
            ),
        ],
        style=WIZARD_STYLE,
    ).ask()
    if answer is None:
        _abort()
    return answer


def _ask_ides() -> List[str]:
    """Prompt the user to select which IDEs to configure.

    Returns:
        List of selected IDE identifiers (e.g. `["cursor", "vscode"]`).
    """
    choices = [
        questionary.Choice(
            IDE_DISPLAY_NAMES[ide],
            value=ide,
        )
        for ide in SUPPORTED_IDES
    ]
    answer = questionary.checkbox(
        "Which IDEs do you use?",
        choices=choices,
        style=WIZARD_STYLE,
    ).ask()
    if answer is None:
        _abort()
    return answer


def _abort() -> None:
    """Print an abort message and exit the process.

    Called when the user cancels an interactive prompt (e.g. Ctrl+C).
    """
    console.print("\n[yellow]Aborted.[/yellow]")
    sys.exit(1)


# ── Non-interactive flow ──────────────────────────────────────────────────────


def _run_non_interactive(args: argparse.Namespace, project_dir: Path) -> RunConfig:
    """Build a `RunConfig` entirely from CLI flags and defaults.

    Used when the ``--yes`` flag is passed to skip interactive prompts.

    Args:
        args: Parsed CLI arguments from argparse.
        project_dir: Absolute path to the Rasa project root.

    Returns:
        A fully populated `RunConfig`.
    """
    ides: List[str] = []
    if raw := getattr(args, "ides", None):
        # Accept a comma-separated list, e.g. "cursor,vscode".
        ides = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]

    mode = getattr(args, "mode", None) or MCP_TOOLS_TRANSPORT_STDIO

    config_kwargs: Dict[str, Any] = {
        "mode": mode,
        "project_path": str(project_dir),
        "docs_mode": getattr(args, "docs", None) or DOCS_MODE_OFFLINE,
        "ide_integrations": ides,
    }

    # Port is only written when HTTP mode is active. Omitted for stdio.
    if mode == MCP_TOOLS_TRANSPORT_HTTP:
        cli_port = getattr(args, "port", None)
        config_kwargs["port"] = (
            cli_port if cli_port is not None else MCP_TOOLS_DEFAULT_PORT
        )

    return RunConfig(**config_kwargs)


# ── Offline docs fetching ────────────────────────────────────────────────────


def _resolve_llms_txt_base_url() -> str:
    """Return the base URL for fetching llms.txt documentation files.

    Reads from the `RASA_LLMS_TXT_BASE_URL` environment variable if set,
    otherwise falls back to the default Rasa documentation URL.

    Returns:
        Base URL string (without trailing slash).
    """
    url = os.getenv(LLMS_TXT_BASE_URL_ENV_VAR, _DEFAULT_LLMS_TXT_BASE_URL)
    return url.rstrip("/")


def _fetch_offline_docs(project_dir: Path) -> None:
    """Download the offline docs bundle (llms.txt files) into the config directory.

    Fetches each file listed in `_LLMS_TXT_FILES` from the resolved base URL
    and saves them under `<project_dir>/.rasa/`.  Failures are reported but
    do not abort the wizard.

    Args:
        project_dir: Absolute path to the Rasa project root.
    """
    base_url = _resolve_llms_txt_base_url()
    dest_dir = project_dir / TOOLS_CONFIG_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    for filename in _LLMS_TXT_FILES:
        url = f"{base_url}/{filename}"
        dest = dest_dir / filename
        console.print(f"  Fetching {filename}…", end=" ")
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as response:
                dest.write_bytes(response.read())
            console.print("[green]✔[/green]")
            console.print(f"  Saved to [bold]{dest}[/bold]")
        except Exception as exc:
            console.print("[red]✖[/red]")
            console.print(
                f"  [yellow]Could not fetch {filename}:[/yellow] {exc}\n"
                f"  Download manually from [link]{url}[/link]\n"
                f"  and place it at {dest}"
            )


def _warn_if_offline_docs_exist(project_dir: Path) -> None:
    """Warn the user if offline docs files exist when switching to online mode.

    Checks for the presence of any llms.txt files in the config directory and
    displays a warning panel if found. The files are not deleted automatically.

    Args:
        project_dir: Absolute path to the Rasa project root.
    """
    dest_dir = project_dir / TOOLS_CONFIG_DIR
    existing_files = [
        filename for filename in _LLMS_TXT_FILES if (dest_dir / filename).exists()
    ]

    if existing_files:
        file_list = "\n".join(f"  • {dest_dir / f}" for f in existing_files)
        console.print()
        console.print(
            Panel(
                "[yellow]⚠[/yellow]  [bold]Offline docs files detected[/bold]\n\n"
                "You selected online documentation mode, "
                "but these offline files exist:\n\n"
                f"{file_list}\n\n"
                "If left in the project, the agent may still read and use them. "
                "Delete them to ensure only online documentation is used.",
                border_style="yellow",
                expand=False,
            )
        )


# ── Agent skills installation ─────────────────────────────────────────────────


def _ask_install_agent_skills() -> bool:
    """Ask the user whether to install Rasa agent skills for the selected IDEs.

    Returns:
        `True` if the user wants to install agent skills, `False` otherwise.
    """
    answer = questionary.confirm(
        "Install Rasa agent skills for your selected IDEs?",
        default=True,
        style=WIZARD_STYLE,
    ).ask()

    if answer is None:
        _abort()

    return bool(answer)


def _fetch_skill_names() -> List[str]:
    """Fetch the list of available skill names from the agent skills repository.

    Returns:
        List of skill directory names (e.g. `["rasa-building-flows", ...]`).
        Returns an empty list if the fetch fails.
    """
    url = _AGENT_SKILLS_API_URL.format(repo=_AGENT_SKILLS_REPO)

    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as response:
            entries = json.loads(response.read())
        # Each skill is a directory. The API also returns non-skill files (e.g. README).
        return [e["name"] for e in entries if e["type"] == "dir"]
    except Exception as exc:
        console.print(f"  [red]✖[/red] Could not fetch skill list: {exc}")
        return []


def _install_agent_skills(project_dir: Path, ides: List[str]) -> None:
    """Download and install Rasa agent skills for each selected IDE.

    Fetches each skill's `SKILL.md` from the agent skills repository and
    writes it to `<base>/<skill-name>/SKILL.md` under the project root.
    IDEs that have no standard skills location (JetBrains) are skipped.

    Args:
        project_dir: Absolute path to the Rasa project root.
        ides: List of IDE identifiers to install skills for.
    """
    console.print()
    skill_names = _fetch_skill_names()
    if not skill_names:
        return

    # Only keep IDEs that have a known skills base directory.
    destinations: Dict[str, Path] = {
        ide: project_dir / base for ide, base in _IDE_SKILLS_BASE.items() if ide in ides
    }

    if not destinations:
        console.print(
            "  [yellow]ℹ[/yellow]  No IDEs with a supported skills location selected."
        )
        return

    console.print(f"  Installing [bold]{len(skill_names)}[/bold] agent skills…")

    for skill in skill_names:
        url = _AGENT_SKILLS_RAW_URL.format(repo=_AGENT_SKILLS_REPO, skill=skill)
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as response:
                content = response.read().decode("utf-8")
        except Exception as exc:
            # Log the failure but continue installing the remaining skills.
            console.print(f"  [red]✖[/red] {skill}: {exc}")
            continue

        # All IDEs share the same layout: <base>/<skill-name>/SKILL.md
        for dest_dir in destinations.values():
            dest = dest_dir / skill / "SKILL.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        ide_labels = ", ".join(IDE_DISPLAY_NAMES.get(ide, ide) for ide in destinations)
        console.print(f"  [green]✔[/green] {skill}  →  {ide_labels}")


def _warn_if_agent_skills_exist(project_dir: Path, ides: List[str]) -> None:
    """Warn the user if agent skills directories exist when skipping installation.

    Checks for non-empty skills directories for the selected IDEs and displays
    a warning panel if any are found. The directories are not deleted automatically.

    Args:
        project_dir: Absolute path to the Rasa project root.
        ides: List of IDE identifiers that were selected.
    """
    existing_dirs = [
        project_dir / base
        for ide, base in _IDE_SKILLS_BASE.items()
        if ide in ides
        and (project_dir / base).is_dir()
        and any((project_dir / base).iterdir())
    ]

    if existing_dirs:
        dir_list = "\n".join(f"  • {d}" for d in existing_dirs)
        console.print()
        console.print(
            Panel(
                "[yellow]⚠[/yellow]  [bold]Agent skills detected[/bold]\n\n"
                "You chose not to install agent skills, "
                "but these skill directories already exist:\n\n"
                f"{dir_list}\n\n"
                "If left in the project, your IDE may still load them. "
                "Delete them manually if you no longer need them.",
                border_style="yellow",
                expand=False,
            )
        )


# ── IDE configuration writers ─────────────────────────────────────────────────


def _write_ide_configs(project_dir: Path, config: RunConfig) -> None:
    """Write MCP configuration files for each selected IDE.

    Delegates to IDE-specific writers registered in `_IDE_CONFIG_WRITERS`.

    Args:
        project_dir: Absolute path to the Rasa project root.
        config: The resolved run configuration.
    """
    if not config.ide_integrations:
        return

    console.print()
    for ide in config.ide_integrations:
        writer = _IDE_CONFIG_WRITERS.get(ide)
        if writer:
            writer(project_dir, config)


def _build_stdio_entry(project_dir: Path) -> Dict[str, Any]:
    """Build an MCP config entry for stdio transport.

    Uses the current Python interpreter to run Rasa as a module, ensuring
    the correct virtual environment is used.  Includes environment variables
    required at runtime (license, LLM proxy URL) so that IDE-spawned
    processes work without relying on the user's shell environment.

    Args:
        project_dir: Absolute path to the Rasa project root.

    Returns:
        Dict with `command`, `args`, and `env` keys for launching the server.
    """
    from rasa.utils.licensing import LICENSE_ENV_VAR, retrieve_license_from_env

    license_value, _ = retrieve_license_from_env()

    return {
        "command": sys.executable,
        "args": [
            "-m",
            "rasa",
            "tools",
            "run",
            "--mode",
            "stdio",
            "--project-path",
            str(project_dir),
        ],
        "env": {
            LICENSE_ENV_VAR: license_value,
            "HELLO_LLM_PROXY_BASE_URL": _HELLO_LLM_PROXY_URL,
        },
    }


def _build_http_entry(port: int) -> Dict[str, Any]:
    """Build an MCP config entry for HTTP transport.

    Args:
        port: The port number for the HTTP listener.

    Returns:
        Dict with a `url` key pointing to the MCP endpoint.
    """
    return {
        "url": f"http://127.0.0.1:{port}/mcp",
    }


def _write_cursor_config(project_dir: Path, config: RunConfig) -> None:
    """Write or update the Cursor IDE MCP configuration.

    Args:
        project_dir: Absolute path to the Rasa project root.
        config: The resolved run configuration.
    """
    path = project_dir / ".cursor" / "mcp.json"
    if config.mode == MCP_TOOLS_TRANSPORT_STDIO:
        entry = _build_stdio_entry(project_dir)
    else:
        entry = _build_http_entry(config.port)
    _merge_mcp_json(path, entry, wrapper_key="mcpServers")
    console.print(f"  [green]✔[/green] Cursor         → {path}")


def _write_vscode_config(project_dir: Path, config: RunConfig) -> None:
    """Write or update the VS Code (GitHub Copilot) MCP configuration.

    Args:
        project_dir: Absolute path to the Rasa project root.
        config: The resolved run configuration.
    """
    path = project_dir / ".vscode" / "mcp.json"
    # VS Code's MCP schema requires an explicit "type" field that Cursor omits.
    if config.mode == MCP_TOOLS_TRANSPORT_STDIO:
        entry: Dict[str, Any] = {
            "type": MCP_TOOLS_TRANSPORT_STDIO,
            **_build_stdio_entry(project_dir),
        }
    else:
        entry = {
            "type": "http",
            **_build_http_entry(config.port),
        }
    _merge_mcp_json(path, entry, wrapper_key="servers")
    console.print(f"  [green]✔[/green] VS Code        → {path}")


def _write_claude_config(project_dir: Path, config: RunConfig) -> None:
    """Write or update the Claude Code MCP configuration.

    Args:
        project_dir: Absolute path to the Rasa project root.
        config: The resolved run configuration.
    """
    path = project_dir / ".mcp.json"
    if config.mode == MCP_TOOLS_TRANSPORT_STDIO:
        entry: Dict[str, Any] = _build_stdio_entry(project_dir)
    else:
        # Claude Code expects a "type" field alongside the url for HTTP.
        entry = {
            "type": "http",
            **_build_http_entry(config.port),
        }
    _merge_mcp_json(path, entry, wrapper_key="mcpServers")
    console.print(f"  [green]✔[/green] Claude Code    → {path}")


def _write_jetbrains_config(project_dir: Path, config: RunConfig) -> None:
    """Print the JetBrains IDEs MCP configuration snippet for manual setup.

    JetBrains IDEs (IntelliJ, WebStorm, PyCharm, etc.) do not use a
    project-level JSON file; the user must paste the snippet into IDE
    settings manually.

    Args:
        project_dir: Absolute path to the Rasa project root.
        config: The resolved run configuration.
    """
    if config.mode == MCP_TOOLS_TRANSPORT_STDIO:
        entry = _build_stdio_entry(project_dir)
    else:
        entry = _build_http_entry(config.port)

    snippet = json.dumps({"mcpServers": {"rasa-tools": entry}}, indent=2)
    console.print(
        "  [yellow]ℹ[/yellow]  JetBrains IDEs → Manual configuration required"
    )
    console.print(
        "\n  [bold]Instructions:[/bold]\n"
        "  1. Open IDE Settings → Tools → AI Assistant → MCP\n"
        "  2. Copy the JSON below\n"
        "  3. Paste into the MCP configuration field\n"
    )
    console.print(snippet)
    console.print()


# Dispatch table: maps IDE identifier → writer function.
# Add new IDEs here without touching _write_ide_configs.
_IDE_CONFIG_WRITERS = {
    "cursor": _write_cursor_config,
    "vscode": _write_vscode_config,
    "claude": _write_claude_config,
    "jetbrains": _write_jetbrains_config,
}


# ── JSON merge helper ─────────────────────────────────────────────────────────


def _merge_mcp_json(path: Path, entry: Dict[str, Any], *, wrapper_key: str) -> None:
    """Read existing JSON (if any), upsert the `rasa-tools` entry, and write back.

    Other MCP servers already configured in the file are left untouched.

    Args:
        path: Target JSON file path (created if missing, along with parents).
        entry: The MCP server configuration dict to store under `rasa-tools`.
        wrapper_key: Top-level key that holds the server map
            (e.g. `"mcpServers"` or `"servers"`).
    """
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            parsed = json.loads(path.read_text())
            # A valid JSON file whose top-level value is not an object (e.g. an
            # array, string, or number) cannot hold server entries — treat it as
            # empty so the upsert below never hits an AttributeError.
            existing = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, OSError):
            # Treat a corrupt or unreadable file as empty rather than raising.
            existing = {}

    # If the wrapper key exists but is not a dict (e.g. an array), reset it so
    # the upsert below doesn't raise a TypeError.
    if not isinstance(existing.get(wrapper_key), dict):
        existing[wrapper_key] = {}

    # Upsert: create or replace only the rasa-tools entry. Leave others intact.
    existing[wrapper_key]["rasa-tools"] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n")


# ── Summary panel ─────────────────────────────────────────────────────────────


def _print_summary(config: RunConfig, config_path: Path) -> None:
    """Print a rich summary panel showing the final configuration.

    Args:
        config: The saved run configuration.
        config_path: Path where the configuration file was written.
    """
    lines = Text()
    lines.append("Config  ", style="bold")
    lines.append(f"{config_path}\n")
    lines.append("Mode    ", style="bold")
    lines.append(f"{config.mode}\n")
    if config.mode == MCP_TOOLS_TRANSPORT_HTTP:
        lines.append("Port    ", style="bold")
        lines.append(f"{config.port}\n")
    lines.append("Docs    ", style="bold")
    lines.append(f"{config.docs_mode.capitalize()}\n")
    if config.ide_integrations:
        lines.append("IDEs    ", style="bold")
        names = ", ".join(
            IDE_DISPLAY_NAMES.get(ide, ide) for ide in config.ide_integrations
        )
        lines.append(f"{names}\n")

    console.print()
    console.print(
        Panel(
            lines,
            title="[bold green]Setup Complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    if config.mode == MCP_TOOLS_TRANSPORT_HTTP:
        next_step = "\n  Start the MCP server: [cyan]rasa tools run[/cyan]\n"
    else:
        next_step = "\n  Start the server from your IDE.\n"
    console.print("\n[bold]Next:[/bold]" + next_step)
