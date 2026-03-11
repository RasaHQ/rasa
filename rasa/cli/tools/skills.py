"""``rasa tools init skills`` — fetch and install Rasa agent skills."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import questionary
from rich.console import Console
from rich.panel import Panel

from rasa.cli.tools.constants import (
    AGENT_SKILLS_API_URL,
    AGENT_SKILLS_RAW_URL,
    AGENT_SKILLS_REPO,
    HTTP_TIMEOUT,
    IDE_DISPLAY_NAMES,
    IDE_SKILLS_BASE,
)
from rasa.cli.tools.models import AgentSkillInfo
from rasa.cli.tools.utils import _precheck, _resolve_ides, _resolve_project_dir
from rasa.version import __version__ as rasa_version

console = Console()

# Entrypoint ===========================================================================


def skills_tools(args: argparse.Namespace) -> None:
    """Entrypoint for `rasa tools init skills`.

    Args:
        args: The CLI arguments.
    """
    _precheck()

    project_dir = _resolve_project_dir(
        getattr(args, "project_path", None), validate_exists=True
    )

    ides = _resolve_ides(getattr(args, "ides", None), project_dir)
    if not ides:
        console.print(
            "[yellow]No IDEs specified.[/yellow]\n"
            "Pass [bold]--ides cursor,vscode[/bold] or run "
            "[bold]rasa tools init[/bold] first to save IDE preferences."
        )
        sys.exit(1)

    non_interactive = getattr(args, "yes", False)
    install_agent_skills(project_dir, ides, non_interactive=non_interactive)


# Orchestrators ========================================================================


def install_agent_skills(
    project_dir: Path,
    ides: List[str],
    non_interactive: bool = False,
) -> None:
    """Download and install Rasa agent skills for each selected IDE.

    Performs safety checks before writing:
    1. Warns if skills already exist and asks to continue.
    2. Checks `rasa_version` in each skill's frontmatter and warns about skills that
       require a newer Rasa version.

    Args:
        project_dir: Absolute path to the Rasa project root.
        ides: List of IDE identifiers to install skills for.
        non_interactive: When `True`, skip confirmation prompts overwrite existing,
            and skip incompatible skills silently.
    """
    console.print()

    skill_names = _fetch_skill_names_from_repo()
    if not skill_names:
        return

    destinations: Dict[str, Path] = {
        ide: project_dir / base for ide, base in IDE_SKILLS_BASE.items() if ide in ides
    }
    if not destinations:
        console.print(
            "  [yellow]ℹ[/yellow]  "
            "No IDEs with a supported skills location selected."
        )
        return

    # Check for existing skills
    existing = _find_existing_skills(destinations)
    if existing and not non_interactive:
        if not _confirm_overwrite_skills(existing):
            console.print("  Skipping skill installation.")
            return

    # Download all skills and parse frontmatter
    console.print(f"  Downloading [bold]{len(skill_names)}[/bold] agent skills…")
    downloaded_skills = _download_skills_from_repo(skill_names)
    if not downloaded_skills:
        return

    # Version compatibility check
    (
        compatible_skills,
        incompatible_skills,
    ) = _partition_by_compatibility(downloaded_skills)

    install_incompatible_skills = False
    if incompatible_skills:
        _warn_incompatible_skills(incompatible_skills, rasa_version)
        if non_interactive:
            console.print("  Skipping incompatible skills " "(non-interactive mode).")
        else:
            install_incompatible_skills = _confirm_install_incompatible()

    # Write skills to disk
    skills_to_install = list(compatible_skills)
    if install_incompatible_skills:
        skills_to_install.extend(incompatible_skills)
    if not skills_to_install:
        console.print("  No compatible skills to install.")
        return

    incompatible_names: Set[str] = (
        {s.name for s in incompatible_skills} if install_incompatible_skills else set()
    )
    _write_skills_to_disk(skills_to_install, destinations, incompatible_names)

    console.print(
        Panel(
            "[bold green]Agent skills installed successfully.[/bold green]",
            border_style="green",
            expand=False,
        )
    )


# Version guard ========================================================================


def _partition_by_compatibility(
    skills: List[AgentSkillInfo],
) -> Tuple[List[AgentSkillInfo], List[AgentSkillInfo]]:
    """Split skills into compatible and incompatible lists."""
    compatible: List[AgentSkillInfo] = []
    incompatible: List[AgentSkillInfo] = []

    for skill in skills:
        if skill.is_compatible_with(rasa_version):
            compatible.append(skill)
        else:
            incompatible.append(skill)

    return compatible, incompatible


def _warn_incompatible_skills(
    incompatible: List[AgentSkillInfo],
    current_ver: str,
) -> None:
    """Print a warning for each incompatible skill."""
    console.print()
    for skill in incompatible:
        console.print(
            f"  [yellow]⚠[/yellow]  [bold]{skill.name}[/bold] was updated "
            f"for Rasa [bold]{skill.rasa_version}[/bold]. "
            f"Your version is [bold]{current_ver}[/bold]."
        )

    console.print()
    console.print(
        Panel(
            "[yellow]⚠[/yellow]  It is [bold]not recommended[/bold] to "
            "install these skills.\n"
            "The agent might use features that are incompatible with your "
            "current Rasa version.\n"
            "To use the new skills, please [bold]update Rasa[/bold].",
            border_style="yellow",
            expand=False,
        )
    )


def _confirm_install_incompatible() -> bool:
    """Ask the user whether to install incompatible skills anyway."""
    answer = questionary.confirm(
        "Install incompatible skills anyway?", default=False
    ).ask()
    if answer is None:
        sys.exit(1)
    return answer


# Network helpers ======================================================================


def _fetch_skill_names_from_repo() -> List[str]:
    """Fetch the list of available skill names from the agent skills repository.

    Returns:
        List of skill directory names (e.g. `["rasa-building-flows", ...]`).
        Returns an empty list if the fetch fails.
    """
    url = AGENT_SKILLS_API_URL.format(repo=AGENT_SKILLS_REPO)

    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
            entries = json.loads(response.read())
        return [e["name"] for e in entries if e["type"] == "dir"]
    except Exception as exc:
        console.print(f"  [red]✖[/red] Could not fetch skill list: {exc}")
        return []


def _download_skills_from_repo(skill_names: List[str]) -> List[AgentSkillInfo]:
    """Download and parse SKILL.md for each skill name.

    Returns:
        List of `AgentSkillInfo` objects for successfully downloaded skills. Skills that
        fail to download are skipped with a warning.
    """
    downloaded: List[AgentSkillInfo] = []
    for skill in skill_names:
        url = AGENT_SKILLS_RAW_URL.format(repo=AGENT_SKILLS_REPO, skill=skill)
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp:
                content = resp.read().decode("utf-8")
            downloaded.append(AgentSkillInfo.from_content(skill, content))
        except Exception as exc:
            console.print(f"  [red]✖[/red] {skill}: {exc}")
    return downloaded


# Local file system ====================================================================


def warn_if_agent_skills_exist(project_dir: Path, ides: List[str]) -> None:
    """Warn if agent skills directories exist when skipping installation.

    Args:
        project_dir: Absolute path to the Rasa project root.
        ides: List of IDE identifiers that were selected.
    """
    existing_dirs = [
        project_dir / base
        for ide, base in IDE_SKILLS_BASE.items()
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


def _find_existing_skills(
    destinations: Dict[str, Path],
) -> List[Path]:
    """Return a list of non-empty skill directories that already exist."""
    return [
        dest_dir
        for dest_dir in destinations.values()
        if dest_dir.is_dir() and any(dest_dir.iterdir())
    ]


def _confirm_overwrite_skills(existing: List[Path]) -> bool:
    """Ask the user whether to overwrite existing agent skills.

    Args:
        existing: Non-empty skill directories that already exist.

    Returns:
        `True` if the user confirms, `False` otherwise.
    """
    dir_list = "\n".join(f"  • {d}" for d in existing)
    console.print()
    console.print(
        Panel(
            "[yellow]⚠[/yellow]  [bold]Agent skills already installed[/bold]\n\n"
            "The following skill directories already exist:\n\n"
            f"{dir_list}\n\n"
            "Continuing will overwrite them with the latest versions.",
            border_style="yellow",
            expand=False,
        )
    )
    answer = questionary.confirm("Overwrite existing skills?", default=False).ask()
    if answer is None:
        sys.exit(1)
    return answer


def _write_skills_to_disk(
    skills: List[AgentSkillInfo],
    destinations: Dict[str, Path],
    incompatible_names: Optional[Set[str]] = None,
) -> None:
    """Write skill files to each IDE destination and print progress."""
    flagged = incompatible_names or set()
    max_name_len = max((len(s.name) for s in skills), default=0)

    for skill in skills:
        for dest_dir in destinations.values():
            dest = dest_dir / skill.name / "SKILL.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(skill.content, encoding="utf-8")

        ide_labels = ", ".join(IDE_DISPLAY_NAMES.get(ide, ide) for ide in destinations)
        padded_name = skill.name.ljust(max_name_len)
        if skill.name in flagged:
            indicator = "[green]✔[/green] [yellow]⚠[/yellow]"
            console.print(f"  {indicator}  {padded_name}  →  {ide_labels}")
            console.print(
                f"       [yellow]╰─ requires Rasa {skill.rasa_version}"
                ", incompatible with your version[/yellow]"
            )
        else:
            indicator = "[green]✔[/green]  "
            console.print(f"  {indicator}  {padded_name}  →  {ide_labels}")
