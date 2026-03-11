"""``rasa tools init docs`` — fetch offline Rasa documentation files."""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel

from rasa.cli.tools.constants import (
    DEFAULT_LLMS_TXT_BASE_URL,
    DOCS_MODE_ONLINE,
    HTTP_TIMEOUT,
    LLMS_TXT_BASE_URL_ENV_VAR,
    LLMS_TXT_FILES,
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.utils import _precheck, _resolve_project_dir

console = Console()


# Entrypoint ===========================================================================


def docs_tools(args: argparse.Namespace) -> None:
    """Entrypoint for `rasa tools init docs`.

    Args:
        args: The CLI arguments.
    """
    _precheck()

    project_dir = _resolve_project_dir(
        getattr(args, "project_path", None), validate_exists=True
    )
    non_interactive = getattr(args, "yes", False)

    fetch_offline_docs(project_dir, non_interactive=non_interactive)
    warn_if_online_docs_configured(project_dir)


# Docs fetching & management ===========================================================


def resolve_llms_txt_base_url() -> str:
    """Return the base URL for fetching llms.txt documentation files.

    Reads from the ``RASA_LLMS_TXT_BASE_URL`` environment variable if set,
    otherwise falls back to the default Rasa documentation URL.

    Returns:
        Base URL string (without trailing slash).
    """
    url = os.getenv(LLMS_TXT_BASE_URL_ENV_VAR, DEFAULT_LLMS_TXT_BASE_URL)
    return url.rstrip("/")


def fetch_offline_docs(project_dir: Path, non_interactive: bool = False) -> None:
    """Download the offline docs bundle (llms.txt files) into the config directory.

    Fetches each file listed in `LLMS_TXT_FILES` from the resolved base URL and saves
    them under `<project_dir>/.rasa/`.  Failures are reported but do not abort the
    process.

    Args:
        project_dir: Absolute path to the Rasa project root.
        non_interactive: When `True`, skip confirmation prompts and overwrite
            existing docs silently.
    """
    base_url = resolve_llms_txt_base_url()
    dest_dir = project_dir / TOOLS_CONFIG_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = [f for f in LLMS_TXT_FILES if (dest_dir / f).exists()]
    if existing and not non_interactive:
        if not _confirm_overwrite_docs(existing, dest_dir):
            console.print("  Skipping docs download.")
            return

    console.print()
    for filename in LLMS_TXT_FILES:
        url = f"{base_url}/{filename}"
        dest = dest_dir / filename
        console.print(f"  Fetching {filename}…", end=" ")
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
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


def _confirm_overwrite_docs(existing: list, dest_dir: Path) -> bool:
    """Ask the user whether to overwrite existing offline docs.

    Args:
        existing: Filenames that already exist on disk.
        dest_dir: Directory where the files live.

    Returns:
        `True` if the user confirms, `False` otherwise.
    """
    file_list = "\n".join(f"  • {dest_dir / f}" for f in existing)
    console.print()
    console.print(
        Panel(
            "[yellow]⚠[/yellow]  [bold]Offline docs already downloaded[/bold]\n\n"
            "The following files already exist:\n\n"
            f"{file_list}\n\n"
            "Continuing will overwrite them with the latest versions.",
            border_style="yellow",
            expand=False,
        )
    )
    answer = questionary.confirm("Overwrite existing docs?", default=False).ask()
    if answer is None:
        sys.exit(1)
    return answer


def warn_if_offline_docs_exist(project_dir: Path) -> None:
    """Warn the user if offline docs files exist when switching to online mode.

    Checks for the presence of any llms.txt files in the config directory and
    displays a warning panel if found. The files are not deleted automatically.

    Args:
        project_dir: Absolute path to the Rasa project root.
    """
    dest_dir = project_dir / TOOLS_CONFIG_DIR
    existing_files = [
        filename for filename in LLMS_TXT_FILES if (dest_dir / filename).exists()
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


def warn_if_online_docs_configured(project_dir: Path) -> None:
    """Warn the user if the project is configured for online documentation.

    Called after downloading offline docs so the user knows the files may
    conflict with their online documentation setting.

    Args:
        project_dir: Absolute path to the Rasa project root.
    """
    if not _is_online_docs_configured(project_dir):
        return

    console.print()
    console.print(
        Panel(
            "[yellow]⚠[/yellow]  [bold]Online documentation mode configured[/bold]\n\n"
            "Your project is currently set to use online documentation.\n"
            "The offline docs files just downloaded may be read by the agent instead "
            "of querying the online API.\n\n"
            "You can either:\n"
            "• Disable the MCP tool for searching docs, or\n"
            "• Delete these files if you prefer to use online docs.",
            border_style="yellow",
            expand=False,
        )
    )


def _is_online_docs_configured(project_dir: Path) -> bool:
    """Check whether the saved project config uses online documentation mode.

    Args:
        project_dir: Absolute path to the Rasa project root.

    Returns:
        `True` if a saved config exists and its `docs_mode` is `"online"`, `False`
        otherwise (including when no config file is present).
    """
    config_path = project_dir / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
    if not config_path.is_file():
        return False

    try:
        from rasa.cli.tools.models import RunConfig

        config = RunConfig.load(config_path)
        return config.docs_mode == DOCS_MODE_ONLINE
    except Exception:
        return False
