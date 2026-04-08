import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from rasa.cli.tools.constants import (
    DEFAULT_RASA_SERVER_URL,
    IDE_DISPLAY_NAMES,
    MCP_TOOLS_TRANSPORT_HTTP,
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.utils import (
    RunConfig,
    _precheck,
    _resolve_project_dir,
)

STATUS_REACHABILITY_TIMEOUT_SECONDS = 3

console = Console()


def status_tools(args: argparse.Namespace) -> None:
    """Entrypoint for `rasa tools status`.

    Inspects the project directory for a saved configuration file and
    prints a summary of settings and basic runtime readiness.  Does not
    require the MCP server to be running.

    Args:
        args: The CLI arguments.
    """
    _precheck()

    project_dir = _resolve_project_dir(getattr(args, "project_path", None))
    config_path = project_dir / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME

    if config_path.is_file():
        config = RunConfig.load(config_path)
        _print_status(project_dir, config_path, config)
    else:
        _print_no_config(project_dir, config_path)


def _print_status(project_dir: Path, config_path: Path, config: RunConfig) -> None:
    """Print a rich status panel with all configured settings.

    Args:
        project_dir: Resolved absolute path to the project root.
        config_path: Path to the configuration file.
        config: The loaded run configuration.
    """
    lines = Text()

    lines.append("Project path   ", style="bold")
    lines.append(f"{project_dir}\n")

    lines.append("Config file    ", style="bold")
    lines.append(f"{config_path}\n")

    lines.append("Mode           ", style="bold")
    lines.append(f"{config.mode}\n")

    if config.mode == MCP_TOOLS_TRANSPORT_HTTP:
        lines.append("Port           ", style="bold")
        lines.append(f"{config.port}\n")

    lines.append("Docs mode      ", style="bold")
    lines.append(f"{config.docs_mode}\n")

    lines.append("Rasa server    ", style="bold")
    rasa_url = config.rasa_server_url or DEFAULT_RASA_SERVER_URL
    lines.append(f"{rasa_url}")
    if rasa_url == DEFAULT_RASA_SERVER_URL:
        lines.append(" (default)", style="dim")
    lines.append("\n")

    reachable = _check_reachability(rasa_url)
    lines.append("               ", style="bold")
    if reachable:
        lines.append("reachable", style="green")
    else:
        lines.append("unreachable", style="red")
    lines.append("\n")

    lines.append("IDEs           ", style="bold")
    if config.ide_integrations:
        names = ", ".join(
            IDE_DISPLAY_NAMES.get(ide, ide) for ide in config.ide_integrations
        )
        lines.append(f"{names}\n")
    else:
        lines.append("none configured\n", style="dim")

    console.print(
        Panel(
            lines,
            title="[bold cyan]Rasa Tools Status[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )


def _print_no_config(project_dir: Path, config_path: Path) -> None:
    """Print a status panel when no configuration file is found.

    Args:
        project_dir: Resolved absolute path to the project root.
        config_path: Expected path to the configuration file.
    """
    lines = Text()

    lines.append("Project path   ", style="bold")
    lines.append(f"{project_dir}\n")

    lines.append("Config file    ", style="bold")
    lines.append(f"{config_path} ")
    lines.append("NOT FOUND", style="red")
    lines.append("\n")

    console.print(
        Panel(
            lines,
            title="[bold cyan]Rasa Tools Status[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    console.print(
        "\n  No configuration found. "
        "Run [bold cyan]rasa tools init[/bold cyan] to set up your environment.\n"
    )


def _check_reachability(url: str) -> bool:
    """Perform a best-effort HTTP reachability check against URL.

    Args:
        url: The base URL to probe (e.g. `http://localhost:5005`).

    Returns:
        True if the server responds within the timeout, False otherwise.
    """
    try:
        with urlopen(url, timeout=STATUS_REACHABILITY_TIMEOUT_SECONDS):
            return True
    except HTTPError:
        # Server responded with a non-2xx status (e.g. 404, 500) — still
        # reachable, just not serving the expected content at this path.
        return True
    except (URLError, OSError, ValueError):
        # URLError: DNS failure, connection refused, timeout, etc.
        # OSError: lower-level network/socket errors.
        # ValueError: malformed URL that urllib cannot parse.
        return False
