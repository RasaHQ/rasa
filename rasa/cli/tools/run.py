"""``rasa tools run`` — start the Rasa MCP server."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from rasa.cli.tools.constants import (
    MCP_TOOLS_DEFAULT_HOST,
    MCP_TOOLS_HTTP_HEALTH_URL_PATTERN,
    MCP_TOOLS_HTTP_URL_PATTERN,
    MCP_TOOLS_TRANSPORT_HTTP,
    MCP_TOOLS_TRANSPORT_STDIO,
    MCP_TOOLS_TRANSPORT_STREAMABLE_HTTP,
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.utils import (
    RunConfig,
    _load_project_dotenv,
    _precheck,
    _redirect_logging_to_stderr,
    _resolve_project_dir,
)
from rasa.shared.exceptions import RasaException

# ── Entrypoint ────────────────────────────────────────────────────────────────


def run_tools(args: argparse.Namespace) -> None:
    """Entrypoint for `rasa tools run`.

    Args:
        args: The CLI arguments.
    """
    # Redirect logging to stderr before anything else — including the license
    # check — so that log messages emitted during startup never reach stdout.
    # In stdio mode stdout is reserved exclusively for JSON-RPC; in HTTP mode
    # this is harmless.
    _redirect_logging_to_stderr()

    # Load the project's .env file early so that RASA_LICENSE (and any other env vars)
    # are available before the license check.
    # override=False ensures explicit env vars still take precedence.
    cli_project_path = getattr(args, "project_path", None)
    _load_project_dotenv(cli_project_path)

    _precheck(file=sys.stderr)
    _validate_config_exclusivity(args)

    config, loaded_from_file = _resolve_tools_run_config(
        cli_mode=args.mode,
        cli_port=args.port,
        cli_config=args.config,
        cli_project_path=cli_project_path,
    )

    project_path = _resolve_project_dir(
        cli_project_path,
        config_project_path=config.project_path if loaded_from_file else None,
        validate_exists=True,
    )

    has_explicit_cli_args = (
        getattr(args, "mode", None) is not None
        or getattr(args, "port", None) is not None
    )
    if not loaded_from_file and not has_explicit_cli_args:
        raise RasaException(
            "No configuration found.\n\n"
            "Run `rasa tools init` to set up your environment,\n"
            "or pass explicit flags (e.g. `rasa tools run --mode stdio`)."
        )

    # Deferred import: rasa.builder.config reads env vars at module level, so it must be
    # imported after _load_project_dotenv() populates os.environ.
    from rasa.builder.config import TOOLS_PROXY_URL, apply_proxy_url

    apply_proxy_url(TOOLS_PROXY_URL)

    is_stdio = config.mode == MCP_TOOLS_TRANSPORT_STDIO

    # Rich console writes to stderr in stdio mode, stdout otherwise
    console = Console(file=sys.stderr if is_stdio else sys.stdout)

    if loaded_from_file:
        try:
            file_contents = Path(loaded_from_file).read_text()
        except Exception:
            file_contents = "<could not read file>"

        console.print(
            Panel(
                f"[bold]Config loaded from:[/bold]\n"
                f"{loaded_from_file}\n\n{file_contents}",
                border_style="dim",
                expand=False,
            )
        )

    mode = config.mode
    port = config.port

    from rasa.builder.copilot.mcp_server.server import run_server

    if mode == MCP_TOOLS_TRANSPORT_STDIO:
        console.print(
            Panel(
                "[bold cyan]MCP Server Starting[/bold cyan]\n"
                "Transport: [bold]stdio[/bold]\n"
                f"Project:   [bold]{project_path}[/bold]",
                border_style="cyan",
                expand=False,
            )
        )
        run_server(
            transport=MCP_TOOLS_TRANSPORT_STDIO,
            project_folder=str(project_path),
        )
    elif mode == MCP_TOOLS_TRANSPORT_HTTP:
        mcp_url = MCP_TOOLS_HTTP_URL_PATTERN.format(
            host=MCP_TOOLS_DEFAULT_HOST, port=port
        )
        health_url = MCP_TOOLS_HTTP_HEALTH_URL_PATTERN.format(
            host=MCP_TOOLS_DEFAULT_HOST, port=port
        )

        lines = Text()
        lines.append("Transport:    ", style="bold")
        lines.append("HTTP\n")
        lines.append("MCP endpoint: ", style="bold")
        lines.append(f"{mcp_url}\n")
        lines.append("Health check: ", style="bold")
        lines.append(f"{health_url}\n")
        lines.append("Project:      ", style="bold")
        lines.append(f"{project_path}")

        console.print(
            Panel(
                lines,
                title="[bold cyan]MCP Server Starting[/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )
        run_server(
            host=MCP_TOOLS_DEFAULT_HOST,
            port=port,
            transport=MCP_TOOLS_TRANSPORT_STREAMABLE_HTTP,
            project_folder=str(project_path),
        )
    else:
        raise ValueError(f"Unsupported transport mode: {mode!r}")


# ── Argument validation ──────────────────────────────────────────────────────


def _validate_config_exclusivity(args: argparse.Namespace) -> None:
    """Raise if --config is combined with --mode, --port, or --project-path."""
    if getattr(args, "config", None) is None:
        return

    conflicts = []
    for name in ("mode", "port", "project_path"):
        if getattr(args, name, None) is not None:
            conflicts.append(f"--{name.replace('_', '-')}")
    if conflicts:
        raise RasaException(
            f"--config cannot be combined with {', '.join(conflicts)}. "
            "When using a config file, all settings are loaded from it."
        )


# ── Run config resolution ────────────────────────────────────────────────────


def _resolve_tools_run_config(
    *,
    cli_mode: Optional[str],
    cli_port: Optional[int],
    cli_config: Optional[str],
    cli_project_path: Optional[str],
) -> Tuple[RunConfig, Optional[Path]]:
    """Resolve the run configuration for `rasa tools run`.

    Returns:
        A tuple of (config, config_path).  *config_path* is the path
        the config was loaded from, or ``None`` when built from CLI
        args / defaults.

    Priority:
    1. Explicit CLI args (--mode, --port) → use those + defaults.
    2. No CLI args, config file found → load it (defaults fill missing).
    3. No CLI args, no config file → pure defaults.
    """
    has_cli_args = cli_mode is not None or cli_port is not None

    if has_cli_args:
        kwargs: Dict[str, Any] = {}
        if cli_mode is not None:
            kwargs["mode"] = cli_mode
        if cli_port is not None:
            kwargs["port"] = cli_port
        return RunConfig(**kwargs), None

    # Locate the config file using CLI > env > cwd as a candidate directory.
    config_search_dir = _resolve_project_dir(cli_project_path)
    config_path = _resolve_config_path(cli_config, config_search_dir)

    if config_path:
        return RunConfig.load(config_path), config_path

    return RunConfig(), None


def _resolve_config_path(
    cli_config: Optional[str],
    project_dir: Path,
) -> Optional[Path]:
    """Return the concrete path to the config file.

    Priority: --config CLI arg > project directory

    Args:
        cli_config: Value of the --config CLI argument (may be None)
        project_dir: Path to the project directory

    Returns:
        Resolved absolute path to the config file, or None if the config file doesn't
        exist under the project directory.

    Raises:
        RasaException: If --config was given but the resolved path does not exist
    """
    if cli_config:
        config_path = Path(cli_config).resolve()
        if config_path.is_dir():
            if config_path.name == TOOLS_CONFIG_DIR:
                config_path = config_path / TOOLS_CONFIG_FILENAME
            else:
                config_path = config_path / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
        if not config_path.is_file():
            raise RasaException(
                f"The config file for `rasa tools run` was given but does not exist:\n"
                f"{config_path.resolve()}"
            )
        return config_path.resolve()

    # In case no --config was given, use the default config path under the project
    # directory if it exists.
    config_path = project_dir / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME

    if config_path.is_file():
        return config_path.resolve()

    return None
