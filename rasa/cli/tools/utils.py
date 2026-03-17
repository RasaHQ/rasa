"""Shared utilities for the ``rasa tools`` CLI command family."""

import logging
import os
import sys
from pathlib import Path
from typing import IO, List, Optional

from rich.console import Console
from rich.panel import Panel

from rasa.cli.tools.constants import (
    MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR,
    TOOLS_CONFIG_DIR,
    TOOLS_CONFIG_FILENAME,
)
from rasa.cli.tools.models import RunConfig
from rasa.shared.exceptions import RasaException

# I/O helpers =======================================================================


def restore_blocking_io() -> None:
    """Restore stdout and stderr to blocking mode.

    `prompt_toolkit` (used by `questionary`) may leave file descriptors in
    non-blocking mode after an interactive prompt.  Subsequent writes in
    particular via `rich.Console.print()` — then raise `BlockingIOError`
    on macOS (errno 35 / EAGAIN).  Calling this after each `.ask()` prevents
    that.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            os.set_blocking(stream.fileno(), True)
        except (AttributeError, OSError):
            pass


# Pre-check ========================================================================


def _precheck(file: Optional[IO[str]] = None) -> None:
    """Validate the Rasa license and display the beta banner.

    Called at the start of every `rasa tools` subcommand. Exits with a
    clear message if no valid license is present in the environment.

    Args:
        file: Output stream for the banner. Defaults to stdout.
            Pass `sys.stderr` when stdout is reserved (e.g. stdio mode).
    """
    from rasa.utils.licensing import validate_license_from_env

    validate_license_from_env()

    banner = (
        "[yellow bold]⚠ Rasa Tools is currently in beta.[/yellow bold]\n"
        "Help us improve it by sending feedback or issues to "
        "[bold]swift@rasa.com[/bold]."
    )
    console = Console(file=file)
    console.print(Panel(banner, border_style="yellow", expand=False))
    console.print()


# Project directory resolution =======================================================


def _resolve_project_dir(
    cli_arg: Optional[str] = None,
    config_project_path: Optional[str] = None,
    validate_exists: bool = False,
) -> Path:
    """Resolve the project directory using the standard priority cascade.

    Priority:
        --project-path CLI arg > RASA_PROJECT_FOLDER env var >
        config file project_path > CWD

    Args:
        cli_arg: Value of the --project-path CLI argument (may be None).
        config_project_path: Value of `project_path` from a loaded config
            file. Only consulted when neither CLI arg nor env var is set.
        validate_exists: When True, raise RasaException if the resolved
            path is not an existing directory.

    Returns:
        Resolved absolute path to the project directory.

    Raises:
        RasaException: If validate_exists is True and the path doesn't exist.
    """
    if cli_arg:
        path = Path(cli_arg).resolve()
    elif env_val := os.getenv(MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR):
        path = Path(env_val).resolve()
    elif config_project_path:
        path = Path(config_project_path).resolve()
    else:
        path = Path.cwd().resolve()

    if validate_exists and not path.is_dir():
        raise RasaException(
            f"Project path does not exist or is not a directory: {path}"
        )
    return path


def _resolve_ides(
    cli_ides: Optional[str],
    project_dir: Path,
) -> List[str]:
    """Resolve the list of IDEs from CLI arg or saved config.

    Priority: CLI arg > saved config file

    Args:
        cli_ides: Comma-separated IDE string from the CLI (may be None).
        project_dir: Absolute path to the Rasa project root.

    Returns:
        List of IDE identifiers.
    """
    if cli_ides:
        return [tok.strip().lower() for tok in cli_ides.split(",") if tok.strip()]

    config_path = project_dir / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
    if config_path.is_file():
        config = RunConfig.load(config_path)
        return config.ide_integrations

    return []


# .env loading =========================================================================


def _load_project_dotenv(cli_project_path: Optional[str] = None) -> None:
    """Load the project's ``.env`` file into ``os.environ``.

    This makes credentials like ``RASA_LICENSE`` available to the
    license check even when the IDE-spawned process has no shell
    environment.  Existing env vars are never overwritten.

    Args:
        cli_project_path: Value of the ``--project-path`` CLI argument
            (may be ``None``).
    """
    from dotenv import load_dotenv

    project_dir = _resolve_project_dir(cli_project_path)
    env_file = project_dir / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


# Logging helpers ======================================================================


def _redirect_logging_to_stderr() -> None:
    """Redirect all logging output from stdout to stderr.

    Must be called before any output is written in stdio mode so that
    log lines never corrupt the JSON-RPC stream on stdout.

    Covers two output paths:
    1. stdlib logging handlers (used after configure_structlog sets up
       LoggerFactory + basicConfig(stream=stdout))
    2. structlog's default PrintLoggerFactory (used before
       configure_structlog runs, e.g. during early startup or in tests)
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if hasattr(handler, "stream") and handler.stream == sys.stdout:
            handler.stream = sys.stderr

    import structlog

    cfg = structlog.get_config()
    cfg["logger_factory"] = structlog.PrintLoggerFactory(file=sys.stderr)
    # filter_by_level requires a stdlib logging.Logger (.disabled attribute);
    # PrintLogger doesn't have it, so drop that processor when using PrintLoggerFactory.
    cfg["processors"] = [
        p
        for p in cfg.get("processors", [])
        if p is not structlog.stdlib.filter_by_level
    ]
    structlog.configure(**cfg)
