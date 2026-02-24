import argparse
import logging
import os
import sys
from pathlib import Path
from typing import IO, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from rasa.cli import SubParsersAction
from rasa.cli.arguments.tools import (
    MCP_TOOLS_DEFAULT_HOST,
    MCP_TOOLS_DEFAULT_PORT,
    MCP_TOOLS_HTTP_HEALTH_URL_PATTERN,
    MCP_TOOLS_HTTP_URL_PATTERN,
    MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR,
    MCP_TOOLS_TRANSPORT_HTTP,
    MCP_TOOLS_TRANSPORT_STDIO,
    MCP_TOOLS_TRANSPORT_STREAMABLE_HTTP,
    set_tools_run_arguments,
)
from rasa.shared.exceptions import RasaException

TOOLS_CONFIG_DIR = ".rasa"
TOOLS_CONFIG_FILENAME = "tools.yaml"


class RunConfig(BaseModel):
    """Persisted run parameters for ``rasa tools run``.

    Add new fields with sensible defaults to extend.  Save, load, and display
    adapt automatically — no other code changes required.
    """

    model_config = {"extra": "ignore"}

    mode: Literal["stdio", "http"] = Field(default=MCP_TOOLS_TRANSPORT_STDIO)
    port: int = Field(default=MCP_TOOLS_DEFAULT_PORT)

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


# CLI registration =====================================================================


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all tools parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    tools_parser = subparsers.add_parser(
        "tools",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Commands for Rasa developer tools.",
    )
    tools_parser.set_defaults(func=lambda _: tools_parser.print_help(None))

    tools_subparsers = tools_parser.add_subparsers()

    run_parser = tools_subparsers.add_parser(
        "run",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Start the Rasa MCP server.",
    )
    run_parser.set_defaults(func=run_tools)
    set_tools_run_arguments(run_parser)


# Entrypoint ===========================================================================


def run_tools(args: argparse.Namespace) -> None:
    """Entrypoint for `rasa tools run`.

    Args:
        args: The CLI arguments.
    """
    _validate_config_exclusivity(args)

    project_path = _resolve_project_path(getattr(args, "project", None))

    config, loaded_from_file = _resolve_tools_run_config(
        cli_mode=args.mode,
        cli_port=args.port,
        cli_config=args.config,
        project_dir=project_path,
    )

    is_stdio = config.mode == MCP_TOOLS_TRANSPORT_STDIO

    # In stdio mode, stdout/stdin carry JSON-RPC messages — redirect logging
    # to stderr immediately, before anything else can write to stdout.
    if is_stdio:
        _redirect_logging_to_stderr()

    output = sys.stderr if is_stdio else sys.stdout

    if loaded_from_file:
        try:
            file_contents = Path(loaded_from_file).read_text()
        except Exception:
            file_contents = "<could not read file>"
        print(
            f"--------------------------------\n"
            f"Loaded run config from\n{loaded_from_file}:\n"
            f"{file_contents}"
            f"--------------------------------",
            file=output,
        )
    else:
        _auto_save_config(config, project_path, output)

    mode = config.mode
    port = config.port

    from rasa.builder.copilot.mcp_server.server import run_server

    if mode == MCP_TOOLS_TRANSPORT_STDIO:
        print(
            "--------------------------------\n"
            "MCP server is running on:\n"
            "stdio\n"
            "--------------------------------",
            file=output,
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
        print(
            f"--------------------------------\n"
            f"MCP server is running on:\n"
            f"{mcp_url}\n"
            f"\n"
            f"Health check:\n"
            f"{health_url}\n"
            f"--------------------------------",
            file=output,
        )
        run_server(
            host=MCP_TOOLS_DEFAULT_HOST,
            port=port,
            transport=MCP_TOOLS_TRANSPORT_STREAMABLE_HTTP,
            project_folder=str(project_path),
        )
    else:
        raise ValueError(f"Unsupported transport mode: {mode!r}")


# Argument validation ==================================================================


def _validate_config_exclusivity(args: argparse.Namespace) -> None:
    """Raise if --config is combined with --mode, --port, or --project."""
    if getattr(args, "config", None) is None:
        return

    conflicts = [
        f"--{name}"
        for name in ("mode", "port", "project")
        if getattr(args, name, None) is not None
    ]
    if conflicts:
        raise RasaException(
            f"--config cannot be combined with {', '.join(conflicts)}. "
            "When using a config file, all settings are loaded from it."
        )


# Run config resolution ================================================================


def _resolve_tools_run_config(
    *,
    cli_mode: Optional[str],
    cli_port: Optional[int],
    cli_config: Optional[str],
    project_dir: Path,
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

    config_path = _resolve_config_path(cli_config, project_dir)

    if config_path:
        return RunConfig.load(config_path), config_path

    return RunConfig(), None


# Path resolutions helpers =============================================================


def _resolve_project_path(cli_arg: Optional[str]) -> Path:
    """Resolve the project path using the priority cascade.

    Priority: --project CLI arg > RASA_PROJECT_FOLDER env var > cwd

    Args:
        cli_arg: Value of the --project CLI argument (may be None)

    Returns:
        Resolved absolute path to the project folder

    Raises:
        RasaException: If the resolved path does not exist
    """
    if cli_arg:
        path = Path(cli_arg)
    elif env_val := os.getenv(MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR):
        path = Path(env_val)
    else:
        path = Path.cwd()

    resolved = path.resolve()
    if not resolved.is_dir():
        raise RasaException(
            f"Project path does not exist or is not a directory: {resolved}"
        )
    return resolved


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


# Config persistence helpers ===========================================================


def _auto_save_config(
    config: RunConfig, project_dir: Path, output: IO[str]
) -> Optional[Path]:
    """Persist *config* to the default location.

    Saving is best-effort: any exception (OSError, serialization, etc.) is
    caught so that ``rasa tools run`` still starts the server.

    Returns:
        The path the config was saved to, or ``None`` on failure.
    """
    dest = project_dir / TOOLS_CONFIG_DIR / TOOLS_CONFIG_FILENAME
    try:
        config.save(dest)
        print(
            f"--------------------------------\n"
            f"Saved run config to\n{dest}:\n"
            f"{dest.read_text()}"
            f"\n"
            f"To edit settings, modify the config file directly "
            f"or pass CLI flags (--mode, --port).\n"
            f"--------------------------------",
            file=output,
        )
    except Exception:
        print(f"Could not auto-save run config: {dest}", file=output)
        return None

    return dest


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
