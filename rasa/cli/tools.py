import argparse
import logging
import os
import sys
from pathlib import Path
from typing import IO, Any, Dict, List, Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from rasa.cli import SubParsersAction
from rasa.cli.arguments.tools import (
    DOCS_MODE_OFFLINE,
    MCP_TOOLS_DEFAULT_HOST,
    MCP_TOOLS_DEFAULT_PORT,
    MCP_TOOLS_HTTP_HEALTH_URL_PATTERN,
    MCP_TOOLS_HTTP_URL_PATTERN,
    MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR,
    MCP_TOOLS_TRANSPORT_HTTP,
    MCP_TOOLS_TRANSPORT_STDIO,
    MCP_TOOLS_TRANSPORT_STREAMABLE_HTTP,
    SUPPORTED_IDES,
    set_tools_init_arguments,
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
    project_path: str = Field(default=".")
    docs_mode: Literal["offline", "online"] = Field(default=DOCS_MODE_OFFLINE)
    ide_integrations: List[str] = Field(default_factory=list)

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

    init_parser = tools_subparsers.add_parser(
        "init",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Interactive setup wizard for Rasa Tools.",
    )
    init_parser.set_defaults(func=init_tools)
    set_tools_init_arguments(init_parser)

    run_parser = tools_subparsers.add_parser(
        "run",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Start the Rasa MCP server.",
    )
    run_parser.set_defaults(func=run_tools)
    set_tools_run_arguments(run_parser)


# Entrypoints ==========================================================================


def _precheck(file: Optional[IO[str]] = None) -> None:
    """Validate the Rasa license and display the beta banner.

    Called at the start of every ``rasa tools`` subcommand. Exits with a
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


def init_tools(args: argparse.Namespace) -> None:
    """Entrypoint for `rasa tools init`.

    Args:
        args: The CLI arguments.
    """
    _precheck()

    from rasa.cli.tools_wizard import run_wizard

    run_wizard(args)


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

    _precheck(file=sys.stderr)
    _validate_config_exclusivity(args)

    cli_project_path = getattr(args, "project_path", None)

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


# Argument validation ==================================================================


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


# Run config resolution ================================================================


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
