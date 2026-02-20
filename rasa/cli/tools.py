import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import structlog

from rasa.cli import SubParsersAction
from rasa.cli.arguments.tools import set_tools_run_arguments
from rasa.shared.exceptions import RasaException

structlogger = structlog.get_logger()


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
    from rasa.builder.copilot.constants import RASA_PROJECT_FOLDER_ENV_VAR

    if cli_arg:
        path = Path(cli_arg)
    elif env_val := os.getenv(RASA_PROJECT_FOLDER_ENV_VAR):
        path = Path(env_val)
    else:
        path = Path.cwd()

    resolved = path.resolve()
    if not resolved.is_dir():
        structlogger.error(
            "cli.tools.invalid_project_path",
            event_info="Project path does not exist or is not a directory",
            path=str(resolved),
            cli_arg=cli_arg,
            env_var=os.getenv(RASA_PROJECT_FOLDER_ENV_VAR),
        )
        raise RasaException(
            f"Project path does not exist or is not a directory: {resolved}"
        )
    return resolved


def run_tools(args: argparse.Namespace) -> None:
    """Entrypoint for `rasa tools run`.

    Args:
        args: The CLI arguments.
    """
    from rasa.builder.copilot.mcp_server.constants import (
        MCP_DEFAULT_HOST,
        MCP_TRANSPORT_STDIO,
        MCP_TRANSPORT_STREAMABLE_HTTP,
    )

    project_path = _resolve_project_path(getattr(args, "project", None))
    mode: str = args.mode
    port: int = args.port

    # CRITICAL: For stdio mode, reconfigure logging to use stderr
    # MCP stdio protocol requires stdout to contain ONLY JSON-RPC messages
    if mode == "stdio":
        # Reconfigure all handlers to use stderr
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if hasattr(handler, "stream") and handler.stream == sys.stdout:
                handler.stream = sys.stderr

    from rasa.builder.copilot.mcp_server.server import run_server

    if mode == "stdio":
        run_server(
            transport=MCP_TRANSPORT_STDIO,
            project_folder=str(project_path),
        )
    else:
        run_server(
            host=MCP_DEFAULT_HOST,
            port=port,
            transport=MCP_TRANSPORT_STREAMABLE_HTTP,
            project_folder=str(project_path),
        )
