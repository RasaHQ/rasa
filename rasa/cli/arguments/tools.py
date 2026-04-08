import argparse

from rasa.cli.tools.constants import (
    DEFAULT_RASA_SERVER_URL,
    MCP_TOOLS_DEFAULT_PORT,
    MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR,
    MCP_TOOLS_TRANSPORT_HTTP,
    MCP_TOOLS_TRANSPORT_STDIO,
    SUPPORTED_IDES,
)


def set_tools_docs_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for `rasa tools init docs`."""
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        dest="project_path",
        help=(
            "Path to the Rasa project folder. "
            "Defaults to the current directory when not specified."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        dest="yes",
        help="Skip confirmation prompts and overwrite existing docs.",
    )


def set_tools_skills_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for `rasa tools init skills`."""
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        dest="yes",
        help="Skip confirmation prompts and accept defaults.",
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        dest="project_path",
        help=(
            "Path to the Rasa project folder. "
            "Defaults to the current directory when not specified."
        ),
    )
    parser.add_argument(
        "--ides",
        type=str,
        default=None,
        dest="ides",
        help=(
            "Comma-separated list of IDEs to install skills for: "
            f"{', '.join(SUPPORTED_IDES)}. "
            "If omitted, reads from the saved configuration."
        ),
    )


def set_tools_init_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for the interactive setup wizard via `rasa tools init`.

    The wizard collects all configuration values interactively, or applies
    defaults in `--yes` mode.
    """
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        dest="yes",
        help="Accept all defaults and skip interactive prompts.",
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        dest="project_path",
        help=(
            "Path to the Rasa project folder. "
            "Defaults to the current directory when not specified."
        ),
    )


def set_tools_status_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for `rasa tools status`."""
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        dest="project_path",
        help=(
            "Path to the Rasa project folder. "
            "Defaults to the current directory when not specified."
        ),
    )


def set_tools_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running the Rasa MCP tools server via `rasa tools run`."""
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        dest="project_path",
        help=(
            "Path to the Rasa project folder. Falls back to the "
            f"{MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR} environment variable, then the "
            "current directory. Cannot be used together with --config."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[MCP_TOOLS_TRANSPORT_STDIO, MCP_TOOLS_TRANSPORT_HTTP],
        default=None,
        dest="mode",
        help=(
            "Transport mode for the MCP server. "
            f"Defaults to '{MCP_TOOLS_TRANSPORT_STDIO}' when not specified. "
            f"'{MCP_TOOLS_TRANSPORT_STDIO}' is the standard for IDE integrations "
            "(requires stdout to contain ONLY JSON-RPC messages — all logging goes "
            f"to stderr); '{MCP_TOOLS_TRANSPORT_HTTP}' starts a network listener. "
            "Cannot be used together with --config."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        dest="port",
        help=(
            "Port for the MCP server in http mode. "
            f"Defaults to {MCP_TOOLS_DEFAULT_PORT} when not specified. "
            "Cannot be used together with --config."
        ),
    )
    parser.add_argument(
        "--rasa-server-url",
        type=str,
        default=None,
        dest="rasa_server_url",
        help=(
            "Base URL of the running Rasa server "
            f"(e.g. {DEFAULT_RASA_SERVER_URL}). "
            "Tells the MCP server where to reach the Rasa assistant. "
            f"Defaults to {DEFAULT_RASA_SERVER_URL}. "
            "Cannot be used together with --config."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        dest="config",
        help="Path to a run configuration file or directory containing "
        ".rasa/tools.yaml. When provided, all settings are loaded from this "
        "file. Cannot be combined with --mode, --port, or --project-path.",
    )
