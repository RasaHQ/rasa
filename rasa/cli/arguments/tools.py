import argparse

# Duplicated here to avoid importing rasa.builder.copilot.* at CLI startup.
# The copilot __init__.py eagerly loads heavy dependencies (agents SDK) which
# makes every CLI command (including "rasa --help") slow.
# Canonical values live in rasa.builder.copilot.mcp_server.constants and
# rasa.builder.copilot.constants — keep these in sync when updating.
MCP_TOOLS_DEFAULT_PORT = 7331
MCP_TOOLS_DEFAULT_HOST = "127.0.0.1"
MCP_TOOLS_TRANSPORT_STDIO = "stdio"
MCP_TOOLS_TRANSPORT_HTTP = "http"
MCP_TOOLS_TRANSPORT_STREAMABLE_HTTP = "streamable-http"
MCP_TOOLS_RASA_PROJECT_FOLDER_ENV_VAR = "RASA_PROJECT_FOLDER"
# keep in sync with MCP_HTTP_URL_PATTERN in rasa.builder.copilot.mcp_server.constants
MCP_TOOLS_HTTP_URL_PATTERN = "http://{host}:{port}/mcp"
MCP_TOOLS_HTTP_HEALTH_URL_PATTERN = "http://{host}:{port}/health"


def set_tools_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running the Rasa MCP tools server via `rasa tools run`."""
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        dest="project",
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
            f"Transport mode for the MCP server (default: "
            f"{MCP_TOOLS_TRANSPORT_STDIO}). '{MCP_TOOLS_TRANSPORT_STDIO}' is the "
            "standard for IDE integrations (requires stdout to contain ONLY JSON-RPC "
            "messages - all logging goes to stderr); "
            f"'{MCP_TOOLS_TRANSPORT_HTTP}' starts a network listener. "
            "Cannot be used together with --config."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        dest="port",
        help=f"Port for the MCP server in http mode "
        f"(default: {MCP_TOOLS_DEFAULT_PORT}). "
        "Cannot be used together with --config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        dest="config",
        help="Path to a run configuration file or directory containing "
        ".rasa/tools.yaml. When provided, all settings are loaded from this "
        "file. Cannot be combined with --mode, --port, or --project.",
    )
