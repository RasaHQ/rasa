import argparse

# Duplicated here to avoid importing rasa.builder.copilot.* at CLI startup.
# The copilot __init__.py eagerly loads heavy dependencies (agents SDK) which
# makes "rasa --help" slow. Canonical value: MCP_TOOLS_DEFAULT_PORT = 7331
# (also defined in rasa.builder.copilot.mcp_server.constants).
MCP_TOOLS_DEFAULT_PORT = 7331


def set_tools_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running the Rasa MCP tools server via `rasa tools run`."""
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        dest="project",
        help="Path to the Rasa project folder. Falls back to the "
        "RASA_PROJECT_FOLDER environment variable, then the current directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stdio", "http"],
        default="stdio",
        dest="mode",
        help="Transport mode for the MCP server. "
        "'stdio' is the standard for IDE integrations (requires stdout to contain "
        "ONLY JSON-RPC messages - all logging goes to stderr); "
        "'http' starts a network listener.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=MCP_TOOLS_DEFAULT_PORT,
        dest="port",
        help="Port for the MCP server (only used in http mode).",
    )
