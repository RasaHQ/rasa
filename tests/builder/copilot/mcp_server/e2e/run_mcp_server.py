#!/usr/bin/env python3
"""Script to run MCP server for testing.

This script is used by E2E tests to start the MCP server in a subprocess.
Usage: python run_test_server.py <project_folder> <host> <port>
"""

import sys
from pathlib import Path

from rasa.builder.copilot.mcp_server.constants import MCP_TRANSPORT_STREAMABLE_HTTP
from rasa.builder.copilot.mcp_server.server import run_server

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_test_server.py <project_folder> <host> <port>")
        sys.exit(1)

    project_folder = Path(sys.argv[1])
    host = sys.argv[2]
    port = int(sys.argv[3])

    # Run server (blocking)
    # run_server() properly initializes the global project folder variable
    try:
        run_server(
            host=host,
            port=port,
            transport=MCP_TRANSPORT_STREAMABLE_HTTP,
            project_folder=str(project_folder),
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
