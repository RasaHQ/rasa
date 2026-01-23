#!/usr/bin/env python3
"""Script to run MCP server for testing.

This script is used by E2E tests to start the MCP server in a subprocess.
Usage: python run_test_server.py <project_folder> <host> <port>
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_test_server.py <project_folder> <host> <port>")
        sys.exit(1)

    project_folder = Path(sys.argv[1])
    host = sys.argv[2]
    port = int(sys.argv[3])

    # Set environment variable
    os.environ["RASA_PROJECT_FOLDER"] = str(project_folder)

    # Import and configure server
    from rasa.builder.copilot.mcp_server.server import mcp

    mcp.settings.host = host
    mcp.settings.port = port

    # Run server (blocking)
    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
