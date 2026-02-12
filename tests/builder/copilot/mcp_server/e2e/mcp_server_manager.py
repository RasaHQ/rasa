"""MCP Server Manager for end-to-end testing."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rasa.builder.copilot.constants import RASA_PROJECT_FOLDER_ENV_VAR
from tests.builder.copilot.mcp_server.e2e.constants import (
    RUN_MCP_SERVER_SCRIPT_PATH,
)
from tests.builder.copilot.mcp_server.e2e.utils import (
    get_subprocess_error_message,
    prepare_test_env,
    start_logging_threads,
    stop_process,
    verify_process_running,
    wait_for_server,
)


class MCPServerManager:
    """Manages MCP server lifecycle for testing."""

    def __init__(self, project_folder: Path, host: str, port: int):
        self.project_folder = project_folder
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self._server_url = f"http://{host}:{port}/mcp"
        self._rasa_server_host: Optional[str] = None
        self._rasa_server_port: Optional[int] = None

    def start(self) -> None:
        """Start the MCP server in a subprocess."""
        # Prepare environment variables
        additional_env = {RASA_PROJECT_FOLDER_ENV_VAR: str(self.project_folder)}
        # Pass Rasa server host/port if available (for talk_to_assistant tool)
        if self._rasa_server_host is not None and self._rasa_server_port is not None:
            additional_env["SERVER_HOST"] = self._rasa_server_host
            additional_env["SERVER_PORT"] = str(self._rasa_server_port)
        env = prepare_test_env(additional_env)

        self.process = subprocess.Popen(
            [
                sys.executable,
                str(RUN_MCP_SERVER_SCRIPT_PATH),
                str(self.project_folder),
                self.host,
                str(self.port),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered
        )

        # Start logging threads if debug mode is enabled
        start_logging_threads(self.process, "MCP Server")

        if not wait_for_server(self.host, self.port):
            error_msg = get_subprocess_error_message(
                self.process, f"MCP server failed to start on {self._server_url}"
            )
            self.stop()
            raise RuntimeError(error_msg)

        # Give server a moment to fully initialize after port is open
        time.sleep(0.5)

        # Verify process is still running
        verify_process_running(
            self.process,
            (
                f"MCP server process terminated immediately "
                f"after starting on {self._server_url}"
            ),
            cleanup_func=self.stop,
        )

    def stop(self) -> None:
        """Stop the MCP server."""
        stop_process(self.process)
        self.process = None

    @property
    def url(self) -> str:
        """Get the server URL."""
        return self._server_url
