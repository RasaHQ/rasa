"""Pytest fixtures for MCP server end-to-end tests."""

import os
import shutil
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import AsyncGenerator, Callable, Generator, Optional

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH
from tests.builder.copilot.mcp_server.e2e.constants import (
    REQUIRED_ENV_VARS,
    RUN_RASA_SERVER_SCRIPT_PATH,
    SIMPLE_ASSISTANT_PATH,
)
from tests.builder.copilot.mcp_server.e2e.mcp_server_manager import MCPServerManager
from tests.builder.copilot.mcp_server.e2e.utils import (
    find_free_port,
    get_subprocess_error_message,
    prepare_test_env,
    start_logging_threads,
    stop_process,
    wait_for_server,
)

# Test server configuration
SERVER_HOST = "127.0.0.1"

# Debug mode: set DEBUG_SERVER_OUTPUT=1 to see all server output
DEBUG_SERVER_OUTPUT = os.environ.get("DEBUG_SERVER_OUTPUT", "").lower() in (
    "1",
    "true",
    "yes",
)


@pytest.fixture(scope="session", autouse=True)
def validate_environment():
    """Validate required environment variables before running any tests.

    This fixture runs automatically before any tests and will fail early
    with a clear error message if required environment variables are missing.
    """
    missing_vars = [
        var for var in REQUIRED_ENV_VARS if var not in os.environ or not os.environ[var]
    ]

    if missing_vars:
        missing_list = ", ".join(missing_vars)
        error_msg = (
            f"Missing required environment variables: {missing_list}\n"
            "Please set these environment variables before running the tests."
        )
        raise ValueError(error_msg)


@pytest.fixture
def test_project_folder(tmp_path: Path) -> Path:
    """Create minimal valid Rasa project by copying from simple_assistant."""
    project = tmp_path / "test_project"

    # Copy the entire simple_assistant folder structure
    shutil.copytree(SIMPLE_ASSISTANT_PATH, project)

    return project


@pytest.fixture
def mcp_server(
    test_project_folder: Path, request: pytest.FixtureRequest
) -> Generator[MCPServerManager, None, None]:
    """Start MCP server for testing.

    If rasa_server fixture is also requested, it will pass the Rasa server
    host and port via environment variables so the MCP server can connect to it.
    """
    # Use dynamic port to avoid conflicts
    port = find_free_port()

    # Check if rasa_server fixture is also requested
    rasa_server_info = None
    if "rasa_server" in request.fixturenames:
        rasa_server_info = request.getfixturevalue("rasa_server")

    manager = MCPServerManager(test_project_folder, SERVER_HOST, port)

    # If rasa_server is available, pass its host/port via env vars
    if rasa_server_info:
        server_port, _ = rasa_server_info
        # Store in manager so it can be passed to subprocess
        manager._rasa_server_port = server_port
        manager._rasa_server_host = SERVER_HOST

    try:
        manager.start()
        yield manager
    finally:
        manager.stop()


@pytest.fixture
async def mcp_client(
    mcp_server: MCPServerManager,
) -> AsyncGenerator[ClientSession, None]:
    """Create an MCP client session connected to the test server."""
    # Verify server is still running before connecting
    if mcp_server.process and mcp_server.process.poll() is not None:
        raise RuntimeError("MCP server process has terminated before client connection")

    stack = AsyncExitStack()
    try:
        # Enter the streamablehttp_client context
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamablehttp_client(mcp_server.url)
        )

        # Enter the ClientSession context
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await session.initialize()
        yield session

    except Exception as e:
        # If connection fails, check if server process is still running
        if mcp_server.process and mcp_server.process.poll() is not None:
            # Server has crashed, try to get error output
            error_msg = get_subprocess_error_message(
                mcp_server.process,
                f"MCP client connection failed: {e}",
                timeout=0.5,
            )
            raise RuntimeError(error_msg) from e
        raise
    finally:
        # Clean up the exit stack, suppressing anyio task group cleanup errors
        try:
            await stack.aclose()
        except RuntimeError as e:
            # Suppress "Attempted to exit cancel scope in a different task" errors
            # These are known issues with anyio and pytest-asyncio teardown
            if "cancel scope" not in str(e).lower():
                raise


@pytest.fixture
def mcp_server_port() -> int:
    """Get a free port for MCP server (for parametrized tests)."""
    return find_free_port()


@pytest.fixture
def rasa_server(
    test_project_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[int, Callable[[str], None]], None, None]:
    """Start a Rasa server with a trained model using rasa.api.run.

    This fixture sets up configuration and patches the builder config.
    It yields a tuple of (port, start_function) where start_function
    can be called with a model path to start the server.

    Yields:
        Tuple of (server_port, start_with_model_function)
    """
    # Find a free port for the server
    server_port = find_free_port()

    # Get endpoints path from project folder
    endpoints_path = test_project_folder / DEFAULT_ENDPOINTS_PATH

    # Patch the builder config so talk_to_assistant uses the correct port
    from rasa.builder import config as builder_config

    monkeypatch.setattr(builder_config, "BUILDER_SERVER_HOST", SERVER_HOST)
    monkeypatch.setattr(builder_config, "BUILDER_SERVER_PORT", server_port)

    # Store process reference for cleanup
    server_process: Optional[subprocess.Popen] = None

    # Function to start the server with a given model path
    def start_with_model(trained_model_path: str) -> None:
        """Start the Rasa server with the given model path in a subprocess."""
        nonlocal server_process

        # Stop any existing server process
        stop_process(server_process)

        # Start server as subprocess
        env = prepare_test_env()

        server_process = subprocess.Popen(
            [
                sys.executable,
                str(RUN_RASA_SERVER_SCRIPT_PATH),
                trained_model_path,
                str(server_port),
                str(endpoints_path),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered
        )

        # Start logging threads if debug mode is enabled
        start_logging_threads(server_process, "Rasa Server")

        # Wait for server to be ready
        if not wait_for_server(SERVER_HOST, server_port, timeout=30.0):
            error_msg = get_subprocess_error_message(
                server_process, f"Rasa server failed to start on port {server_port}"
            )
            stop_process(server_process, timeout=1.0)
            raise RuntimeError(error_msg)

        # Give server a moment to fully initialize after port is open
        time.sleep(0.5)

        # Verify process is still running
        from tests.builder.copilot.mcp_server.e2e.utils import verify_process_running

        verify_process_running(
            server_process,
            (
                f"Rasa server process terminated immediately "
                f"after starting on port {server_port}"
            ),
        )

    # Yield tuple: (port, start_function)
    yield (server_port, start_with_model)

    # Cleanup: stop server process if it's still running
    stop_process(server_process)
