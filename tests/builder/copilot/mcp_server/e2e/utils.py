"""Utility functions for MCP server end-to-end testing."""

import os
import socket
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import structlog

logger = structlog.get_logger()

# Debug mode: set DEBUG_SERVER_OUTPUT=1 to see all server output
# Note: Use pytest -s flag to see debug output (pytest captures stdout by default)
DEBUG_SERVER_OUTPUT = os.environ.get("DEBUG_SERVER_OUTPUT", "").lower() in (
    "1",
    "true",
    "yes",
)


def get_subprocess_error_message(
    process: subprocess.Popen, base_message: str, timeout: float = 1.0
) -> str:
    """Extract error message from subprocess output.

    Args:
        process: The subprocess to get output from
        base_message: Base error message
        timeout: Timeout for process.communicate()

    Returns:
        Error message with subprocess output appended if available
    """
    error_msg = base_message
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if stderr:
            stderr_text = stderr.decode("utf-8", errors="ignore")
            error_msg += f"\nServer stderr: {stderr_text}"
        if stdout:
            stdout_text = stdout.decode("utf-8", errors="ignore")
            error_msg += f"\nServer stdout: {stdout_text}"
    except subprocess.TimeoutExpired:
        logger.warning(
            "subprocess.communicate.timeout",
            event_info="Timeout while trying to get subprocess error message",
            timeout=timeout,
        )
    return error_msg


def stop_process(process: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
    """Stop a subprocess gracefully, killing if necessary.

    Args:
        process: The subprocess to stop
        timeout: Timeout for graceful termination
    """
    if process:
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        except Exception:
            pass


def prepare_test_env(
    additional_env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Prepare environment variables for test subprocesses.

    Args:
        additional_env: Additional environment variables to set

    Returns:
        Environment dictionary with required test variables
    """
    from tests.builder.copilot.mcp_server.e2e.constants import REQUIRED_ENV_VARS

    env = os.environ.copy()
    # Pass through required environment variables for validation and training
    for key in REQUIRED_ENV_VARS:
        if key in os.environ:
            env[key] = os.environ[key]
    # Add any additional environment variables
    if additional_env:
        env.update(additional_env)
    return env


def verify_process_running(
    process: Optional[subprocess.Popen],
    error_base_message: str,
    cleanup_func: Optional[Callable[[], None]] = None,
) -> None:
    """Verify that a process is still running after startup.

    Args:
        process: The process to verify
        error_base_message: Base message for error if process terminated
        cleanup_func: Optional cleanup function to call if process terminated

    Raises:
        RuntimeError: If process has terminated
    """
    if process and process.poll() is not None:
        error_msg = get_subprocess_error_message(process, error_base_message)
        if cleanup_func:
            cleanup_func()
        raise RuntimeError(error_msg)


def find_free_port() -> int:
    """Find a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except Exception as e:
            logger.debug(
                "wait_for_server.connection_error",
                event_info=f"Error while waiting for server: {e}",
                host=host,
                port=port,
                error=str(e),
            )
        time.sleep(0.1)
    return False


def log_stream(stream, prefix: str, server_name: str, debug_mode: bool = False) -> None:
    """Consume output from a stream in a background thread.

    Always consumes the stream to prevent subprocess blocking, but only logs
    when debug_mode is enabled.

    Args:
        stream: The stream to read from (stdout or stderr)
        prefix: Prefix for log lines (e.g., "[MCP] " or "[Rasa] ")
        server_name: Name of the server for logging
        debug_mode: If True, log output; if False, silently consume

    Note: Use pytest -s flag to see this output (pytest captures stdout by default)
    """
    try:
        for line in iter(stream.readline, b""):
            if line:
                if debug_mode:
                    line_str = line.decode("utf-8", errors="replace").rstrip()
                    # Use stderr so output is visible even if pytest captures stdout
                    # (though -s flag is still recommended for full visibility)
                    print(
                        f"{prefix}{server_name}: {line_str}",
                        file=sys.stderr,
                        flush=True,
                    )
                # Always consume the line to prevent pipe buffer from filling
    except Exception as e:
        logger.debug(
            "log_stream.error",
            event_info=f"Error while consuming stream: {e}",
            server_name=server_name,
            error=str(e),
        )
    finally:
        stream.close()


def start_logging_threads(
    process: subprocess.Popen, server_name: str
) -> list[threading.Thread]:
    """Start background threads to consume server output.

    Always starts threads to consume stdout/stderr to prevent subprocess blocking
    when pipe buffers fill up. Only logs output when DEBUG_SERVER_OUTPUT is enabled.

    Args:
        process: The subprocess to consume output from
        server_name: Name of the server for logging

    Returns:
        List of consumption threads (for cleanup if needed)
    """
    threads = []
    # Always consume stdout to prevent blocking, even if not logging
    if process.stdout:
        stdout_thread = threading.Thread(
            target=log_stream,
            args=(process.stdout, "[STDOUT] ", server_name, DEBUG_SERVER_OUTPUT),
            daemon=True,
        )
        stdout_thread.start()
        threads.append(stdout_thread)

    # Always consume stderr to prevent blocking, even if not logging
    # Note: Many applications use stderr for all logging output, not just errors
    if process.stderr:
        stderr_thread = threading.Thread(
            target=log_stream,
            args=(process.stderr, "[LOG] ", server_name, DEBUG_SERVER_OUTPUT),
            daemon=True,
        )
        stderr_thread.start()
        threads.append(stderr_thread)

    return threads
