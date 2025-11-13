"""MCP server connection utilities."""

import asyncio
import warnings
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, ClassVar, Dict, List, Optional

import structlog
from httpx import HTTPStatusError
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from rasa.shared.agents.auth.agent_auth_manager import AgentAuthManager
from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.exceptions import AuthenticationError

structlogger = structlog.get_logger()


# Suppress RuntimeWarning about unawaited coroutines when MCP server is not reachable.
warnings.filterwarnings(
    "ignore",
    message=".*BaseEventLoop.create_server.*was never awaited.*",
    category=RuntimeWarning,
)


class MCPServerConnection:
    """
    Manages connection to an MCP server, with optional authentication.

    This class handles:
    - Connection establishment and cleanup
    - Session lifecycle management
    - Authentication via AgentAuthManager (API Key, OAuth2, mTLS, etc.)
    """

    # Timeout for ping operations in seconds
    PING_TIMEOUT_SECONDS = 3.0

    _SUPPORTED_SERVER_TYPES: ClassVar[List[str]] = ["http", "https"]

    def __init__(
        self,
        server_name: str,
        server_url: str,
        server_type: str,
        auth_manager: Optional[AgentAuthManager] = None,
    ):
        """
        Initialize the MCP server connection.

        Args:
            server_name: Server name to identify the server
            server_url: Server URL
            server_type: Server type (currently only 'http' is supported)
            auth_manager: Optional AgentAuthManager instance for this connection
        """
        self.server_name = server_name
        self.server_url = server_url
        self.server_type = server_type
        self._auth_manager = auth_manager
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None

    @classmethod
    def from_config(cls, server_config: Dict[str, Any]) -> "MCPServerConnection":
        """Initialize the MCP server connection from a configuration dictionary."""
        auth_config = server_config.get("additional_params")
        _auth_manager = AgentAuthManager.load_auth(auth_config)
        return cls(
            server_config["name"],
            server_config["url"],
            server_config.get("type", "http"),
            _auth_manager,
        )

    @staticmethod
    @asynccontextmanager
    async def open_mcp_session(
        url: str, auth_strategy: Optional[AgentAuthStrategy] = None
    ) -> AsyncIterator[ClientSession]:
        """
        Open a streamable MCP session, ensuring that initialization
        completes before yielding.
        """
        async with streamablehttp_client(url, auth=auth_strategy) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()  # handshake done here
                yield session

    async def connect(self) -> None:
        """Establish connection to the MCP server.

        Raises:
            ValueError: If the server type is not supported.
            ConnectionError: If connection fails.
            AuthenticationError: If authentication fails.
        """
        if self.server_type not in self._SUPPORTED_SERVER_TYPES:
            raise ValueError(f"Unsupported server type: {self.server_type}")

        # Create a new exit stack for this connection to avoid task boundary issues
        self.exit_stack = AsyncExitStack()

        try:
            # Get authentication strategy that adheres to httpx.Auth.
            auth_strategy = (
                self._auth_manager.get_auth() if self._auth_manager else None
            )

            # Register the wrapped context manager in the stack
            self.session = await self.exit_stack.enter_async_context(
                self.open_mcp_session(self.server_url, auth_strategy)
            )

        except Exception as eg:
            for exc in getattr(eg, "exceptions", [eg]):
                event_info = (
                    f"Failed to connect to MCP server `{self.server_name}`. \nOriginal "
                    f"error: {exc!s}"
                )
                if isinstance(exc, HTTPStatusError):
                    status_code = exc.response.status_code
                    structlogger.error(
                        "mcp_server_connection.connect.http_status_error",
                        event_info=event_info,
                        server_name=self.server_name,
                        server_url=self.server_url,
                        status_code=status_code,
                        response_text=exc.response.reason_phrase,
                    )
                    await self._cleanup()
                    if status_code in [400, 401, 403]:
                        raise AuthenticationError(str(exc)) from eg
                    elif status_code == 404:
                        raise Exception(str(exc)) from eg
                    else:
                        raise ConnectionError(str(exc)) from eg
                else:
                    structlogger.error(
                        "mcp_server_connection.connect.other_exception",
                        event_info=event_info,
                        server_name=self.server_name,
                        server_url=self.server_url,
                        error=str(exc),
                    )
            await self._cleanup()
            raise ConnectionError(str(exc)) from eg

        except asyncio.CancelledError as e:
            event_info = f"Connection to MCP server `{self.server_name}` was cancelled."
            structlogger.error(
                "mcp_server_connection.connect.connection_cancelled",
                event_info=event_info,
                server_name=self.server_name,
                server_url=self.server_url,
            )
            # Clean up on cancellation
            await self._cleanup()
            raise ConnectionError(e) from e

    async def ensure_active_session(self) -> ClientSession:
        """
        Ensure an active session is available.

        If no session exists or the current session is inactive,
        a new connection will be established.

        Returns:
            Active ClientSession instance.
        """
        if self.session is None:
            await self.connect()
            structlogger.info(
                "mcp_server_connection.ensure_active_session.no_session",
                server_name=self.server_name,
                server_url=self.server_url,
                event_info=(
                    "No session found, connecting to the server "
                    f"`{self.server_name}` @ `{self.server_url}`"
                ),
            )
        if self.session:
            try:
                # Add timeout to prevent hanging when MCP server is down
                await asyncio.wait_for(
                    self.session.send_ping(), timeout=self.PING_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError as e:
                structlogger.error(
                    "mcp_server_connection.ensure_active_session.ping_timeout",
                    server_name=self.server_name,
                    server_url=self.server_url,
                    event_info=(
                        "Ping timed out, Server not reachable - "
                        f"`{self.server_name}` @ `{self.server_url}`"
                    ),
                )
                raise e
            except Exception as e:
                structlogger.warning(
                    "mcp_server_connection.ensure_active_session.ping_failed",
                    error=str(e),
                    server_name=self.server_name,
                    server_url=self.server_url,
                    event_info=(
                        "Ping failed, trying to reconnect to the server "
                        f"`{self.server_name}` @ `{self.server_url}`"
                    ),
                )
                # Cleanup existing session
                await self.close()
                # Attempt to reconnect now
                await self.connect()
                structlogger.info(
                    "mcp_server_connection.ensure_active_session.reconnected",
                    server_name=self.server_name,
                    server_url=self.server_url,
                    event_info=(
                        "Reconnected to the server "
                        f"`{self.server_name}` @ `{self.server_url}`"
                    ),
                )
        assert self.session is not None  # Ensures type for mypy
        return self.session

    async def close(self) -> None:
        """Close the connection and clean up resources."""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Internal cleanup method to safely close resources."""
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
            except asyncio.CancelledError:
                # Handle cancellation gracefully - this is expected during shutdown
                structlogger.debug(
                    "mcp_server_connection.cleanup.cancelled",
                    server_name=self.server_name,
                    event_info=(
                        f"Cleanup cancelled for {self.server_name} - this is expected "
                        f"during shutdown"
                    ),
                )
            except Exception as e:
                # Handle other errors
                structlogger.warning(
                    "mcp_server_connection.cleanup.failed",
                    server_name=self.server_name,
                    error=str(e),
                )
            finally:
                self.exit_stack = None
                self.session = None
