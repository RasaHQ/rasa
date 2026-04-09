"""Traced MCP Server wrapper for Langfuse integration.

This module provides a wrapper around MCPServerStreamableHttp that adds
Langfuse tracing for MCP tool calls, enabling linked traces across the
client and server boundaries.

Based on the Langfuse MCP tracing guide:
https://langfuse.com/docs/integrations/mcp

Reference implementation:
https://github.com/langfuse/langfuse-examples/tree/main/applications/mcp-tracing
"""

import asyncio
import socket
from contextlib import suppress
from typing import Any, Optional
from urllib.parse import urlparse

import structlog
from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult

from rasa.builder.logging_utils import log_exception
from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
    observe,
    with_langfuse,
)
from rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry import (  # noqa: E501
    MCPLifecycleLangfuseTelemetry,
)

structlogger = structlog.get_logger()


class TracedMCPServerWrapper(MCPServerStreamableHttp):
    """Wrapper around MCPServerStreamableHttp that adds Langfuse tracing.

    This wrapper intercepts MCP tool calls and creates Langfuse spans for each
    tool execution, enabling end-to-end tracing of tool calls within the main
    copilot generation trace.

    The wrapper propagates trace context to the MCP server via HTTP headers,
    allowing the server to link its operations to the client trace.
    """

    def __init__(
        self,
        name: str,
        params: Any,
        client_session_timeout_seconds: int = 120,
        cache_tools_list: bool = True,
        max_retry_attempts: int = 3,
    ) -> None:
        """Initialize the traced MCP server wrapper.

        Args:
            name: Name of the MCP server
            params: Parameters for the MCP server (url, timeout, etc.)
            client_session_timeout_seconds: Timeout for client session
            cache_tools_list: Whether to cache the tools list
            max_retry_attempts: Maximum number of retry attempts
        """
        super().__init__(
            name=name,
            params=params,
            client_session_timeout_seconds=client_session_timeout_seconds,
            cache_tools_list=cache_tools_list,
            max_retry_attempts=max_retry_attempts,
        )
        self._api_endpoint: str = (
            params.get("url", "unknown") if isinstance(params, dict) else "unknown"
        )

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------
    @observe(as_type="generation")
    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> CallToolResult:
        """Call an MCP tool with Langfuse tracing.

        This method wraps the underlying MCP server's call_tool method and
        creates a Langfuse span for the tool execution.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            meta: Optional metadata for the tool call

        Returns:
            The result from the tool execution
        """
        MCPLifecycleLangfuseTelemetry.update_tool_call_span(
            tool_name=tool_name,
            mcp_server_name=self.name,
        )

        trace_id = None
        with with_langfuse() as lf:
            if lf:
                langfuse_client = lf.get_client()
                trace_id = langfuse_client.get_current_trace_id()

        structlogger.debug(
            "traced_mcp_server.tool_call_start",
            tool_name=tool_name,
            arguments=arguments,
            trace_id=trace_id,
        )

        try:
            result = await super().call_tool(tool_name, arguments, meta)

            structlogger.debug(
                "traced_mcp_server.tool_call_complete",
                tool_name=tool_name,
                trace_id=trace_id,
            )

            return result

        except Exception as e:
            log_exception(
                event_name="traced_mcp_server.tool_call_error",
                event_info="MCP tool call failed",
                exc=e,
                tool_name=tool_name,
                trace_id=trace_id,
            )
            MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception(e)
            raise

        except BaseException as e:
            # CancelledError and other BaseExceptions bypass @observe's
            # error tracking. Log so the failure is not completely silent.
            log_exception(
                event_name="traced_mcp_server.tool_call_error.raised_base_exception",
                event_info="MCP tool call failed",
                exc=e,
                tool_name=tool_name,
                trace_id=trace_id,
            )
            MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception(e)
            raise

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @staticmethod
    async def check_health(host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if the MCP server is reachable via a TCP connection.

        Runs the TCP probe off the event loop to avoid blocking when MCP
        is slow or down.

        Args:
            host: Hostname or IP of the MCP server.
            port: Port of the MCP server.
            timeout: Connection timeout in seconds.

        Returns:
            True if the server accepted a TCP connection, False otherwise.
        """

        def _tcp_probe() -> bool:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                return False

        return await asyncio.to_thread(_tcp_probe)

    async def _assert_server_reachable(self) -> None:
        """Raise if the MCP server port is not accepting connections.

        Also traces health to Langfuse (single source of health observability).
        """
        try:
            host, port = self._parse_server_url()
            is_healthy = await self.check_health(host, port)
        except ValueError as e:
            structlogger.warning(
                "traced_mcp_server.assert_server_reachable.error",
                error=str(e),
            )
            is_healthy = False
            host = None
            port = None

        raw_url = self.params.get("url", "") if isinstance(self.params, dict) else ""
        MCPLifecycleLangfuseTelemetry.trace_health(
            is_healthy=is_healthy,
            host=host,
            port=port,
            raw_url=raw_url,
        )

        if not is_healthy:
            structlogger.warning(
                "traced_mcp_server.assert_server_reachable.error",
                error=f"MCP server is not reachable at url: {raw_url}",
            )
            raise ConnectionError(f"MCP server is not reachable at url: {raw_url}")

    def _parse_server_url(self) -> tuple[str, int]:
        """Extract (host, port) from the configured MCP server URL.

        Raises:
            ValueError: If the hostname cannot be determined from the URL.
        """
        url = self.params.get("url", "") if isinstance(self.params, dict) else ""
        if "://" not in url:
            raise ValueError(
                f"MCP server URL must include a scheme "
                f"(e.g. http:// or https://), got: {url!r}"
            )
        parsed_url = urlparse(url)

        if not parsed_url.hostname:
            raise ValueError(f"Cannot determine host from MCP server URL: {url!r}")

        port = parsed_url.port
        if port is None:
            port = 443 if parsed_url.scheme == "https" else 80

        return parsed_url.hostname, port

    async def _diagnose_and_trace_health(self) -> None:
        """Best-effort health probe for diagnostic tracing only.

        Suppresses probe failures (including :class:`asyncio.CancelledError` on Py3.11+
        where it is not a subclass of :class:`Exception`) so diagnostics never replace
        the cancellation from the surrounding ``__aenter__`` / ``__aexit__`` path.
        """
        with suppress(asyncio.CancelledError, Exception):
            await self._assert_server_reachable()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "TracedMCPServerWrapper":
        """Establish the MCP connection, logging and tracing the outcome."""
        try:
            await super().__aenter__()  # type: ignore[no-untyped-call]

            # Connection created successfully, trace health.
            MCPLifecycleLangfuseTelemetry.trace_health(
                is_healthy=True,
                host=None,
                port=None,
                raw_url=self._api_endpoint,
            )
            self._log_and_trace_lifecycle_event(
                event_name="traced_mcp_server.create_connection.created",
                event_info="MCP server connection created",
                metadata={
                    "api_endpoint": self._api_endpoint,
                },
            )
            return self
        except asyncio.CancelledError as e:
            self._log_and_trace_lifecycle_base_exception(
                event_name="traced_mcp_server.create_connection.cancelled",
                event_info="Error during MCP server connection creation",
                exc=e,
                metadata={
                    "api_endpoint": self._api_endpoint,
                    "reason": "error",
                },
            )
            # Best-effort: diagnose and trace MCP server health.
            await self._diagnose_and_trace_health()
            raise
        except Exception as e:
            self._log_and_trace_lifecycle_error(
                event_name="traced_mcp_server.create_connection.error",
                event_info="MCP server connection failed during creation",
                exc=e,
            )
            raise

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the MCP connection, logging and tracing the outcome."""
        try:
            await super().__aexit__(exc_type, exc_val, exc_tb)  # type: ignore[no-untyped-call]

            # The context body raised an exception; cleanup succeeded but
            # the shutdown was triggered by an error, not a normal exit.
            if exc_type is not None:
                self._log_and_trace_lifecycle_error(
                    event_name="traced_mcp_server.close_connection.closed_on_error",
                    event_info="MCP server connection closed due to error",
                    exc=exc_val,
                )
            # Normal shutdown — no exception from the context body.
            else:
                self._log_and_trace_lifecycle_event(
                    event_name="traced_mcp_server.close_connection.closed",
                    event_info="MCP server connection closed normally",
                    metadata={
                        "api_endpoint": self._api_endpoint,
                        "reason": "normal",
                    },
                )
            return

        # Handle exceptions during cleanup/teardown
        except asyncio.CancelledError as e:
            self._log_and_trace_lifecycle_base_exception(
                event_name="traced_mcp_server.close_connection.cancelled",
                event_info="Error during MCP server connection cleanup",
                exc=e,
                metadata={
                    "api_endpoint": self._api_endpoint,
                    "reason": "error",
                },
            )
            raise
        except Exception as e:
            self._log_and_trace_lifecycle_error(
                event_name="traced_mcp_server.close_connection.cleanup_error",
                event_info="Error during MCP server connection cleanup",
                exc=e,
            )
            raise

    # ------------------------------------------------------------------
    # Error logging helpers
    # ------------------------------------------------------------------

    def _log_and_trace_lifecycle_event(
        self,
        event_name: str,
        event_info: str,
        metadata: dict[str, Any],
    ) -> None:
        structlogger.info(
            event_name,
            event_info=event_info,
            **metadata,
        )
        MCPLifecycleLangfuseTelemetry.emit_lifecycle_event(
            span_name=event_name,
            mcp_server_name=self.name,
            api_endpoint=self._api_endpoint,
            metadata=metadata,
        )

    def _log_and_trace_lifecycle_error(
        self,
        event_name: str,
        event_info: str,
        exc: Optional[BaseException],
        reason: str = "error",
    ) -> None:
        """Log an exception and emit a Langfuse lifecycle span."""
        fields: dict[str, Any]
        if exc is not None:
            exception_fields = log_exception(
                event_name=event_name,
                event_info=event_info,
                exc=exc,
                api_endpoint=self._api_endpoint,
                reason=reason,
            )
            fields = exception_fields.to_dict()
        else:
            fields = {}
        MCPLifecycleLangfuseTelemetry.emit_lifecycle_event(
            span_name=event_name,
            mcp_server_name=self.name,
            api_endpoint=self._api_endpoint,
            metadata={
                "api_endpoint": self._api_endpoint,
                "reason": reason,
            },
            output=fields,
            level="ERROR",
        )

    def _log_and_trace_lifecycle_base_exception(
        self,
        event_name: str,
        event_info: str,
        exc: BaseException,
        metadata: dict[str, Any],
    ) -> None:
        """Log a base exception and emit a dedicated Langfuse lifecycle span."""
        exception_fields = log_exception(
            event_name=event_name,
            event_info=event_info,
            exc=exc,
            **metadata,
        )
        MCPLifecycleLangfuseTelemetry.emit_lifecycle_event(
            span_name=event_name,
            mcp_server_name=self.name,
            api_endpoint=self._api_endpoint,
            metadata=metadata,
            output=exception_fields.to_dict(),
            level="ERROR",
        )
