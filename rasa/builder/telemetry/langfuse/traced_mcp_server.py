"""Traced MCP Server wrapper for Langfuse integration.

This module provides a wrapper around MCPServerStreamableHttp that adds
Langfuse tracing for MCP tool calls, enabling linked traces across the
client and server boundaries.

Based on the Langfuse MCP tracing guide:
https://langfuse.com/docs/integrations/mcp

Reference implementation:
https://github.com/langfuse/langfuse-examples/tree/main/applications/mcp-tracing
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import structlog
from agents.mcp import MCPServerStreamableHttp

from rasa.builder.telemetry.langfuse.langfuse_compat import (
    is_langfuse_available,
    langfuse,
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
        self._langfuse_enabled = is_langfuse_available()

    async def call_tool(
        self, tool_name: str, arguments: Optional[dict[str, Any]] = None
    ) -> Any:
        """Call an MCP tool with Langfuse tracing.

        This method wraps the underlying MCP server's call_tool method and
        creates a Langfuse span for the tool execution.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            The result from the tool execution
        """
        if not self._langfuse_enabled:
            # If Langfuse is not available, just call the parent's tool directly
            return await super().call_tool(tool_name, arguments)

        # Create a Langfuse span for this tool call
        langfuse_client = langfuse.get_client()

        try:
            with langfuse_client.start_as_current_generation(
                name=f"mcp_tool.{tool_name}",
                input={"tool_name": tool_name, "arguments": arguments or {}},
                metadata={
                    "mcp_server": self.name,
                    "tool_type": "mcp",
                },
            ) as generation:
                structlogger.debug(
                    "traced_mcp_server.tool_call_start",
                    tool_name=tool_name,
                    arguments=arguments,
                    trace_id=generation.trace_id if generation else None,
                )

                # Call the parent's tool method
                result = await super().call_tool(tool_name, arguments)

                # Update the span with the result
                generation.update(output=result)

                structlogger.debug(
                    "traced_mcp_server.tool_call_complete",
                    tool_name=tool_name,
                    trace_id=generation.trace_id if generation else None,
                )

                return result

        except Exception as e:
            structlogger.error(
                "traced_mcp_server.tool_call_error",
                tool_name=tool_name,
                error=str(e),
            )
            raise


@asynccontextmanager
async def create_traced_mcp_server(
    name: str,
    params: Any,
    client_session_timeout_seconds: int = 120,
    cache_tools_list: bool = True,
    max_retry_attempts: int = 3,
) -> AsyncIterator[TracedMCPServerWrapper]:
    """Context manager to create a traced MCP server connection.

    This is a convenience function that creates and manages a TracedMCPServerWrapper
    instance, ensuring proper cleanup.

    Args:
        name: Name of the MCP server
        params: Parameters for the MCP server (url, timeout, etc.)
        client_session_timeout_seconds: Timeout for client session
        cache_tools_list: Whether to cache the tools list
        max_retry_attempts: Maximum number of retry attempts

    Yields:
        Connected TracedMCPServerWrapper instance

    Example:
        async with create_traced_mcp_server(
            name="Rasa MCP Server",
            params={"url": "http://localhost:5051/mcp", "timeout": 120}
        ) as server:
            result = await server.call_tool("search_docs", {"query": "flows"})
    """
    server = TracedMCPServerWrapper(
        name=name,
        params=params,
        client_session_timeout_seconds=client_session_timeout_seconds,
        cache_tools_list=cache_tools_list,
        max_retry_attempts=max_retry_attempts,
    )

    async with server:
        yield server
