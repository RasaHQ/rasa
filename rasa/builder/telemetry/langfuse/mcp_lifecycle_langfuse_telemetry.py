"""Langfuse telemetry for MCP server connection lifecycle.

Emits in-trace spans within the current copilot endpoint trace so that
MCP lifecycle events (connection created, closed, error) are visible
alongside tool calls.
"""

from typing import Any, Optional

import structlog

from rasa.builder.logging_utils import ExceptionFields
from rasa.builder.telemetry.langfuse.langfuse_compat import (
    with_langfuse,
)

structlogger = structlog.get_logger()


class MCPLifecycleLangfuseTelemetry:
    """Langfuse telemetry helpers for MCP connection lifecycle events."""

    @staticmethod
    def trace_health(
        is_healthy: bool,
        host: Optional[str],
        port: Optional[int],
        raw_url: Optional[str],
    ) -> None:
        """Trace MCP server health status as a span in the current trace.

        Observational only; failures (e.g. get_client or span creation) are
        logged and swallowed so they do not abort the request.

        Args:
            is_healthy: Whether the MCP server is reachable.
            host: MCP server host. Can be None if the host cannot be determined.
            port: MCP server port. Can be None if the port cannot be determined.
            raw_url: Raw URL of the MCP server. Can be None if the URL cannot be
                determined.
        """
        try:
            with with_langfuse() as lf:
                if not lf:
                    return

                langfuse_client = lf.get_client()
                status = "running" if is_healthy else "not_running"
                with langfuse_client.start_as_current_span(
                    name="mcp_lifecycle.health_check",
                ) as span:
                    span.update(
                        output={"mcp_server_health": status},
                        input={
                            "host": host,
                            "port": port,
                            "raw_url": raw_url,
                        },
                    )
        except Exception as e:
            structlogger.debug(
                "mcp_lifecycle_telemetry.trace_health.failed",
                error=repr(e),
            )

    @staticmethod
    def emit_lifecycle_event(
        span_name: str,
        mcp_server_name: str,
        api_endpoint: str,
        metadata: dict[str, Any],
        output: Optional[dict[str, Any]] = None,
        level: Optional[str] = None,
    ) -> None:
        """Emit a Langfuse span for an MCP lifecycle event.

        The span is created inside the current trace context (typically the copilot
        endpoint trace) so it appears next to tool-call observations.

        Args:
            span_name: Name of the lifecycle span.
            mcp_server_name: Name of the MCP server.
            api_endpoint: API endpoint of the MCP server.
            metadata: Metadata to attach to the span (filterable context).
            output: Optional output/result of the span (use for error details).
            level: Optional span level (e.g. "ERROR" for failure spans).
        """
        try:
            with with_langfuse() as lf:
                if not lf:
                    return
                langfuse_client = lf.get_client()
                with langfuse_client.start_as_current_span(name=span_name) as span:
                    span.update(
                        level=level,
                        input={
                            "mcp_server_name": mcp_server_name,
                            "api_endpoint": api_endpoint,
                        },
                        output=output,
                        metadata=metadata,
                    )
        except Exception as e:
            structlogger.debug(
                "mcp_lifecycle_telemetry.emit_lifecycle_event.failed",
                error=repr(e),
            )

    @staticmethod
    def update_tool_call_span(
        tool_name: str,
        mcp_server_name: str,
    ) -> None:
        """Rename the current generation span to ``mcp_tool.<tool_name>``."""
        try:
            with with_langfuse() as lf:
                if not lf:
                    return
                langfuse_client = lf.get_client()
                langfuse_client.update_current_span(
                    name=f"mcp_tool.{tool_name}",
                    metadata={
                        "mcp_server": mcp_server_name,
                        "tool_type": "mcp",
                    },
                )
        except Exception as e:
            structlogger.debug(
                "mcp_lifecycle_telemetry.update_tool_call_span.failed",
                error=repr(e),
            )

    @staticmethod
    def mark_current_span_with_base_exception(error: BaseException) -> None:
        """Best-effort: mark the current Langfuse span as ERROR."""
        try:
            with with_langfuse() as lf:
                if not lf:
                    return
                langfuse_client = lf.get_client()
                fields = ExceptionFields.from_exception(error).to_dict()
                langfuse_client.update_current_span(
                    level="ERROR",
                    output=fields,
                )
        except Exception as e:
            structlogger.debug(
                "mcp_lifecycle_telemetry.mark_current_span_with_base_exception.failed",
                error=repr(e),
            )
