"""FastMCP tool-decorator instrumentation for Segment telemetry.

Wraps ``FastMCP.tool(...)`` so every ``@mcp.tool(...)`` declaration emits a
single ``mcp_tool_called`` event on each invocation.
"""

import asyncio
import functools
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import structlog
from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rasa.builder.telemetry.segment_integration.mcp_tools_segment_telemetry import (
        MCPToolsSegmentTelemetry,
    )

structlogger = structlog.get_logger()

_mcp_telemetry: Optional["MCPToolsSegmentTelemetry"] = None
_mcp_server_name: Optional[str] = None


def _get_mcp_telemetry() -> "MCPToolsSegmentTelemetry":
    """Lazily build the telemetry sender on first tool invocation.

    Defers importing ``MCPToolsSegmentTelemetry`` (and transitively
    ``rasa.telemetry``) until the first tool call to keep server startup fast.
    """
    global _mcp_telemetry
    if _mcp_telemetry is None:
        from rasa.builder.telemetry.segment_integration.mcp_tools_segment_telemetry import (  # noqa: E501
            MCPToolsSegmentTelemetry,
        )

        assert (
            _mcp_server_name is not None
        ), "instrument_mcp_tools() must be called before any tool invocation"
        _mcp_telemetry = MCPToolsSegmentTelemetry(mcp_server=_mcp_server_name)
    return _mcp_telemetry


def instrument_mcp_tools(mcp_instance: FastMCP, *, mcp_server: str) -> None:
    """Wrap ``mcp_instance.tool(...)`` with Segment telemetry.

    Emits an invocation log line and a Segment ``mcp_tool_called`` event for
    each invocation. Must be called once after the FastMCP instance is
    constructed and BEFORE any ``@mcp.tool(...)`` decorator runs; the tool
    body itself is unchanged.
    """
    global _mcp_server_name
    _mcp_server_name = mcp_server

    original_tool = mcp_instance.tool

    def instrumented_tool(*args: Any, **kwargs: Any) -> Callable:
        register = original_tool(*args, **kwargs)
        tool_name = kwargs.get("name")

        def wrap(fn: Callable) -> Callable:
            resolved_name = tool_name or fn.__name__

            @functools.wraps(fn)
            async def instrumented(*a: Any, **kw: Any) -> Any:
                structlogger.info(
                    "rasa_mcp_server.tool_invoked",
                    event_info=f"Rasa MCP tool '{resolved_name}' invoked",
                    tool_name=resolved_name,
                )
                start = time.perf_counter()
                success = False
                error_type: Optional[str] = None
                error_message: Optional[str] = None
                try:
                    result = await fn(*a, **kw)
                    # Tools report failures two ways: by raising, or by
                    # returning a structured response with ``success=False``
                    # (so the LLM can act on the error). Treat both as
                    # failures for telemetry.
                    success = getattr(result, "success", True)
                    if not success:
                        error_type = "ToolReportedFailure"
                        error_message = getattr(result, "error", None)
                    return result
                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    raise
                finally:
                    duration_ms = int((time.perf_counter() - start) * 1000)
                    try:
                        telemetry_ = _get_mcp_telemetry()
                        await asyncio.to_thread(
                            telemetry_.track_tool_called,
                            tool_name=resolved_name,
                            duration_ms=duration_ms,
                            success=success,
                            error_type=error_type,
                            error_message=error_message,
                        )
                    except Exception as e:
                        # Telemetry must never break a tool call.
                        structlogger.debug(
                            "rasa_mcp_server.telemetry.failed",
                            tool_name=resolved_name,
                            error=str(e),
                        )

            return register(instrumented)

        return wrap

    mcp_instance.tool = instrumented_tool  # type: ignore[method-assign]
