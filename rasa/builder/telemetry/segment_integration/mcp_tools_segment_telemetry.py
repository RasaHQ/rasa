"""Segment telemetry for MCP tool invocations.

MCP tools can be invoked by the Rasa Copilot (via the agent SDK) or by
third-party agents (Claude Code, Cursor, etc.) connecting directly to the
Rasa MCP server.

A single ``mcp_tool_called`` event is emitted per completed invocation. On
failure it carries ``success=False`` along with ``error_type`` and
``error_message``.
"""

from typing import Optional

from rasa.builder.telemetry.segment_integration.segment_compat import (
    resolve_default_user_id,
    track,
)
from rasa.builder.telemetry.segment_integration.shared import (
    MCP_TOOL_CALLED_EVENT,
    now_iso,
)


class MCPToolsSegmentTelemetry:
    """Stateful telemetry sender bound to a single MCP server identity.

    Instantiate once per MCP server process; call ``track_tool_called`` from
    the tool-invocation wrapper.
    """

    def __init__(
        self,
        *,
        mcp_server: str,
        user_id: Optional[str] = None,
    ) -> None:
        self._mcp_server = mcp_server
        self._user_id = user_id or resolve_default_user_id()

    def track_tool_called(
        self,
        *,
        tool_name: str,
        duration_ms: int,
        success: bool,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a completed tool invocation.

        ``error_type`` and ``error_message`` are populated when
        ``success`` is ``False`` — either because the tool raised, or
        because it returned a response with ``success=False``.
        """
        track(
            MCP_TOOL_CALLED_EVENT,
            self._user_id,
            {
                "tool_name": tool_name,
                "mcp_server": self._mcp_server,
                "duration_ms": duration_ms,
                "success": success,
                "error_type": error_type,
                "error_message": error_message,
                "timestamp": now_iso(),
            },
        )
