"""Agent hooks for tracking tool execution events."""

import asyncio
from typing import Any

import structlog
from agents import Agent, AgentHooks, Tool
from agents.run_context import RunContextWrapper

from rasa.builder.copilot.models import MCPToolCall, MCPToolCallStatus

structlogger = structlog.get_logger()


class RasaCopilotHooks(AgentHooks):
    """Custom AgentHooks for tracking MCP tool events.

    This hooks class captures tool start and end events and queues them
    for processing by the response handler.
    """

    def __init__(self, mcp_queue: asyncio.Queue[MCPToolCall]) -> None:
        """Initialize the hooks with a queue for MCP tool call events.

        Args:
            mcp_queue: Queue to put MCPToolCall events for processing.
        """
        self._mcp_queue = mcp_queue

    async def on_tool_start(
        self, context: RunContextWrapper, agent: Agent, tool: Tool
    ) -> None:
        """Called when a tool starts execution."""
        structlogger.debug(
            "agent_sdk.tool.start",
            tool_name=tool.name,
        )
        # Queue MCP tool call event
        await self._mcp_queue.put(
            MCPToolCall(
                tool_name=tool.name,
                status=MCPToolCallStatus.CALLED,
            )
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: Any
    ) -> None:
        """Called when a tool completes execution."""
        structlogger.debug(
            "agent_sdk.tool.end",
            tool_name=tool.name,
        )
        # Queue MCP tool completion event
        await self._mcp_queue.put(
            MCPToolCall(
                tool_name=tool.name,
                status=MCPToolCallStatus.COMPLETED,
                output=result,
            )
        )
