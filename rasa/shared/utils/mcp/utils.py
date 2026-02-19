"""MCP utilities."""

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, Optional

import structlog

if TYPE_CHECKING:
    from mcp.client.session import ClientSession
    from mcp.types import CallToolResult

    from rasa.core.config.available_endpoints import MCPMetaMapConfig

structlogger = structlog.get_logger()


async def call_tool_with_meta(
    session: "ClientSession",
    tool_name: str,
    arguments: Dict[str, Any],
    read_timeout_seconds: timedelta,
    meta: Dict[str, Any],
) -> "CallToolResult":
    """Call MCP session call_tool, passing meta if the SDK supports it.

    Older MCP SDK versions do not accept a meta parameter; this function
    falls back to calling without meta in that case.
    """
    if meta:
        structlogger.debug(
            "mcp_agent.execute_mcp_tool.meta_keys",
            tool_name=tool_name,
            meta_keys=list(meta.keys()),
        )
        try:
            return await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=read_timeout_seconds,
                meta=meta,
            )
        except TypeError:
            # SDK version does not support meta parameter
            pass
    return await session.call_tool(
        tool_name,
        arguments,
        read_timeout_seconds=read_timeout_seconds,
    )


def mcp_server_exists(mcp_server: str) -> bool:
    """Check if an MCP server exists in the configured endpoints.

    Args:
        mcp_server: The name of the MCP server to check.

    Returns:
        True if the MCP server exists, False otherwise.
    """
    from rasa.core.config.configuration import Configuration

    endpoints = Configuration.get_instance().endpoints
    if (mcp_server_list := endpoints.mcp_servers) is None:
        return False

    mcp_server_names = [server.name for server in mcp_server_list]
    return mcp_server in mcp_server_names


def build_mcp_meta(
    meta_map_config: Optional["MCPMetaMapConfig"],
    slots: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the _meta dict for an MCP tool call from config and slot values.

    Static entries are added first; then from_slots entries.
    """
    if not meta_map_config:
        return {}
    meta: Dict[str, Any] = {}
    if meta_map_config.static:
        meta.update(meta_map_config.static)
    if meta_map_config.from_slots:
        for entry in meta_map_config.from_slots:
            meta[entry.param] = slots.get(entry.slot)
    return meta
