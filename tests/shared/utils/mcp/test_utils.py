"""Tests for MCP utilities."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.shared.utils.mcp.utils import call_tool_with_meta


@pytest.mark.asyncio
async def test_call_tool_with_meta_fallback_when_sdk_rejects_meta():
    """When the MCP server/SDK does not accept tool call with meta (TypeError),
    call_tool_with_meta falls back to calling without the meta parameter.
    """
    tool_name = "test_tool"
    arguments = {"key": "value"}
    read_timeout = timedelta(seconds=30)
    meta = {"session_id": "s1", "user_id": "u1"}

    fallback_result = MagicMock()
    fallback_result.isError = False
    fallback_result.content = [{"text": "ok"}]

    session = AsyncMock()
    # First call (with meta) raises TypeError - SDK doesn't support meta
    session.call_tool = AsyncMock(
        side_effect=[
            TypeError("call_tool() got an unexpected keyword argument 'meta'"),
            fallback_result,
        ]
    )

    result = await call_tool_with_meta(
        session, tool_name, arguments, read_timeout, meta
    )

    assert result is fallback_result
    assert session.call_tool.call_count == 2

    # First call: with meta
    session.call_tool.assert_any_call(
        tool_name,
        arguments,
        read_timeout_seconds=read_timeout,
        meta=meta,
    )
    # Second call: without meta (fallback)
    session.call_tool.assert_any_call(
        tool_name,
        arguments,
        read_timeout_seconds=read_timeout,
    )
