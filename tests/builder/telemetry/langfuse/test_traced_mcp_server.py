"""Tests for TracedMCPServerWrapper."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from agents.mcp import MCPServerStreamableHttp

from rasa.builder.telemetry.langfuse.traced_mcp_server import (
    TracedMCPServerWrapper,
    create_traced_mcp_server,
)


class TestTracedMCPServerWrapper:
    """Tests for the TracedMCPServerWrapper class."""

    def test_initialization(self) -> None:
        """Test that TracedMCPServerWrapper can be initialized."""
        wrapper = TracedMCPServerWrapper(
            name="Mock MCP Server",
            params={"url": "mock:5051/mcp", "timeout": 120},
            client_session_timeout_seconds=120,
            cache_tools_list=True,
            max_retry_attempts=3,
        )

        assert wrapper is not None
        assert isinstance(wrapper, TracedMCPServerWrapper)

    def test_langfuse_enabled_flag(self) -> None:
        """Test that the Langfuse enabled flag is set correctly."""
        from rasa.builder.telemetry.langfuse.langfuse_compat import (
            is_langfuse_available,
        )

        wrapper = TracedMCPServerWrapper(
            name="Mock MCP Server",
            params={"url": "mock:5051/mcp", "timeout": 120},
        )

        # The wrapper should have the same langfuse availability as the system
        assert wrapper._langfuse_enabled == is_langfuse_available()

    @pytest.mark.asyncio
    @patch.object(MCPServerStreamableHttp, "call_tool", new_callable=AsyncMock)
    @patch(
        "rasa.builder.telemetry.langfuse.traced_mcp_server.is_langfuse_available",
        return_value=False,
    )
    async def test_call_tool_without_langfuse(
        self,
        mock_is_langfuse_available: Mock,
        mock_mcp_server_call_tool: AsyncMock,
    ) -> None:
        """Test call_tool when Langfuse is not available."""
        wrapper = TracedMCPServerWrapper(
            name="Mock MCP Server",
            params={"url": "mock:5051/mcp", "timeout": 120},
        )

        # Set the return value
        expected_result = {"status": "success", "data": "test_data"}
        mock_mcp_server_call_tool.return_value = expected_result

        result = await wrapper.call_tool("test_tool", {"arg1": "value1"})

        # Verify parent's call_tool was called
        mock_mcp_server_call_tool.assert_called_once_with(
            "test_tool", {"arg1": "value1"}
        )
        assert result == expected_result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name,arguments,expected_result,side_effect,expected_exception,verify_update",
        [
            # Success case with arguments
            (
                "search_docs",
                {"query": "flows"},
                {"status": "success", "output": "tool_output"},
                None,
                None,
                True,
            ),
            # None arguments case
            (
                "list_tools",
                None,
                {"status": "success"},
                None,
                None,
                True,
            ),
            # Empty arguments case
            (
                "simple_tool",
                {},
                {"status": "success"},
                None,
                None,
                True,
            ),
            # Exception case
            (
                "failing_tool",
                {"param": "value"},
                None,
                ValueError("Tool execution failed"),
                ValueError,
                False,
            ),
        ],
    )
    @patch.object(MCPServerStreamableHttp, "call_tool", new_callable=AsyncMock)
    @patch("rasa.builder.telemetry.langfuse.traced_mcp_server.langfuse.get_client")
    @patch(
        "rasa.builder.telemetry.langfuse.traced_mcp_server.is_langfuse_available",
        return_value=True,
    )
    async def test_call_tool_with_langfuse(
        self,
        mock_is_langfuse_available: Mock,
        mock_get_client: Mock,
        mock_mcp_server_call_tool: AsyncMock,
        tool_name: str,
        arguments: dict | None,
        expected_result: dict | None,
        side_effect: Exception | None,
        expected_exception: type[Exception] | None,
        verify_update: bool,
    ) -> None:
        """Test call_tool with Langfuse enabled - various scenarios."""
        wrapper = TracedMCPServerWrapper(
            name="Mock MCP Server",
            params={"url": "mock:5051/mcp", "timeout": 120},
        )

        # Mock langfuse client and generation
        mock_generation = MagicMock()
        mock_generation.trace_id = "test-trace-123"
        mock_generation.__enter__ = Mock(return_value=mock_generation)
        mock_generation.__exit__ = Mock(return_value=False)

        mock_client = MagicMock()
        mock_client.start_as_current_generation.return_value = mock_generation
        mock_get_client.return_value = mock_client

        # Setup parent call_tool mock
        if side_effect:
            mock_mcp_server_call_tool.side_effect = side_effect
        else:
            mock_mcp_server_call_tool.return_value = expected_result

        if expected_exception:
            # Verify exception is re-raised
            with pytest.raises(expected_exception):
                await wrapper.call_tool(tool_name, arguments)
        else:
            result = await wrapper.call_tool(tool_name, arguments)

            # Verify langfuse span was created with correct parameters
            expected_args = arguments if arguments is not None else {}
            mock_client.start_as_current_generation.assert_called_once_with(
                name=f"mcp_tool.{tool_name}",
                input={
                    "tool_name": tool_name,
                    "arguments": expected_args,
                },
                metadata={
                    "mcp_server": "Mock MCP Server",
                    "tool_type": "mcp",
                },
            )

            # Verify parent's call_tool was called
            mock_mcp_server_call_tool.assert_called_once_with(tool_name, arguments)

            if verify_update:
                # Verify generation was updated with result
                mock_generation.update.assert_called_once_with(output=expected_result)

            # Verify result is returned
            assert result == expected_result


class TestCreateTracedMCPServer:
    """Tests for the create_traced_mcp_server context manager."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name,params,kwargs,should_raise,exception_type,verify_name",
        [
            # Basic case with all parameters
            (
                "Test Server",
                {"url": "mock:5051/mcp", "timeout": 120},
                {
                    "client_session_timeout_seconds": 60,
                    "cache_tools_list": False,
                    "max_retry_attempts": 5,
                },
                False,
                None,
                True,
            ),
            # Default parameters case
            (
                "Default Server",
                {"url": "mock:5051/mcp"},
                {},
                False,
                None,
                True,
            ),
            # Exception handling case
            (
                "Failing Server",
                {"url": "mock:5051/mcp"},
                {},
                True,
                RuntimeError,
                False,
            ),
        ],
    )
    @patch.object(TracedMCPServerWrapper, "__aexit__", new_callable=AsyncMock)
    @patch.object(TracedMCPServerWrapper, "__aenter__", new_callable=AsyncMock)
    async def test_create_traced_mcp_server(
        self,
        mock_aenter: AsyncMock,
        mock_aexit: AsyncMock,
        name: str,
        params: dict,
        kwargs: dict,
        should_raise: bool,
        exception_type: type[Exception] | None,
        verify_name: bool,
    ) -> None:
        """Test create_traced_mcp_server with various scenarios."""
        mock_aenter.return_value = None

        if should_raise:
            # Make __aexit__ return False to propagate the exception
            mock_aexit.return_value = False
            assert exception_type is not None
            test_exception = exception_type("Connection failed")

            with pytest.raises(exception_type, match="Connection failed"):
                async with create_traced_mcp_server(
                    name=name,
                    params=params,
                    **kwargs,
                ):
                    raise test_exception

            # Verify __aexit__ was called even with exception
            mock_aexit.assert_called_once()
        else:
            async with create_traced_mcp_server(
                name=name,
                params=params,
                **kwargs,
            ) as server:
                # Verify we got a TracedMCPServerWrapper instance
                assert isinstance(server, TracedMCPServerWrapper)
                if verify_name:
                    assert server.name == name

            # Verify context manager methods were called
            mock_aenter.assert_called_once()
            mock_aexit.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(MCPServerStreamableHttp, "call_tool", new_callable=AsyncMock)
    @patch(
        "rasa.builder.telemetry.langfuse.traced_mcp_server.is_langfuse_available",
        return_value=False,
    )
    @patch.object(TracedMCPServerWrapper, "__aexit__", new_callable=AsyncMock)
    @patch.object(TracedMCPServerWrapper, "__aenter__", new_callable=AsyncMock)
    async def test_create_traced_mcp_server_can_call_tool(
        self,
        mock_aenter: AsyncMock,
        mock_aexit: AsyncMock,
        mock_is_langfuse_available: Mock,
        mock_mcp_server_call_tool: AsyncMock,
    ) -> None:
        """Test that server created by context manager can call tools."""
        # Set the return value
        mock_mcp_server_call_tool.return_value = {"result": "success"}

        async with create_traced_mcp_server(
            name="Tool Server",
            params={"url": "mock:5051/mcp"},
        ) as server:
            result = await server.call_tool("test_tool", {"arg": "val"})

            mock_mcp_server_call_tool.assert_called_once_with(
                "test_tool", {"arg": "val"}
            )
            assert result == {"result": "success"}
