"""Tests for TracedMCPServerWrapper."""

import asyncio
from typing import ClassVar
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents.mcp import MCPServerStreamableHttp

from rasa.builder.telemetry.langfuse_integration.traced_mcp_server import (
    TracedMCPServerWrapper,
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

    @pytest.mark.asyncio
    @patch.object(MCPServerStreamableHttp, "call_tool", new_callable=AsyncMock)
    @patch(
        "rasa.builder.telemetry.langfuse_integration.langfuse_compat.is_langfuse_available",
        return_value=False,
    )
    async def test_call_tool_without_langfuse(
        self,
        mock_is_langfuse_available: AsyncMock,
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
            "test_tool", {"arg1": "value1"}, None
        )
        assert result == expected_result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name,arguments,expected_result,side_effect,expected_exception",
        [
            # Success case with arguments
            (
                "search_docs",
                {"query": "flows"},
                {"status": "success", "output": "tool_output"},
                None,
                None,
            ),
            # None arguments case
            (
                "list_tools",
                None,
                {"status": "success"},
                None,
                None,
            ),
            # Empty arguments case
            (
                "simple_tool",
                {},
                {"status": "success"},
                None,
                None,
            ),
            # Exception case
            (
                "failing_tool",
                {"param": "value"},
                None,
                ValueError("Tool execution failed"),
                ValueError,
            ),
        ],
    )
    @patch.object(MCPServerStreamableHttp, "call_tool", new_callable=AsyncMock)
    async def test_call_tool_scenarios(
        self,
        mock_mcp_server_call_tool: AsyncMock,
        tool_name: str,
        arguments: dict | None,
        expected_result: dict | None,
        side_effect: Exception | None,
        expected_exception: type[Exception] | None,
    ) -> None:
        """Test call_tool with various argument and exception scenarios."""
        wrapper = TracedMCPServerWrapper(
            name="Mock MCP Server",
            params={"url": "mock:5051/mcp", "timeout": 120},
        )

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

            # Verify parent's call_tool was called
            mock_mcp_server_call_tool.assert_called_once_with(
                tool_name, arguments, None
            )

            # Verify result is returned
            assert result == expected_result


class TestParseServerUrl:
    """Tests for _parse_server_url (host/port derivation)."""

    @pytest.mark.parametrize(
        "url,expected_host,expected_port",
        [
            ("http://localhost/mcp", "localhost", 80),
            ("https://localhost/mcp", "localhost", 443),
            ("http://example.com", "example.com", 80),
            ("https://mcp.example.com/path", "mcp.example.com", 443),
            ("http://127.0.0.1:5051/mcp", "127.0.0.1", 5051),
            ("https://host:8443/mcp", "host", 8443),
        ],
    )
    def test_parse_server_url_derives_host_and_port(
        self,
        url: str,
        expected_host: str,
        expected_port: int,
    ) -> None:
        """URLs without explicit port use default port (http->80, https->443)."""
        wrapper = TracedMCPServerWrapper(
            name="Test",
            params={"url": url},
        )
        host, port = wrapper._parse_server_url()
        assert host == expected_host
        assert port == expected_port

    def test_parse_server_url_raises_when_no_hostname(self) -> None:
        """Missing hostname in URL raises ValueError."""
        wrapper = TracedMCPServerWrapper(
            name="Test",
            params={"url": "http:///path"},
        )
        with pytest.raises(ValueError, match="Cannot determine host"):
            wrapper._parse_server_url()

    @pytest.mark.parametrize(
        "url",
        [
            "localhost:80",
            "localhost/mcp",
            "example.com:9000/path",
        ],
    )
    def test_parse_server_url_raises_when_scheme_missing(self, url: str) -> None:
        """Scheme-less URLs raise ValueError instead of silently assuming http."""
        wrapper = TracedMCPServerWrapper(
            name="Test",
            params={"url": url},
        )
        with pytest.raises(ValueError, match="must include a scheme"):
            wrapper._parse_server_url()


class TestAssertServerReachable:
    """Tests for _assert_server_reachable (health check)."""

    VALID_URLS: ClassVar[list[tuple[str, str, int]]] = [
        ("http://localhost/mcp", "localhost", 80),
        ("https://localhost/mcp", "localhost", 443),
        ("http://example.com", "example.com", 80),
        ("https://mcp.example.com/path", "mcp.example.com", 443),
        ("http://127.0.0.1:5051/mcp", "127.0.0.1", 5051),
        ("https://host:8443/mcp", "host", 8443),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url,expected_host,expected_port", VALID_URLS)
    @patch.object(TracedMCPServerWrapper, "check_health", new_callable=AsyncMock)
    async def test_healthy_server_does_not_raise(
        self,
        mock_check_health: AsyncMock,
        url: str,
        expected_host: str,
        expected_port: int,
    ) -> None:
        mock_check_health.return_value = True
        wrapper = TracedMCPServerWrapper(name="Test", params={"url": url})
        await wrapper._assert_server_reachable()
        mock_check_health.assert_called_once_with(expected_host, expected_port)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url,expected_host,expected_port", VALID_URLS)
    @patch.object(TracedMCPServerWrapper, "check_health", new_callable=AsyncMock)
    async def test_unhealthy_server_raises_connection_error(
        self,
        mock_check_health: AsyncMock,
        url: str,
        expected_host: str,
        expected_port: int,
    ) -> None:
        mock_check_health.return_value = False
        wrapper = TracedMCPServerWrapper(name="Test", params={"url": url})
        with pytest.raises(ConnectionError, match="not reachable"):
            await wrapper._assert_server_reachable()
        mock_check_health.assert_called_once_with(expected_host, expected_port)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_url",
        [
            "http:///path",
            "",
            "://no-scheme",
            "localhost:80",
            "localhost/mcp",
            "example.com:9000/path",
        ],
    )
    @patch.object(TracedMCPServerWrapper, "check_health", new_callable=AsyncMock)
    async def test_bad_url_raises_connection_error(
        self,
        mock_check_health: AsyncMock,
        bad_url: str,
    ) -> None:
        wrapper = TracedMCPServerWrapper(name="Test", params={"url": bad_url})
        with pytest.raises(ConnectionError, match="not reachable"):
            await wrapper._assert_server_reachable()
        mock_check_health.assert_not_called()


class TestTracedMCPServerWrapperContextManager:
    """Tests for TracedMCPServerWrapper used as async context manager."""

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
    async def test_wrapper_as_context_manager(
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
        """Test TracedMCPServerWrapper as async context manager."""
        wrapper = TracedMCPServerWrapper(
            name=name,
            params=params,
            **kwargs,
        )
        mock_aenter.return_value = wrapper

        if should_raise:
            # Make __aexit__ return False to propagate the exception
            mock_aexit.return_value = False
            assert exception_type is not None
            test_exception = exception_type("Connection failed")

            with pytest.raises(exception_type, match="Connection failed"):
                async with wrapper as server:
                    raise test_exception

            mock_aexit.assert_called_once()
        else:
            async with wrapper as server:
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
        "rasa.builder.telemetry.langfuse_integration.langfuse_compat.is_langfuse_available",
        return_value=False,
    )
    @patch.object(TracedMCPServerWrapper, "__aexit__", new_callable=AsyncMock)
    @patch.object(TracedMCPServerWrapper, "__aenter__", new_callable=AsyncMock)
    async def test_wrapper_context_manager_can_call_tool(
        self,
        mock_aenter: AsyncMock,
        mock_aexit: AsyncMock,
        mock_is_langfuse_available: Mock,
        mock_mcp_server_call_tool: AsyncMock,
    ) -> None:
        """Test that wrapper used as context manager can call tools."""
        mock_mcp_server_call_tool.return_value = {"result": "success"}

        wrapper = TracedMCPServerWrapper(
            name="Tool Server",
            params={"url": "mock:5051/mcp"},
        )
        mock_aenter.return_value = wrapper

        async with wrapper as server:
            result = await server.call_tool("test_tool", {"arg": "val"})

            mock_mcp_server_call_tool.assert_called_once_with(
                "test_tool", {"arg": "val"}, None
            )
            assert result == {"result": "success"}


class TestTracedMCPServerWrapperEnterExitTracing:
    """Tests that __aenter__ and __aexit__ call the correct tracing methods."""

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.emit_lifecycle_event"
    )
    @patch.object(
        MCPServerStreamableHttp,
        "__aenter__",
        new_callable=AsyncMock,
    )
    async def test_enter_emits_created_when_parent_returns(
        self,
        mock_parent_aenter: AsyncMock,
        mock_emit_lifecycle_event: Mock,
        mock_mark_current_span_with_base_exception: Mock,
    ) -> None:
        # Given
        name = "Test MCP Server"
        api_endpoint = "http://localhost:5050/mcp"
        wrapper = TracedMCPServerWrapper(
            name=name,
            params={"url": api_endpoint, "timeout": 120},
        )
        mock_parent_aenter.return_value = wrapper

        # When
        await wrapper.__aenter__()

        # Then
        mock_emit_lifecycle_event.assert_called_once()
        call_kw = mock_emit_lifecycle_event.call_args[1]
        assert call_kw["span_name"] == ("traced_mcp_server.create_connection.created")
        assert call_kw["mcp_server_name"] == name
        assert call_kw["api_endpoint"] == api_endpoint
        assert call_kw.get("level") is None
        mock_mark_current_span_with_base_exception.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.emit_lifecycle_event"
    )
    @patch.object(
        MCPServerStreamableHttp,
        "__aenter__",
        new_callable=AsyncMock,
    )
    async def test_enter_emits_error_when_parent_raises_exception(
        self,
        mock_parent_aenter: AsyncMock,
        mock_emit_lifecycle_event: Mock,
        mock_mark_current_span_with_base_exception: Mock,
    ) -> None:
        # Given
        name = "Test MCP Server"
        api_endpoint = "http://localhost:5050/mcp"
        wrapper = TracedMCPServerWrapper(
            name=name,
            params={"url": api_endpoint, "timeout": 120},
        )
        mock_parent_aenter.side_effect = ConnectionError("Connection refused")

        # When
        with pytest.raises(ConnectionError, match="Connection refused"):
            await wrapper.__aenter__()

        # Then
        mock_emit_lifecycle_event.assert_called_once()
        call_kw = mock_emit_lifecycle_event.call_args[1]
        assert call_kw["span_name"] == ("traced_mcp_server.create_connection.error")
        assert call_kw["level"] == "ERROR"
        mock_mark_current_span_with_base_exception.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(TracedMCPServerWrapper, "check_health", new_callable=AsyncMock)
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.emit_lifecycle_event"
    )
    @patch.object(
        MCPServerStreamableHttp,
        "__aenter__",
        new_callable=AsyncMock,
    )
    async def test_enter_health_probe_base_exception_is_swallowed(
        self,
        mock_parent_aenter: AsyncMock,
        mock_emit_lifecycle_event: Mock,
        mock_check_health: AsyncMock,
    ) -> None:
        """Health probe BaseException is swallowed, original error re-raised."""
        wrapper = TracedMCPServerWrapper(
            name="Test MCP Server",
            params={"url": "http://localhost:5050/mcp", "timeout": 120},
        )
        original_error = asyncio.CancelledError()
        mock_parent_aenter.side_effect = original_error
        mock_check_health.side_effect = asyncio.CancelledError("probe cancelled")

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await wrapper.__aenter__()

        assert exc_info.value is original_error

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_healthy", [True, False])
    @patch.object(TracedMCPServerWrapper, "check_health", new_callable=AsyncMock)
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.trace_health"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.emit_lifecycle_event"
    )
    @patch.object(
        MCPServerStreamableHttp,
        "__aenter__",
        new_callable=AsyncMock,
    )
    async def test_enter_diagnoses_health_and_reraises_base_exception(
        self,
        mock_parent_aenter: AsyncMock,
        mock_emit_lifecycle_event: Mock,
        mock_trace_health: Mock,
        mock_check_health: AsyncMock,
        is_healthy: bool,
    ) -> None:
        """CancelledError is always re-raised; health is traced regardless."""
        # Given
        wrapper = TracedMCPServerWrapper(
            name="Test MCP Server",
            params={"url": "http://localhost:5050/mcp", "timeout": 120},
        )
        mock_parent_aenter.side_effect = asyncio.CancelledError()
        mock_check_health.return_value = is_healthy

        # When
        with pytest.raises(asyncio.CancelledError):
            await wrapper.__aenter__()

        # Then – health was probed and traced
        mock_check_health.assert_called_once_with("localhost", 5050)
        mock_trace_health.assert_called_once_with(
            is_healthy=is_healthy,
            host="localhost",
            port=5050,
            raw_url="http://localhost:5050/mcp",
        )

        # Then – lifecycle event was traced
        mock_emit_lifecycle_event.assert_called()
        call_kw = mock_emit_lifecycle_event.call_args[1]
        assert call_kw["span_name"] == ("traced_mcp_server.create_connection.cancelled")
        assert call_kw["level"] == "ERROR"

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.emit_lifecycle_event"
    )
    @patch.object(
        MCPServerStreamableHttp,
        "__aexit__",
        new_callable=AsyncMock,
    )
    async def test_exit_emits_closed_when_normal_exit(
        self,
        mock_parent_aexit: AsyncMock,
        mock_emit_lifecycle_event: Mock,
        mock_mark_current_span_with_base_exception: Mock,
    ) -> None:
        # Given
        name = "Test MCP Server"
        api_endpoint = "http://localhost:5050/mcp"
        wrapper = TracedMCPServerWrapper(
            name=name,
            params={"url": api_endpoint, "timeout": 120},
        )
        mock_parent_aexit.return_value = None
        exc_type, exc_val, exc_tb = None, None, None

        # When
        await wrapper.__aexit__(exc_type, exc_val, exc_tb)

        # Then
        mock_emit_lifecycle_event.assert_called_once()
        call_kw = mock_emit_lifecycle_event.call_args[1]
        assert call_kw["span_name"] == ("traced_mcp_server.close_connection.closed")
        assert call_kw["mcp_server_name"] == name
        assert call_kw["metadata"].get("reason") == "normal"
        assert call_kw.get("level") is None
        mock_mark_current_span_with_base_exception.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.emit_lifecycle_event"
    )
    @patch.object(
        MCPServerStreamableHttp,
        "__aexit__",
        new_callable=AsyncMock,
    )
    async def test_exit_emits_closed_on_error_when_context_raised(
        self,
        mock_parent_aexit: AsyncMock,
        mock_emit_lifecycle_event: Mock,
        mock_mark_current_span_with_base_exception: Mock,
    ) -> None:
        # Given
        name = "Test MCP Server"
        api_endpoint = "http://localhost:5050/mcp"
        wrapper = TracedMCPServerWrapper(
            name=name,
            params={"url": api_endpoint, "timeout": 120},
        )
        mock_parent_aexit.return_value = None
        exc_val = ValueError("context body raised")
        exc_type, exc_tb = type(exc_val), None

        # When
        await wrapper.__aexit__(exc_type, exc_val, exc_tb)

        # Then
        mock_emit_lifecycle_event.assert_called_once()
        call_kw = mock_emit_lifecycle_event.call_args[1]
        assert call_kw["span_name"] == (
            "traced_mcp_server.close_connection.closed_on_error"
        )
        assert call_kw["level"] == "ERROR"
        mock_mark_current_span_with_base_exception.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.emit_lifecycle_event"
    )
    @patch.object(
        MCPServerStreamableHttp,
        "__aexit__",
        new_callable=AsyncMock,
    )
    async def test_exit_emits_cleanup_error_when_parent_raises_exception(
        self,
        mock_parent_aexit: AsyncMock,
        mock_emit_lifecycle_event: Mock,
        mock_mark_current_span_with_base_exception: Mock,
    ) -> None:
        # Given
        name = "Test MCP Server"
        api_endpoint = "http://localhost:5050/mcp"
        wrapper = TracedMCPServerWrapper(
            name=name,
            params={"url": api_endpoint, "timeout": 120},
        )
        mock_parent_aexit.side_effect = RuntimeError("cleanup failed")
        exc_type, exc_val, exc_tb = None, None, None

        # When
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await wrapper.__aexit__(exc_type, exc_val, exc_tb)

        # Then
        mock_emit_lifecycle_event.assert_called_once()
        call_kw = mock_emit_lifecycle_event.call_args[1]
        assert call_kw["span_name"] == (
            "traced_mcp_server.close_connection.cleanup_error"
        )
        assert call_kw["level"] == "ERROR"
        mock_mark_current_span_with_base_exception.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.telemetry.langfuse_integration.traced_mcp_server."
        "MCPLifecycleLangfuseTelemetry.emit_lifecycle_event"
    )
    @patch.object(
        MCPServerStreamableHttp,
        "__aexit__",
        new_callable=AsyncMock,
    )
    async def test_exit_traces_lifecycle_when_parent_raises_base_exception(
        self,
        mock_parent_aexit: AsyncMock,
        mock_emit_lifecycle_event: Mock,
    ) -> None:
        # Given
        name = "Test MCP Server"
        api_endpoint = "http://localhost:5050/mcp"
        wrapper = TracedMCPServerWrapper(
            name=name,
            params={"url": api_endpoint, "timeout": 120},
        )
        mock_parent_aexit.side_effect = asyncio.CancelledError()
        exc_type, exc_val, exc_tb = None, None, None

        # When
        with pytest.raises(asyncio.CancelledError):
            await wrapper.__aexit__(exc_type, exc_val, exc_tb)

        # Then
        mock_emit_lifecycle_event.assert_called_once()
        call_kw = mock_emit_lifecycle_event.call_args[1]
        assert call_kw["span_name"] == ("traced_mcp_server.close_connection.cancelled")
        assert call_kw["level"] == "ERROR"
