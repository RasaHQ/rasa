"""Unit tests for MCP lifecycle Langfuse telemetry."""

from unittest.mock import Mock, patch

import pytest

from rasa.builder.logging_utils import ExceptionFields
from rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry import (  # noqa: E501
    MCPLifecycleLangfuseTelemetry,
)


def _make_langfuse_context_mock(mock_client: Mock):
    """Build a mock for with_langfuse() that yields an object with get_client()."""
    mock_lf = Mock()
    mock_lf.get_client.return_value = mock_client
    mock_cm = Mock()
    mock_cm.__enter__ = Mock(return_value=mock_lf)
    mock_cm.__exit__ = Mock(return_value=False)
    return mock_cm


class TestMCPLifecycleLangfuseTelemetryTraceHealth:
    """Tests for MCPLifecycleLangfuseTelemetry.trace_health."""

    @pytest.mark.parametrize(
        "is_healthy,host,port,expected_status",
        [
            (True, "localhost", 5050, "running"),
            (False, "localhost", 5050, "not_running"),
            (True, "127.0.0.1", 8080, "running"),
            (False, "mcp.example.com", 443, "not_running"),
        ],
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_trace_health_emits_span_with_status(
        self,
        mock_with_langfuse: Mock,
        is_healthy: bool,
        host: str,
        port: int,
        expected_status: str,
    ) -> None:
        # Given
        mock_client = Mock()
        mock_span = Mock()
        mock_client.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_client.start_as_current_span.return_value.__exit__ = Mock(
            return_value=False
        )
        mock_with_langfuse.return_value = _make_langfuse_context_mock(mock_client)

        # When
        MCPLifecycleLangfuseTelemetry.trace_health(
            is_healthy=is_healthy,
            host=host,
            port=port,
            raw_url=f"{host}:{port}",
        )

        # Then
        mock_client.start_as_current_span.assert_called_once_with(
            name="mcp_lifecycle.health_check"
        )
        mock_span.update.assert_called_once_with(
            output={"mcp_server_health": expected_status},
            input={"host": host, "port": port, "raw_url": f"{host}:{port}"},
        )

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_trace_health_when_langfuse_yields_none_returns_early(
        self, mock_with_langfuse: Mock
    ) -> None:
        # Given
        mock_with_langfuse.return_value.__enter__ = Mock(return_value=None)
        mock_with_langfuse.return_value.__exit__ = Mock(return_value=False)

        # When
        MCPLifecycleLangfuseTelemetry.trace_health(
            is_healthy=True,
            host="localhost",
            port=5050,
            raw_url="localhost:5050",
        )

        # Then
        mock_with_langfuse.assert_called_once()
        mock_with_langfuse.return_value.__exit__.assert_called_once()

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.structlogger"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_trace_health_catches_exception_and_logs(
        self, mock_with_langfuse: Mock, mock_structlogger: Mock
    ) -> None:
        # Given
        mock_with_langfuse.return_value = _make_langfuse_context_mock(Mock())
        mock_with_langfuse.return_value.__enter__.side_effect = RuntimeError(
            "langfuse broken"
        )

        # When
        MCPLifecycleLangfuseTelemetry.trace_health(
            is_healthy=True,
            host="localhost",
            port=5050,
            raw_url="localhost:5050",
        )

        # Then
        mock_structlogger.debug.assert_called_once()
        assert "mcp_lifecycle_telemetry.trace_health.failed" in str(
            mock_structlogger.debug.call_args
        )
        assert "error" in mock_structlogger.debug.call_args[1]


class TestMCPLifecycleLangfuseTelemetryEmitLifecycleEvent:
    """Tests for MCPLifecycleLangfuseTelemetry.emit_lifecycle_event."""

    @pytest.mark.parametrize(
        "span_name,mcp_server_name,api_endpoint,metadata,output,level",
        [
            (
                "mcp_lifecycle.connected",
                "fileserver",
                "http://localhost:5050/mcp",
                {"transport": "stdio"},
                None,
                None,
            ),
            (
                "mcp_lifecycle.closed",
                "runner",
                "http://127.0.0.1:8080/mcp",
                {},
                {"reason": "idle"},
                None,
            ),
            (
                "mcp_lifecycle.error",
                "tools",
                "https://mcp.example.com/sse",
                {"retry": True},
                {"error": "Connection refused"},
                "ERROR",
            ),
        ],
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_emit_lifecycle_event_emits_span(
        self,
        mock_with_langfuse: Mock,
        span_name: str,
        mcp_server_name: str,
        api_endpoint: str,
        metadata: dict,
        output: dict | None,
        level: str | None,
    ) -> None:
        # Given
        mock_client = Mock()
        mock_span = Mock()
        mock_client.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_client.start_as_current_span.return_value.__exit__ = Mock(
            return_value=False
        )
        mock_with_langfuse.return_value = _make_langfuse_context_mock(mock_client)

        # When
        MCPLifecycleLangfuseTelemetry.emit_lifecycle_event(
            span_name=span_name,
            mcp_server_name=mcp_server_name,
            api_endpoint=api_endpoint,
            metadata=metadata,
            output=output,
            level=level,
        )

        # Then
        mock_client.start_as_current_span.assert_called_once_with(name=span_name)
        mock_span.update.assert_called_once_with(
            level=level,
            input={
                "mcp_server_name": mcp_server_name,
                "api_endpoint": api_endpoint,
            },
            output=output,
            metadata=metadata,
        )

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_emit_lifecycle_event_when_langfuse_none_returns_early(
        self, mock_with_langfuse: Mock
    ) -> None:
        # Given
        mock_with_langfuse.return_value.__enter__ = Mock(return_value=None)
        mock_with_langfuse.return_value.__exit__ = Mock(return_value=False)

        # When
        MCPLifecycleLangfuseTelemetry.emit_lifecycle_event(
            span_name="mcp_lifecycle.connected",
            mcp_server_name="test",
            api_endpoint="http://localhost/mcp",
            metadata={},
        )

        # Then
        mock_with_langfuse.assert_called_once()

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.structlogger"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_emit_lifecycle_event_catches_exception_and_logs(
        self, mock_with_langfuse: Mock, mock_structlogger: Mock
    ) -> None:
        # Given
        mock_with_langfuse.return_value = _make_langfuse_context_mock(Mock())
        mock_with_langfuse.return_value.__enter__.side_effect = OSError("network error")

        # When
        MCPLifecycleLangfuseTelemetry.emit_lifecycle_event(
            span_name="mcp_lifecycle.connected",
            mcp_server_name="test",
            api_endpoint="http://localhost/mcp",
            metadata={},
        )

        # Then
        mock_structlogger.debug.assert_called_once()
        assert "emit_lifecycle_event.failed" in str(mock_structlogger.debug.call_args)


class TestMCPLifecycleLangfuseTelemetryUpdateToolCallSpan:
    """Tests for MCPLifecycleLangfuseTelemetry.update_tool_call_span."""

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_update_tool_call_span_updates_current_span(
        self, mock_with_langfuse: Mock
    ) -> None:
        # Given
        mock_client = Mock()
        mock_with_langfuse.return_value = _make_langfuse_context_mock(mock_client)
        tool_name = "test_tool"
        mcp_server_name = "test_mcp_server"

        # When
        MCPLifecycleLangfuseTelemetry.update_tool_call_span(
            tool_name=tool_name,
            mcp_server_name=mcp_server_name,
        )

        # Then
        mock_client.update_current_span.assert_called_once_with(
            name=f"mcp_tool.{tool_name}",
            metadata={
                "mcp_server": mcp_server_name,
                "tool_type": "mcp",
            },
        )

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_update_tool_call_span_when_langfuse_none_returns_early(
        self, mock_with_langfuse: Mock
    ) -> None:
        # Given
        mock_with_langfuse.return_value.__enter__ = Mock(return_value=None)
        mock_with_langfuse.return_value.__exit__ = Mock(return_value=False)

        # When
        MCPLifecycleLangfuseTelemetry.update_tool_call_span(
            tool_name="read_file",
            mcp_server_name="fileserver",
        )

        # Then
        mock_with_langfuse.assert_called_once()

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.structlogger"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_update_tool_call_span_catches_exception_and_logs(
        self, mock_with_langfuse: Mock, mock_structlogger: Mock
    ) -> None:
        # Given
        mock_with_langfuse.return_value = _make_langfuse_context_mock(Mock())
        mock_with_langfuse.return_value.__enter__.side_effect = ValueError("no client")

        # When
        MCPLifecycleLangfuseTelemetry.update_tool_call_span(
            tool_name="read_file",
            mcp_server_name="fileserver",
        )

        # Then
        mock_structlogger.debug.assert_called_once()
        assert "update_tool_call_span.failed" in str(mock_structlogger.debug.call_args)


class TestMCPLifecycleLangfuseTelemetryMarkCurrentSpanWithBaseException:
    """Tests for MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception."""

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("invalid value"),
            RuntimeError("something went wrong"),
            KeyError("missing key"),
        ],
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_mark_current_span_with_base_exception_updates_span(
        self,
        mock_with_langfuse: Mock,
        error: BaseException,
    ) -> None:
        # Given
        mock_client = Mock()
        mock_with_langfuse.return_value = _make_langfuse_context_mock(mock_client)
        expected_output = ExceptionFields.from_exception(error).to_dict()

        # When
        MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception(error)

        # Then
        mock_client.update_current_span.assert_called_once_with(
            level="ERROR",
            output=expected_output,
        )

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_mark_current_span_with_base_exception_when_langfuse_none_returns_early(
        self, mock_with_langfuse: Mock
    ) -> None:
        # Given
        mock_with_langfuse.return_value.__enter__ = Mock(return_value=None)
        mock_with_langfuse.return_value.__exit__ = Mock(return_value=False)

        # When
        MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception(
            ValueError("test")
        )

        # Then
        mock_with_langfuse.assert_called_once()

    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.structlogger"
    )
    @patch(
        "rasa.builder.telemetry.langfuse_integration.mcp_lifecycle_langfuse_telemetry.with_langfuse"
    )
    def test_mark_current_span_with_base_exception_catches_exception_and_logs(
        self, mock_with_langfuse: Mock, mock_structlogger: Mock
    ) -> None:
        # Given
        mock_with_langfuse.return_value = _make_langfuse_context_mock(Mock())
        mock_with_langfuse.return_value.__enter__.side_effect = OSError("langfuse down")

        # When
        MCPLifecycleLangfuseTelemetry.mark_current_span_with_base_exception(
            ValueError("original error")
        )

        # Then
        mock_structlogger.debug.assert_called_once()
        assert "mark_current_span_with_base_exception.failed" in str(
            mock_structlogger.debug.call_args
        )
