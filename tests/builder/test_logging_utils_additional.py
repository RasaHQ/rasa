"""Additional tests for logging utilities - covering utility functions."""

from typing import Any, MutableMapping
from unittest.mock import MagicMock, patch

from rasa.builder.logging_utils import (
    _sanitize_headers,
    _try_to_extract_event_dict_json,
    attach_request_id_processor,
    capture_exception_with_context,
    clear_recent_logs,
    collecting_logs_processor,
    ensure_correlation_id_on_request,
    extract_request_context,
    get_log_count,
    get_recent_logs,
    log_request_end,
    log_request_start,
)


class TestRecentLogsBuffer:
    """Test recent logs buffer functions."""

    def test_get_recent_logs_returns_string(self) -> None:
        """Test that get_recent_logs returns a string."""
        result = get_recent_logs()
        assert isinstance(result, str)

    def test_clear_recent_logs_empties_buffer(self) -> None:
        """Test that clear_recent_logs empties the buffer."""
        # Add some logs first using the processor
        event_dict: MutableMapping[str, Any] = {
            "event": "test.event",
            "event_info": "Test message",
        }
        collecting_logs_processor(None, "info", event_dict)

        # Clear logs
        clear_recent_logs()

        # Buffer should be empty
        assert get_log_count() == 0
        assert get_recent_logs() == ""

    def test_get_log_count_returns_int(self) -> None:
        """Test that get_log_count returns an integer."""
        result = get_log_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_collecting_logs_processor_adds_to_buffer(self) -> None:
        """Test that collecting_logs_processor adds logs to buffer."""
        clear_recent_logs()

        event_dict: MutableMapping[str, Any] = {
            "event": "test.processor.event",
            "event_info": "Test processor message",
        }

        # Process at info level (not debug)
        collecting_logs_processor(None, "info", event_dict)

        # Should have one log entry
        assert get_log_count() >= 1
        logs = get_recent_logs()
        assert "Test processor message" in logs

    def test_collecting_logs_processor_skips_debug(self) -> None:
        """Test that debug logs are skipped by collector."""
        clear_recent_logs()

        event_dict: MutableMapping[str, Any] = {
            "event": "test.debug.event",
            "event_info": "Debug message should be skipped",
        }

        initial_count = get_log_count()

        # Process at debug level
        collecting_logs_processor(None, "debug", event_dict)

        # Count should not have increased
        assert get_log_count() == initial_count

    def test_collecting_logs_processor_includes_event_dict_json(self) -> None:
        """Test that event dict is included in log entry."""
        clear_recent_logs()

        event_dict: MutableMapping[str, Any] = {
            "event": "test.json.event",
            "event_info": "Event with extra data",
            "extra_key": "extra_value",
        }

        collecting_logs_processor(None, "error", event_dict)

        logs = get_recent_logs()
        assert "extra_key" in logs
        assert "extra_value" in logs


class TestTryToExtractEventDictJson:
    """Test _try_to_extract_event_dict_json function."""

    def test_empty_dict(self) -> None:
        """Test with empty dict."""
        result = _try_to_extract_event_dict_json({})
        assert result == ""

    def test_ignored_keys_excluded(self) -> None:
        """Test that ignored keys are excluded."""
        event_dict: MutableMapping[str, Any] = {
            "event": "should be ignored",
            "event_info": "also ignored",
            "level": "info",
            "correlation_id": "abc123",
            "timestamp": "2024-01-01",
            "custom_key": "should be included",
        }

        result = _try_to_extract_event_dict_json(event_dict)

        assert "custom_key" in result
        assert "should be included" in result
        # Ignored keys should not appear as keys
        assert "event_info=" not in result
        assert "level=" not in result

    def test_long_values_truncated(self) -> None:
        """Test that long values are truncated to 100 chars."""
        event_dict: MutableMapping[str, Any] = {
            "long_value": "x" * 200,  # 200 chars
        }

        result = _try_to_extract_event_dict_json(event_dict)

        # Should be truncated with "..."
        assert "..." in result
        # The full 200 chars should not be present
        assert "x" * 200 not in result

    def test_multiple_keys(self) -> None:
        """Test with multiple keys."""
        event_dict: MutableMapping[str, Any] = {
            "key1": "value1",
            "key2": "value2",
            "key3": 42,
        }

        result = _try_to_extract_event_dict_json(event_dict)

        assert "key1" in result
        assert "value1" in result
        assert "key2" in result
        assert "value2" in result
        assert "key3" in result
        assert "42" in result

    def test_non_serializable_value(self) -> None:
        """Test with non-JSON-serializable values (uses default=str)."""
        event_dict: MutableMapping[str, Any] = {
            "object": object(),  # Not directly JSON serializable
        }

        # Should not raise, uses default=str
        result = _try_to_extract_event_dict_json(event_dict)
        assert "object" in result


class TestSanitizeHeaders:
    """Test _sanitize_headers function."""

    def test_user_agent_preserved(self) -> None:
        """Test that User-Agent header is preserved."""
        headers = {"User-Agent": "Mozilla/5.0", "Other": "value"}
        result = _sanitize_headers(headers)

        assert "user-agent" in result
        assert result["user-agent"] == "Mozilla/5.0"

    def test_user_id_preserved(self) -> None:
        """Test that X-User-Id header is preserved."""
        from rasa.builder.auth import HEADER_USER_ID

        headers = {HEADER_USER_ID: "user123", "Other": "value"}
        result = _sanitize_headers(headers)

        assert HEADER_USER_ID in result
        assert result[HEADER_USER_ID] == "user123"

    def test_user_id_preserved_case_insensitive(self) -> None:
        """Test that x-user-id header is preserved regardless of case."""
        from rasa.builder.auth import HEADER_USER_ID

        # Test with lowercase input
        headers = {"x-user-id": "user456", "Other": "value"}
        result = _sanitize_headers(headers)

        assert HEADER_USER_ID in result
        assert result[HEADER_USER_ID] == "user456"

    def test_authorization_redacted(self) -> None:
        """Test that Authorization header is redacted."""
        headers = {"Authorization": "Bearer secret-token-12345"}
        result = _sanitize_headers(headers)

        assert "authorization" in result
        assert "Bearer" in result["authorization"]
        assert "secret-token-12345" not in result["authorization"]
        assert "<redacted>" in result["authorization"]

    def test_authorization_redacted_basic_auth(self) -> None:
        """Test that Basic auth is redacted."""
        headers = {"Authorization": "Basic dXNlcjpwYXNzd29yZA=="}
        result = _sanitize_headers(headers)

        assert "authorization" in result
        assert "Basic" in result["authorization"]
        assert "dXNlcjpwYXNzd29yZA==" not in result["authorization"]

    def test_case_insensitive_header_matching(self) -> None:
        """Test that header matching is case-insensitive."""
        headers = {
            "user-agent": "Test Agent",
            "AUTHORIZATION": "Bearer token",
        }
        result = _sanitize_headers(headers)

        assert "user-agent" in result
        assert "authorization" in result

    def test_other_headers_excluded(self) -> None:
        """Test that other headers are not included."""
        headers = {
            "Content-Type": "application/json",
            "X-Custom-Header": "custom-value",
            "Cookie": "session=abc123",
        }
        result = _sanitize_headers(headers)

        # These should not be in the result
        assert "content-type" not in result
        assert "x-custom-header" not in result
        assert "cookie" not in result

    def test_empty_headers(self) -> None:
        """Test with empty headers dict."""
        result = _sanitize_headers({})
        assert result == {}

    def test_authorization_present_but_empty(self) -> None:
        """Test authorization header when empty."""
        headers = {"Authorization": ""}
        result = _sanitize_headers(headers)

        assert "authorization" in result
        assert result["authorization"] == "present"


class TestEnsureCorrelationIdOnRequest:
    """Test ensure_correlation_id_on_request function."""

    def test_creates_correlation_id_when_missing(self) -> None:
        """Test that correlation ID is created when missing."""
        mock_request = MagicMock()
        mock_request.ctx = MagicMock(spec=[])  # No correlation_id attribute

        result = ensure_correlation_id_on_request(mock_request)

        assert result is not None
        assert len(result) == 32  # uuid4 hex is 32 chars
        assert mock_request.ctx.correlation_id == result

    def test_preserves_existing_correlation_id(self) -> None:
        """Test that existing correlation ID is preserved."""
        mock_request = MagicMock()
        mock_request.ctx.correlation_id = "existing-id-123"

        result = ensure_correlation_id_on_request(mock_request)

        assert result == "existing-id-123"

    def test_creates_new_id_when_empty_string(self) -> None:
        """Test that new ID is created when existing is empty string."""
        mock_request = MagicMock()
        mock_request.ctx.correlation_id = ""

        result = ensure_correlation_id_on_request(mock_request)

        assert result != ""
        assert len(result) == 32


class TestAttachRequestIdProcessor:
    """Test attach_request_id_processor function."""

    def test_returns_event_dict_unchanged_without_request(self) -> None:
        """Test that event_dict is returned unchanged when no request context."""
        event_dict: MutableMapping[str, Any] = {
            "event": "test.event",
            "data": "some data",
        }

        # Mock Request.get_current to raise (no request context)
        with patch(
            "rasa.builder.logging_utils.Request.get_current",
            side_effect=Exception("No request context"),
        ):
            result = attach_request_id_processor(None, "info", event_dict)

        assert result == event_dict
        assert "correlation_id" not in result

    def test_attaches_correlation_id_from_request(self) -> None:
        """Test that correlation_id is attached from request."""
        event_dict: MutableMapping[str, Any] = {
            "event": "test.event",
        }

        mock_request = MagicMock()
        mock_request.ctx.correlation_id = "test-correlation-123"

        with patch(
            "rasa.builder.logging_utils.Request.get_current",
            return_value=mock_request,
        ):
            result = attach_request_id_processor(None, "info", event_dict)

        assert result["correlation_id"] == "test-correlation-123"


class TestExtractRequestContext:
    """Test extract_request_context function."""

    def test_returns_empty_dict_without_request(self) -> None:
        """Test that empty dict is returned when no request context."""
        with patch(
            "rasa.builder.logging_utils.Request.get_current",
            side_effect=Exception("No request"),
        ):
            result = extract_request_context()

        assert result == {}

    def test_extracts_request_fields(self) -> None:
        """Test that request fields are extracted."""
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.path = "/api/test"
        mock_request.args = {"key": "value"}
        mock_request.json = {"data": "json"}
        mock_request.content_length = 100
        mock_request.remote_addr = "127.0.0.1"
        mock_request.headers = {"User-Agent": "Test"}
        mock_request.ctx.correlation_id = "test-id"

        with patch(
            "rasa.builder.logging_utils.Request.get_current",
            return_value=mock_request,
        ):
            result = extract_request_context()

        assert result["method"] == "POST"
        assert result["path"] == "/api/test"
        assert result["has_json"] is True
        assert result["content_length"] == 100


class TestLogRequestStartEnd:
    """Test log_request_start and log_request_end functions."""

    def test_log_request_start_returns_float(self) -> None:
        """Test that log_request_start returns a float timestamp."""
        mock_request = MagicMock()
        mock_request.ctx.correlation_id = "test-id"
        mock_request.method = "GET"
        mock_request.path = "/test"
        mock_request.headers = {}

        with patch(
            "rasa.builder.logging_utils.Request.get_current",
            return_value=mock_request,
        ):
            result = log_request_start(mock_request)

        assert isinstance(result, float)
        assert result > 0

    def test_log_request_end_calculates_latency(self) -> None:
        """Test that log_request_end calculates latency correctly."""
        mock_request = MagicMock()
        mock_request.ctx.correlation_id = "test-id"
        mock_request.method = "GET"
        mock_request.path = "/test"

        mock_response = MagicMock()
        mock_response.status = 200

        import time

        start_time = time.perf_counter()

        # Should not raise any exceptions
        log_request_end(mock_request, mock_response, start_time)


class TestCaptureExceptionWithContext:
    """Test capture_exception_with_context function."""

    def test_captures_exception_with_extra_and_tags(self) -> None:
        """Test capturing exception with extra context and tags."""
        exc = ValueError("Test error")

        # Mock sentry_sdk to avoid actual calls
        with patch("rasa.builder.logging_utils.sentry_sdk") as mock_sentry:
            mock_scope = MagicMock()
            mock_sentry.configure_scope.return_value.__enter__ = MagicMock(
                return_value=mock_scope
            )
            mock_sentry.configure_scope.return_value.__exit__ = MagicMock(
                return_value=None
            )

            with patch(
                "rasa.builder.logging_utils.extract_request_context",
                return_value={"path": "/test", "method": "GET"},
            ):
                capture_exception_with_context(
                    exc,
                    event_id="test.error",
                    extra={"custom": "data"},
                    tags={"env": "test"},
                )

            # Verify sentry was called
            mock_sentry.capture_exception.assert_called_once_with(exc)

    def test_handles_sentry_failure_gracefully(self) -> None:
        """Test that Sentry failures don't crash the function."""
        exc = ValueError("Test error")

        with patch(
            "rasa.builder.logging_utils.sentry_sdk.configure_scope",
            side_effect=Exception("Sentry failed"),
        ):
            with patch(
                "rasa.builder.logging_utils.extract_request_context",
                return_value={},
            ):
                # Should not raise
                capture_exception_with_context(exc, event_id="test.error")
