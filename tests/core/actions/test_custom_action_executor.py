from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.custom_action_executor import (
    RetryCustomActionExecutor,
    dispatch_stream_chunk,
    warn_on_duplicate_streamed_responses,
)
from rasa.core.actions.grpc_custom_action_executor import GRPCCustomActionExecutor
from rasa.core.actions.http_custom_action_executor import HTTPCustomActionExecutor
from rasa.core.channels import OutputChannel
from rasa.core.channels.channel import CollectingOutputChannel
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig


@pytest.mark.asyncio
async def test_retry_executor_handles_successful_first_call(
    mock_endpoint: EndpointConfig,
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    response_data = {"events": [], "responses": []}
    mock_endpoint.request = AsyncMock(return_value=response_data)

    http_executor = HTTPCustomActionExecutor("test_action", mock_endpoint)
    retry_executor = RetryCustomActionExecutor(http_executor)

    result = await retry_executor.run(tracker, domain, include_domain=False)

    assert result == response_data
    assert mock_endpoint.request.call_count == 1


@pytest.mark.asyncio
async def test_retry_executor_retries_on_missing_domain(
    mock_endpoint: EndpointConfig,
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    response_data = {"events": [], "responses": []}

    mock_endpoint.request = AsyncMock(
        side_effect=[
            {"missing_domain": True},
            response_data,
        ]
    )

    http_executor = HTTPCustomActionExecutor("test_action", mock_endpoint)
    retry_executor = RetryCustomActionExecutor(http_executor)

    result = await retry_executor.run(tracker, domain, include_domain=False)

    assert result == response_data
    assert mock_endpoint.request.call_count == 2

    # Verify first call was without domain, second with domain
    calls = mock_endpoint.request.call_args_list
    assert "domain" not in calls[0][1]["json"]  # First call should exclude domain
    assert "domain" in calls[1][1]["json"]  # Second call should include domain


@pytest.mark.asyncio
async def test_retry_executor_raises_after_two_missing_domain_responses(
    mock_endpoint: EndpointConfig,
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    mock_endpoint.request = AsyncMock(return_value={"missing_domain": True})

    http_executor = HTTPCustomActionExecutor("test_action", mock_endpoint)
    retry_executor = RetryCustomActionExecutor(http_executor)

    # Execute the action - should raise after second attempt
    with pytest.raises(DomainNotFound):
        await retry_executor.run(tracker, domain, include_domain=False)

    # Verify both attempts were made
    assert mock_endpoint.request.call_count == 2


# ---------------------------------------------------------------------------
# RetryCustomActionExecutor.run_streaming() tests
# ---------------------------------------------------------------------------


class _FakeOutputChannel(OutputChannel):
    """Minimal channel that records streaming calls for assertions."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        super().__init__()

    @property
    def supports_streaming(self) -> bool:
        return True

    async def send_response_chunk_start(self, recipient_id: str, **kw: Any) -> None:
        pass

    async def send_response_chunk(
        self, recipient_id: str, chunk: str, **kw: Any
    ) -> None:
        self.chunks.append(chunk)

    async def send_response_chunk_end(self, recipient_id: str, **kw: Any) -> None:
        pass


class _StreamingInnerExecutor:
    """Fake inner executor that exposes run_streaming()."""

    def __init__(self, result: dict, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.call_count = 0

    async def run_streaming(
        self,
        tracker: Any,
        domain: Any,
        output_channel: Any,
        include_domain: bool = False,
    ) -> dict:
        self.call_count += 1
        if self._raises:
            raise self._raises
        return self._result

    async def run_with_result(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _NonStreamingInnerExecutor:
    """Fake inner executor that does NOT expose run_streaming()."""

    async def run_with_result(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_retry_executor_run_streaming_delegates_to_inner(
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    expected = {"events": [], "responses": []}
    inner = _StreamingInnerExecutor(result=expected)
    retry = RetryCustomActionExecutor(inner)  # type: ignore[arg-type]

    channel = _FakeOutputChannel()
    result = await retry.run_streaming(
        tracker=tracker, domain=domain, output_channel=channel
    )

    assert result == expected
    assert inner.call_count == 1


@pytest.mark.asyncio
async def test_retry_executor_run_streaming_retries_on_domain_not_found(
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    expected = {"events": [], "responses": []}
    inner = _StreamingInnerExecutor(result=expected, raises=DomainNotFound())
    # Override raises to only fire on first call
    inner._raises = None

    async def run_streaming_side_effect(
        tracker: Any,
        domain: Any,
        output_channel: Any,
        include_domain: bool = False,
    ) -> dict:
        inner.call_count += 1
        if inner.call_count == 1:
            raise DomainNotFound()
        return expected

    inner.run_streaming = run_streaming_side_effect  # type: ignore[method-assign]

    retry = RetryCustomActionExecutor(inner)  # type: ignore[arg-type]
    channel = _FakeOutputChannel()
    result = await retry.run_streaming(
        tracker=tracker, domain=domain, output_channel=channel
    )

    assert result == expected
    assert inner.call_count == 2


def test_non_streaming_executor_has_supports_streaming_false() -> None:
    """Executors that don't override run_streaming must have supports_streaming=False.

    RemoteAction._can_stream checks executor.supports_streaming (after looking
    through the RetryCustomActionExecutor wrapper).  The flag must default to
    False on the base class so that executors which inherit the
    NotImplementedError stub do not accidentally activate the streaming path.
    """
    inner = _NonStreamingInnerExecutor()
    assert not getattr(inner, "supports_streaming", False)


# ---------------------------------------------------------------------------
# dispatch_stream_chunk() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_stream_chunk_text_only_calls_send_response_chunk() -> None:
    """A plain-text payload is forwarded via send_response_chunk."""
    channel = AsyncMock()
    await dispatch_stream_chunk(channel, "user1", {"text": "hello"})
    channel.send_response_chunk.assert_awaited_once_with("user1", "hello")
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_stream_chunk_empty_payload_sends_empty_string() -> None:
    """An empty payload results in send_response_chunk called with an empty string."""
    channel = AsyncMock()
    await dispatch_stream_chunk(channel, "user1", {})
    channel.send_response_chunk.assert_awaited_once_with("user1", "")
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_stream_chunk_buttons_calls_send_response() -> None:
    """A payload containing 'buttons' is routed to send_response."""
    channel = AsyncMock()
    payload: Dict[str, Any] = {
        "text": "Choose:",
        "buttons": [{"title": "Yes", "payload": "/yes"}],
    }
    await dispatch_stream_chunk(channel, "user1", payload)
    channel.send_response.assert_awaited_once_with("user1", payload)
    channel.send_response_chunk.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_stream_chunk_image_calls_send_response() -> None:
    """A payload containing 'image' is routed to send_response."""
    channel = AsyncMock()
    payload: Dict[str, Any] = {"image": "https://example.com/img.png"}
    await dispatch_stream_chunk(channel, "user1", payload)
    channel.send_response.assert_awaited_once_with("user1", payload)
    channel.send_response_chunk.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_stream_chunk_custom_calls_send_response() -> None:
    """A payload containing 'custom' is routed to send_response."""
    channel = AsyncMock()
    payload: Dict[str, Any] = {"custom": {"key": "value"}}
    await dispatch_stream_chunk(channel, "user1", payload)
    channel.send_response.assert_awaited_once_with("user1", payload)
    channel.send_response_chunk.assert_not_awaited()


# ---------------------------------------------------------------------------
# warn_on_duplicate_streamed_responses() tests
# ---------------------------------------------------------------------------


def test_warn_on_duplicate_streamed_responses_logs_warning_on_duplicate() -> None:
    """A warning is emitted when a final response was already streamed."""
    response: Dict[str, Any] = {"text": "Hello world"}
    with patch(
        "rasa.core.actions.custom_action_executor.structlogger"
    ) as mock_structlogger:
        warn_on_duplicate_streamed_responses(
            action_name="my_action",
            streamed_payloads=[response],
            final_responses=[response],
        )
    mock_structlogger.warning.assert_called_once()
    event_key = mock_structlogger.warning.call_args[0][0]
    assert event_key.endswith("duplicate_response")


def test_warn_on_duplicate_streamed_responses_no_warning_when_no_match() -> None:
    """No warning is emitted when the final response differs from streamed chunks."""
    with patch(
        "rasa.core.actions.custom_action_executor.structlogger"
    ) as mock_structlogger:
        warn_on_duplicate_streamed_responses(
            action_name="my_action",
            streamed_payloads=[{"text": "Hello"}],
            final_responses=[{"text": "Hello world"}],
        )
    mock_structlogger.warning.assert_not_called()


def test_warn_on_duplicate_streams_no_warning_for_empty_final_responses() -> None:
    """No warning is emitted when the final responses list is empty."""
    with patch(
        "rasa.core.actions.custom_action_executor.structlogger"
    ) as mock_structlogger:
        warn_on_duplicate_streamed_responses(
            action_name="my_action",
            streamed_payloads=[{"text": "Hello"}],
            final_responses=[],
        )
    mock_structlogger.warning.assert_not_called()


def test_warn_on_duplicate_streamed_responses_stops_after_first_duplicate() -> None:
    """Only one warning is emitted even if multiple final responses are duplicates."""
    response_a: Dict[str, Any] = {"text": "A"}
    response_b: Dict[str, Any] = {"text": "B"}
    with patch(
        "rasa.core.actions.custom_action_executor.structlogger"
    ) as mock_structlogger:
        warn_on_duplicate_streamed_responses(
            action_name="my_action",
            streamed_payloads=[response_a, response_b],
            final_responses=[response_a, response_b],
        )
    mock_structlogger.warning.assert_called_once()


def test_can_stream_returns_false_for_http_executor() -> None:
    """_can_stream() is False when the inner executor is HTTPCustomActionExecutor.

    HTTPCustomActionExecutor.supports_streaming = False, so the unary path is
    used regardless of what the output channel reports.
    """
    from rasa.core.actions.action import RemoteAction

    endpoint = EndpointConfig(url="http://localhost:5055/webhook")
    remote_action = RemoteAction("my_action", endpoint)
    # Use a streaming-capable channel to confirm the executor flag is what blocks it.
    from rasa.core.channels.rest import QueueOutputChannel

    assert remote_action._can_stream(QueueOutputChannel()) is False


def test_can_stream_returns_false_when_channel_does_not_support_streaming() -> None:
    """_can_stream() is False for a streaming-capable executor + non-streaming channel.

    CollectingOutputChannel.supports_streaming = False, which is the channel used
    for plain REST webhook requests.  Even a gRPC executor must not activate the
    streaming path when the channel cannot consume streamed chunks.
    """
    from rasa.core.actions.action import RemoteAction

    endpoint = EndpointConfig(url="grpc://localhost:5055")
    remote_action = RemoteAction("my_action", endpoint)
    assert remote_action._can_stream(CollectingOutputChannel()) is False


def test_can_stream_returns_true_for_grpc_executor_and_streaming_channel() -> None:
    """_can_stream() is True when both the executor and channel support streaming.

    GRPCCustomActionExecutor.supports_streaming = True and
    QueueOutputChannel.supports_streaming = True, so the streaming path is
    activated end-to-end for SSE REST requests backed by a gRPC action server.
    """
    from rasa.core.actions.action import RemoteAction
    from rasa.core.channels.rest import QueueOutputChannel

    endpoint = EndpointConfig(url="grpc://localhost:5055")
    remote_action = RemoteAction("my_action", endpoint)
    assert remote_action._can_stream(QueueOutputChannel()) is True


def test_sanitize_payload_converts_top_level_tuple_to_list() -> None:
    """Top-level tuples are converted to lists."""
    result = GRPCCustomActionExecutor._sanitize_payload((1, 2, 3))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_sanitize_payload_converts_nested_tuples_recursively() -> None:
    """Tuples nested inside dicts and lists are recursively converted."""
    payload = {"slots": {"values": (1, (2, 3))}, "events": [(4, 5)]}
    result = GRPCCustomActionExecutor._sanitize_payload(payload)
    assert result == {"slots": {"values": [1, [2, 3]]}, "events": [[4, 5]]}


def test_sanitize_payload_leaves_lists_and_scalars_unchanged() -> None:
    """Lists, dicts, and scalar values pass through unchanged."""
    payload = {"text": "hello", "count": 42, "flags": [True, False]}
    result = GRPCCustomActionExecutor._sanitize_payload(payload)
    assert result == payload


@pytest.mark.asyncio
async def test_retry_executor_run_streaming_uses_include_domain_true_on_retry(
    tracker: DialogueStateTracker,
    domain: Domain,
) -> None:
    """The retry call passes include_domain=True so the domain is included.

    RetryCustomActionExecutor.run_streaming() catches DomainNotFound and
    retries exactly once.  The retry must set include_domain=True; without
    this the SDK would raise DomainNotFound again indefinitely.
    """
    expected = {"events": [], "responses": []}
    call_include_domain: list[bool] = []

    async def run_streaming_side_effect(
        tracker: Any,
        domain: Any,
        output_channel: Any,
        include_domain: bool = False,
    ) -> dict:
        call_include_domain.append(include_domain)
        if len(call_include_domain) == 1:
            raise DomainNotFound()
        return expected

    inner = _StreamingInnerExecutor(result=expected)
    inner.run_streaming = run_streaming_side_effect  # type: ignore[method-assign]

    retry = RetryCustomActionExecutor(inner)  # type: ignore[arg-type]
    channel = _FakeOutputChannel()
    result = await retry.run_streaming(
        tracker=tracker, domain=domain, output_channel=channel
    )

    assert result == expected
    assert call_include_domain == [
        False,
        True,
    ], "First call must omit domain; retry must set include_domain=True"
