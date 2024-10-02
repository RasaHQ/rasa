import functools
import threading
import typing
import uuid
from typing import Any, Callable, Dict, List, Text

import pytest
import requests

from tests.conftest import wait
from tests.integration_tests.conftest import send_message_to_rasa_server
from tests.integration_tests.tracing.conftest import (
    ACTION_SERVER_ACTION_TRIGGERED,
    ACTION_SERVER_FORM_FILL_MESSAGE,
    ACTION_SERVER_FORM_VALIDATION_ACTION_TRIGGERED,
    ACTION_SERVER_FORM_TRIGGER_MESSAGE,
    ACTION_SERVER_JAEGER_TRACING_SERVICE_NAME,
    ACTION_SERVER_OTLP_ACTION_SERVER_NAME,
    ACTION_SERVER_PARENT_SPAN_NAME,
    ACTION_SERVER_PARENT_SUB_SPAN_NAME,
    ACTION_SERVER_SPAN_NAME,
    ACTION_SERVER_TRIGGER_MESSAGE,
    RASA_JAEGER_TRACING_SERVICE_NAME,
    RASA_OTLP_TRACING_SERVICE_NAME,
    RASA_SERVER_JAEGER,
    RASA_SERVER_JAEGER_NO_ACTION_SERVER,
    RASA_SERVER_OTLP,
    RASA_SERVER_OTLP_NO_ACTION_SERVER,
    RASA_SERVER_PARENT_SPAN_NAME,
    RASA_SERVER_PROCESSOR_SPAN_NAME,
    RASA_SERVER_PROCESSOR_SUB_SPAN_NAME,
    DIRECT_CUSTOM_ACTION_EXECUTION_SUB_SPAN_NAME,
    RASA_SERVER_TRIGGER_MESSAGE,
    TraceQueryTimestamps,
    GRPC_RASA_SERVER_JAEGER,
    GRPC_SSL_RASA_SERVER_JAEGER,
    GRPC_ACTION_SERVER_PARENT_SUB_SPAN_NAME,
)

if typing.TYPE_CHECKING:
    # imports need to be scoped to functions and for type checking as
    # the dependencies are not available locally for unit testing
    from api_v3.query_service_pb2 import SpansResponseChunk, TraceQueryParameters
    from api_v3.query_service_pb2_grpc import QueryServiceStub
    from model_pb2 import Span


@pytest.mark.parametrize(
    "tracing_service_name, rasa_server_endpoint",
    [
        (RASA_JAEGER_TRACING_SERVICE_NAME, RASA_SERVER_JAEGER),
        (RASA_OTLP_TRACING_SERVICE_NAME, RASA_SERVER_OTLP),
    ],
)
def test_traces_get_sent_to_backend(
    jaeger_query_service: "QueryServiceStub",
    tracing_service_name: Text,
    rasa_server_endpoint: Text,
    trace_query_timestamps: TraceQueryTimestamps,
) -> None:
    if tracing_service_name == RASA_OTLP_TRACING_SERVICE_NAME:
        pytest.skip("Temporary disabled due to TLS timeout error")

    from api_v3.query_service_pb2 import TraceQueryParameters
    from model_pb2 import Span

    sender_id, _ = send_message_to_rasa_server(rasa_server_endpoint)
    params = TraceQueryParameters(
        service_name=tracing_service_name,
        operation_name="Agent.handle_message",
        start_time_min=trace_query_timestamps.min_time,
        start_time_max=trace_query_timestamps.max_time,
    )

    @wait_for_spans
    def _spans_with_sender_id() -> List[Span]:
        spans = _fetch_spans(jaeger_query_service, params)
        return _filter_spans_by_attributes(spans, {"sender_id": sender_id})

    spans_with_sender_id = _spans_with_sender_id()
    assert len(spans_with_sender_id) > 0


@pytest.mark.parametrize(
    "tracing_service_name, rasa_server_endpoint, parent_span_name",
    [
        (
            ACTION_SERVER_JAEGER_TRACING_SERVICE_NAME,
            RASA_SERVER_JAEGER,
            ACTION_SERVER_PARENT_SPAN_NAME,
        ),
        (
            ACTION_SERVER_OTLP_ACTION_SERVER_NAME,
            RASA_SERVER_OTLP,
            ACTION_SERVER_PARENT_SPAN_NAME,
        ),
        (
            ACTION_SERVER_JAEGER_TRACING_SERVICE_NAME,
            GRPC_RASA_SERVER_JAEGER,
            GRPC_ACTION_SERVER_PARENT_SUB_SPAN_NAME,
        ),
        (
            ACTION_SERVER_JAEGER_TRACING_SERVICE_NAME,
            GRPC_SSL_RASA_SERVER_JAEGER,
            GRPC_ACTION_SERVER_PARENT_SUB_SPAN_NAME,
        ),
    ],
)
def test_trace_context_propagated_to_action_server(
    jaeger_query_service: "QueryServiceStub",
    tracing_service_name: Text,
    rasa_server_endpoint: Text,
    trace_query_timestamps: TraceQueryTimestamps,
    parent_span_name: Text,
) -> None:
    if rasa_server_endpoint == RASA_SERVER_OTLP:
        pytest.skip("Temporary disabled due to TLS timeout error")

    from api_v3.query_service_pb2 import TraceQueryParameters
    from model_pb2 import Span

    sender_id, _ = send_message_to_rasa_server(
        rasa_server_endpoint, ACTION_SERVER_TRIGGER_MESSAGE
    )

    params = TraceQueryParameters(
        service_name=tracing_service_name,
        operation_name=ACTION_SERVER_SPAN_NAME,
        start_time_min=trace_query_timestamps.min_time,
        start_time_max=trace_query_timestamps.max_time,
    )

    @wait_for_spans
    def _spans_for_user_turn() -> List[Span]:
        spans = _fetch_spans(jaeger_query_service, params)
        return _filter_spans_by_attributes(
            spans,
            {
                "action_name": ACTION_SERVER_ACTION_TRIGGERED,
                "sender_id": sender_id,
            },
        )

    spans_for_user_turn = _spans_for_user_turn()
    action_server_spans = _filter_spans_by_name(
        spans_for_user_turn, ACTION_SERVER_SPAN_NAME
    )
    parent_spans = _filter_spans_by_name(spans_for_user_turn, parent_span_name)
    action_server_span = action_server_spans[0]
    parent_span = parent_spans[0]

    assert action_server_span.trace_id == parent_span.trace_id


@pytest.mark.parametrize(
    "tracing_service_name, rasa_server_endpoint",
    [
        (
            ACTION_SERVER_JAEGER_TRACING_SERVICE_NAME,
            RASA_SERVER_JAEGER,
        ),
        (
            ACTION_SERVER_OTLP_ACTION_SERVER_NAME,
            RASA_SERVER_OTLP,
        ),
    ],
)
def test_trace_context_propagated_to_action_server_with_form_validation_action(
    jaeger_query_service: "QueryServiceStub",
    tracing_service_name: Text,
    rasa_server_endpoint: Text,
    trace_query_timestamps: TraceQueryTimestamps,
) -> None:
    if rasa_server_endpoint == RASA_SERVER_OTLP:
        pytest.skip("Temporary disabled due to TLS timeout error")

    from api_v3.query_service_pb2 import TraceQueryParameters
    from model_pb2 import Span

    # trigger form
    sender_id, _ = send_message_to_rasa_server(
        rasa_server_endpoint, ACTION_SERVER_FORM_TRIGGER_MESSAGE
    )
    params = TraceQueryParameters(
        service_name=tracing_service_name,
        operation_name=ACTION_SERVER_SPAN_NAME,
        start_time_min=trace_query_timestamps.min_time,
        start_time_max=trace_query_timestamps.max_time,
    )

    # fill form
    sender_id, _ = send_message_to_rasa_server(
        rasa_server_endpoint, ACTION_SERVER_FORM_FILL_MESSAGE, sender_id
    )

    @wait_for_spans
    def _spans_for_user_turn() -> List[Span]:
        spans = _fetch_spans(jaeger_query_service, params)
        return _filter_spans_by_attributes(
            spans,
            {
                "action_name": ACTION_SERVER_FORM_VALIDATION_ACTION_TRIGGERED,
                "sender_id": sender_id,
            },
        )

    spans_for_user_turn = _spans_for_user_turn()

    action_server_spans = _filter_spans_by_name(
        spans_for_user_turn, ACTION_SERVER_SPAN_NAME
    )
    parent_spans = _filter_spans_by_name(
        spans_for_user_turn, ACTION_SERVER_PARENT_SUB_SPAN_NAME
    )
    action_server_span = action_server_spans[0]
    parent_span = parent_spans[0]

    assert action_server_span.trace_id == parent_span.trace_id


@pytest.mark.parametrize(
    "tracing_service_name, rasa_server_endpoint",
    [
        (
            RASA_JAEGER_TRACING_SERVICE_NAME,
            RASA_SERVER_JAEGER_NO_ACTION_SERVER,
        ),
        (
            RASA_OTLP_TRACING_SERVICE_NAME,
            RASA_SERVER_OTLP_NO_ACTION_SERVER,
        ),
    ],
)
def test_missing_action_server_endpoint_does_not_stop_tracing(
    jaeger_query_service: "QueryServiceStub",
    tracing_service_name: Text,
    rasa_server_endpoint: Text,
    trace_query_timestamps: TraceQueryTimestamps,
) -> None:
    if tracing_service_name == RASA_OTLP_TRACING_SERVICE_NAME:
        pytest.skip("Temporary disabled due to TLS timeout error")

    from api_v3.query_service_pb2 import TraceQueryParameters
    from model_pb2 import Span

    sender_id, _ = send_message_to_rasa_server(
        rasa_server_endpoint, ACTION_SERVER_TRIGGER_MESSAGE
    )
    tracker = _fetch_tracker(server_location=rasa_server_endpoint, sender_id=sender_id)
    message_id = tracker.get("latest_message", {}).get("message_id")

    params = TraceQueryParameters(
        service_name=tracing_service_name,
        operation_name=ACTION_SERVER_PARENT_SPAN_NAME,
        start_time_min=trace_query_timestamps.min_time,
        start_time_max=trace_query_timestamps.max_time,
    )

    @wait_for_spans
    def _spans_for_user_turn() -> List[Span]:
        spans = _fetch_spans(jaeger_query_service, params)
        return _filter_spans_by_attributes(
            spans,
            {
                "action_name": ACTION_SERVER_ACTION_TRIGGERED,
                "sender_id": sender_id,
                "message_id": message_id,
            },
        )

    spans_for_user_turn = _spans_for_user_turn()
    parent_spans = _filter_spans_by_name(
        spans_for_user_turn, ACTION_SERVER_PARENT_SPAN_NAME
    )
    parent_span = parent_spans[0]

    # assert no error code returned by parent span
    assert parent_span.status.code == 0


@pytest.mark.parametrize(
    "tracing_service_name, rasa_server_endpoint, sub_span_name",
    [
        (
            RASA_JAEGER_TRACING_SERVICE_NAME,
            RASA_SERVER_JAEGER_NO_ACTION_SERVER,
            RASA_SERVER_PROCESSOR_SUB_SPAN_NAME,
        ),
        (
            RASA_OTLP_TRACING_SERVICE_NAME,
            RASA_SERVER_OTLP_NO_ACTION_SERVER,
            RASA_SERVER_PROCESSOR_SUB_SPAN_NAME,
        ),
        (
            RASA_JAEGER_TRACING_SERVICE_NAME,
            RASA_SERVER_JAEGER_NO_ACTION_SERVER,
            DIRECT_CUSTOM_ACTION_EXECUTION_SUB_SPAN_NAME,
        ),
        (
            RASA_OTLP_TRACING_SERVICE_NAME,
            RASA_SERVER_OTLP_NO_ACTION_SERVER,
            DIRECT_CUSTOM_ACTION_EXECUTION_SUB_SPAN_NAME,
        ),
    ],
)
def test_context_propagated_to_subspans_in_rasa_server(
    jaeger_query_service: "QueryServiceStub",
    tracing_service_name: Text,
    rasa_server_endpoint: Text,
    sub_span_name: Text,
    trace_query_timestamps: TraceQueryTimestamps,
) -> None:
    if tracing_service_name == RASA_OTLP_TRACING_SERVICE_NAME:
        pytest.skip("Temporary disabled due to TLS timeout error")

    from api_v3.query_service_pb2 import TraceQueryParameters
    from model_pb2 import Span

    send_message_to_rasa_server(rasa_server_endpoint, RASA_SERVER_TRIGGER_MESSAGE)

    params = TraceQueryParameters(
        service_name=tracing_service_name,
        operation_name=sub_span_name,
        start_time_min=trace_query_timestamps.min_time,
        start_time_max=trace_query_timestamps.max_time,
    )

    @wait_for_spans
    def _spans_for_user_turn() -> List[Span]:
        spans = _fetch_spans(jaeger_query_service, params)
        return _filter_spans_by_attributes(
            spans,
            {},
        )

    spans_for_user_turn = _spans_for_user_turn()
    processor_sub_spans = _filter_spans_by_name(spans_for_user_turn, sub_span_name)
    sub_parent_spans = _filter_spans_by_name(
        spans_for_user_turn, RASA_SERVER_PROCESSOR_SPAN_NAME
    )
    parent_spans = _filter_spans_by_name(
        spans_for_user_turn, RASA_SERVER_PARENT_SPAN_NAME
    )
    processor_sub_span = processor_sub_spans[0]
    sub_parent_span = sub_parent_spans[0]
    parent_span = parent_spans[0]

    assert processor_sub_span.trace_id == sub_parent_span.trace_id
    assert sub_parent_span.trace_id == parent_span.trace_id


@pytest.mark.parametrize(
    "tracing_service_name, rasa_server_endpoint",
    [
        (
            RASA_JAEGER_TRACING_SERVICE_NAME,
            RASA_SERVER_JAEGER_NO_ACTION_SERVER,
        ),
        (
            RASA_OTLP_TRACING_SERVICE_NAME,
            RASA_SERVER_OTLP_NO_ACTION_SERVER,
        ),
    ],
)
def test_headers_context_propagated_to_rasa(
    jaeger_query_service: "QueryServiceStub",
    tracing_service_name: Text,
    rasa_server_endpoint: Text,
    trace_query_timestamps: TraceQueryTimestamps,
) -> None:
    if tracing_service_name == RASA_OTLP_TRACING_SERVICE_NAME:
        pytest.skip("Temporary disabled due to TLS timeout error")

    from api_v3.query_service_pb2 import TraceQueryParameters
    from model_pb2 import Span

    sender_id = str(uuid.uuid4())
    traceparent = "00-ec6fbe3de34342bac2e9fe6e955354cb-c04faeec2b0670e4-01"
    requests.post(
        f"{rasa_server_endpoint}/webhooks/rest/webhook",
        json={"sender": sender_id, "message": RASA_SERVER_TRIGGER_MESSAGE},
        headers={"traceparent": traceparent},
    )

    params = TraceQueryParameters(
        service_name=tracing_service_name,
        operation_name=RASA_SERVER_PARENT_SPAN_NAME,
        start_time_min=trace_query_timestamps.min_time,
        start_time_max=trace_query_timestamps.max_time,
    )

    @wait_for_spans
    def _spans_for_user_turn() -> List[Span]:
        spans = _fetch_spans(jaeger_query_service, params)
        return _filter_spans_by_attributes(
            spans,
            {"sender_id": sender_id},
        )

    spans_for_user_turn = _spans_for_user_turn()
    parent_spans = _filter_spans_by_name(
        spans_for_user_turn, RASA_SERVER_PARENT_SPAN_NAME
    )
    parent_span = parent_spans[0]

    assert int.from_bytes(parent_span.trace_id, byteorder="big") == int(
        traceparent.split("-")[1], 16
    )


def _fetch_tracker(server_location: Text, sender_id: Text = "test") -> Dict[str, Any]:
    tracker_response = requests.get(
        f"{server_location}/conversations/{sender_id}/tracker"
    )
    return tracker_response.json()


def wait_for_spans(span_query_function: Callable) -> Callable:
    from model_pb2 import Span

    result_available_event = threading.Event()

    @functools.wraps(span_query_function)
    def wrapper() -> List[Span]:
        wait(
            lambda: len(span_query_function()) > 0,
            result_available_event=result_available_event,
            timeout_seconds=30,
            max_sleep=4,
            waiting_for="tracing messages to be sent to backend",
        )

        return span_query_function()

    return wrapper


def _fetch_spans(
    jaeger_query_service: "QueryServiceStub", params: "TraceQueryParameters"
) -> List["Span"]:
    from api_v3.query_service_pb2 import FindTracesRequest

    request = FindTracesRequest(query=params)
    response = jaeger_query_service.FindTraces(request)
    return _collect_spans(response)


def _collect_spans(response_chunk: "SpansResponseChunk") -> List[Dict[str, Any]]:
    from model_pb2 import Span

    spans: List[Span] = []
    for chunk in response_chunk:
        for resource_span in chunk.resource_spans:
            for scope_spans in resource_span.scope_spans:
                for span in scope_spans.spans:
                    spans.append(span)
    return spans


def _collect_span_attributes(span: "Span") -> Dict[str, str]:
    span_attributes = {
        attribute.key: attribute.value.string_value for attribute in span.attributes
    }
    return span_attributes


def _filter_spans_by_attributes(
    spans: List["Span"], attributes: Dict[str, str]
) -> List["Span"]:
    eligible_spans = [
        span
        for span in spans
        if attributes.items() <= _collect_span_attributes(span).items()
    ]
    return eligible_spans


def _filter_spans_by_name(spans: List["Span"], span_name: Text) -> List["Span"]:
    eligible_spans = [span for span in spans if span.name == span_name]
    return eligible_spans
