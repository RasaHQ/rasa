from typing import Any, Dict

from rasa.builder.models import (
    JobStatus,
    JobStatusEvent,
    ServerSentEvent,
    ServerSentEventType,
)


def test_server_sent_event_type_members():
    assert {t.value for t in ServerSentEventType} == {
        ServerSentEventType.progress,
        ServerSentEventType.error,
        ServerSentEventType._EOF,
    }


def test_job_status_members():
    expected = {
        JobStatus.received.value,
        JobStatus.done.value,
        JobStatus.error.value,
        JobStatus.generating.value,
        JobStatus.generation_success.value,
        JobStatus.generation_error.value,
        JobStatus.training.value,
        JobStatus.train_success.value,
        JobStatus.train_error.value,
        JobStatus.validating.value,
        JobStatus.validation_success.value,
        JobStatus.validation_error.value,
        JobStatus.copilot_analysis_start.value,
        JobStatus.copilot_analyzing.value,
        JobStatus.copilot_analysis_success.value,
        JobStatus.copilot_analysis_error.value,
    }
    assert {s.value for s in JobStatus} == expected


def test_build_progress_event():
    event = "ok"
    data = {"event": event}
    sse = ServerSentEvent.build(event=event, data=data)
    assert sse.event == event
    assert sse.data == data
    assert sse.format() == 'event: ok\ndata: {"event": "ok"}\n\n'


def test_eof_event():
    event = ServerSentEvent.eof()
    assert event.event == ServerSentEventType._EOF.value
    assert event.data == {}
    assert event.format() == "event: _EOF\ndata: {}\n\n"


def test_job_status_event_progress():
    event = JobStatusEvent.from_status(status=JobStatus.training.value)
    assert isinstance(event, JobStatusEvent)
    assert event.event == ServerSentEventType.progress.value
    assert event.data == {"status": JobStatus.training.value}
    assert event.format() == 'event: progress\ndata: {"status": "training"}\n\n'


def test_job_status_event_error():
    message = "error"
    event = JobStatusEvent.from_status(
        status=JobStatus.train_error.value,
        message=message,
    )

    assert isinstance(event, JobStatusEvent)
    assert event.event == ServerSentEventType.error.value
    assert event.data == {"status": JobStatus.train_error.value, "message": message}

    assert (
        event.format()
        == 'event: error\ndata: {"status": "train_error", "message": "error"}\n\n'
    )


def test_job_status_event_with_payload():
    """Test JobStatusEvent.from_status with payload parameter."""
    payload: Dict[str, Any] = {"custom_field": "custom_value", "number": 42}
    event = JobStatusEvent.from_status(
        status=JobStatus.training.value,
        message="Custom message",
        payload=payload,
    )

    assert isinstance(event, JobStatusEvent)
    # When message is provided, event type becomes 'error'
    assert event.event == ServerSentEventType.error.value
    expected_data: Dict[str, Any] = {
        "status": JobStatus.training.value,
        "message": "Custom message",
        "custom_field": "custom_value",
        "number": 42,
    }
    assert event.data == expected_data

    expected_format = (
        "event: error\n"
        'data: {"status": "training", "message": "Custom message", '
        '"custom_field": "custom_value", "number": 42}\n\n'
    )
    assert event.format() == expected_format


def test_job_status_event_with_payload_no_message():
    """Test JobStatusEvent.from_status with payload parameter but no message."""
    payload: Dict[str, Any] = {"custom_field": "custom_value", "number": 42}
    event = JobStatusEvent.from_status(
        status=JobStatus.training.value,
        payload=payload,
    )

    assert isinstance(event, JobStatusEvent)
    # When no message is provided, event type is 'progress'
    assert event.event == ServerSentEventType.progress.value
    expected_data: Dict[str, Any] = {
        "status": JobStatus.training.value,
        "custom_field": "custom_value",
        "number": 42,
    }
    assert event.data == expected_data

    expected_format = (
        "event: progress\n"
        'data: {"status": "training", "custom_field": "custom_value", "number": 42}\n\n'
    )
    assert event.format() == expected_format
