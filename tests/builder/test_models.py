from typing import Any, Dict

import pytest

from rasa.builder.models import (
    CommitDiffWithContentsResponse,
    CommitFileContents,
    JobStatus,
    JobStatusEvent,
    RestoreFromBackupRequest,
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
        JobStatus.heartbeat.value,
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
        JobStatus.copilot_template_prompt.value,
        JobStatus.copilot_welcome_message.value,
        JobStatus.train_success_message.value,
        JobStatus.commit.value,
        # Git-specific statuses
        JobStatus.cloning.value,
        JobStatus.clone_success.value,
        JobStatus.clone_error.value,
        JobStatus.switching_branch.value,
        JobStatus.branch_switch_success.value,
        JobStatus.branch_switch_error.value,
        JobStatus.rolling_back.value,
        JobStatus.rollback_success.value,
        JobStatus.rollback_message.value,
        JobStatus.rollback_error.value,
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


class TestRestoreFromBackupRequest:
    """Test RestoreFromBackupRequest model validation."""

    def test_valid_presigned_url_request(self):
        valid_url = "https://s3.amazonaws.com/bucket/path?signature=test"
        request = RestoreFromBackupRequest(presigned_url=valid_url)
        assert request.presigned_url == valid_url

    def test_presigned_url_request_strips_whitespace(self):
        valid_url = "https://s3.amazonaws.com/bucket/path?signature=test"
        request = RestoreFromBackupRequest(presigned_url=f"  {valid_url}  ")
        assert request.presigned_url == valid_url

    def test_presigned_url_request_empty_data_fails(self):
        with pytest.raises(ValueError, match="String should have at least 1 character"):
            RestoreFromBackupRequest(presigned_url="")

    def test_presigned_url_request_whitespace_only_fails(self):
        with pytest.raises(ValueError, match="Presigned URL cannot be empty"):
            RestoreFromBackupRequest(presigned_url="   ")

    def test_presigned_url_request_invalid_url_fails(self):
        with pytest.raises(ValueError, match="must be a valid HTTP/HTTPS URL"):
            RestoreFromBackupRequest(presigned_url="not-a-valid-url")


class TestCommitFileContents:
    """Test CommitFileContents model validation."""

    def test_valid_commit_file_contents(self):
        valid_commit_file_contents = CommitFileContents(
            status="R",
            content_original="original content",
            content_modified="modified content",
            path_original="path/to/original.txt",
            path_modified="path/to/modified.txt",
        )
        assert valid_commit_file_contents.status == "R"
        assert valid_commit_file_contents.content_original == "original content"
        assert valid_commit_file_contents.content_modified == "modified content"
        assert valid_commit_file_contents.path_original == "path/to/original.txt"
        assert valid_commit_file_contents.path_modified == "path/to/modified.txt"

    def test_invalid_status(self):
        with pytest.raises(
            ValueError, match="Invalid status: X, must be one of R, A, M, D"
        ):
            CommitFileContents(
                status="X",
                content_original="original content",
                content_modified="modified content",
                path_original="path/to/original.txt",
                path_modified="path/to/modified.txt",
            )

    def test_optional_path_defaults_to_none(self):
        valid_commit_file_contents = CommitFileContents(
            status="D",
        )
        assert valid_commit_file_contents.content_original is None
        assert valid_commit_file_contents.content_modified is None
        assert valid_commit_file_contents.path_original is None
        assert valid_commit_file_contents.path_modified is None

    @pytest.mark.parametrize(
        "status",
        ["R", "A", "M", "D"],
        ids=["Renamed", "Added", "Modified", "Deleted"],
    )
    def test_all_valid_statuses(self, status: str):
        """Test that all valid statuses are accepted."""
        commit_file_contents = CommitFileContents(
            status=status,
            content_original="original",
            content_modified="modified",
        )
        assert commit_file_contents.status == status


class TestCommitDiffWithContentsResponse:
    """Test CommitDiffWithContentsResponse model."""

    def test_valid_response_with_files(self):
        """Test creating response with multiple files."""
        files = {
            "domain.yml": CommitFileContents(
                status="M",
                content_original="old content",
                content_modified="new content",
            ),
            "flows.yml": CommitFileContents(
                status="A",
                content_original="",
                content_modified="new flow content",
            ),
        }
        response = CommitDiffWithContentsResponse(files=files)
        assert len(response.files) == 2
        assert "domain.yml" in response.files
        assert "flows.yml" in response.files
        assert response.files["domain.yml"].status == "M"
        assert response.files["flows.yml"].status == "A"

    def test_empty_files_dict(self):
        """Test creating response with empty files dict."""
        response = CommitDiffWithContentsResponse(files={})
        assert response.files == {}
