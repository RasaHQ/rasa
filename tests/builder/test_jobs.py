"""Tests for rasa.builder.jobs module."""

import io
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from rasa.builder.copilot.models import (
    CopilotGenerationContext,
    GeneratedContent,
    ReferenceEntry,
    ReferenceSection,
    ResponseCategory,
    ResponseCompleteness,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.exceptions import (
    TrainingError,
    ValidationError,
)
from rasa.builder.job_manager import JobInfo, job_manager
from rasa.builder.jobs import (
    _safe_tar_members,
    run_backup_to_bot_job,
    run_copilot_go_back_in_time_success_job,
    run_copilot_template_prompt_job,
    run_copilot_training_error_analysis_job,
    run_copilot_training_success_job,
    run_copilot_welcome_message_job,
    run_prompt_to_bot_job,
    run_replace_all_files_job,
    run_template_to_bot_job,
)
from rasa.builder.models import JobStatus
from rasa.builder.project_generator import DEFAULT_COMMIT_INFO, ProjectGenerator
from rasa.cli.scaffold import ProjectTemplateName


@pytest.fixture
def mock_app() -> MagicMock:
    """Create a mock Sanic app for testing."""
    project_generator = MagicMock()
    app = MagicMock()
    app.ctx = SimpleNamespace()
    app.ctx.project_generator = project_generator
    return app


@pytest.fixture
def sample_bot_files() -> Dict[str, Any]:
    """Sample bot files for testing."""
    return {
        "config.yml": "version: '3.1'\npipeline: []",
        "domain.yml": "version: '3.1'\nintents: []",
        "data/nlu.yml": "version: '3.1'\nnlu: []",
    }


@pytest.fixture
def mock_job() -> JobInfo:
    """Mock JobInfo."""
    job = MagicMock(spec=JobInfo)
    job.id = "test_job_123"
    job.commit_sha = None
    job.put = AsyncMock()
    job._queue = MagicMock()
    job._queue.put_nowait = MagicMock()
    return job


class TestRunReplaceAllFilesJob:
    """Test the run_replace_all_files_job function."""

    @pytest.fixture
    def job_status_tracker(self) -> Tuple[List[str], Callable]:
        """Fixture to track status events while preserving original behavior."""
        status_events = []

        # Import the original function
        from rasa.builder.jobs import (
            push_job_status_event as original_push_job_status_event,
        )

        async def track_status_event(job_param, status, **kwargs):
            # Handle both JobStatus enum and string values
            if hasattr(status, "value"):
                status_events.append(status.value)
            else:
                status_events.append(status)
            # Call the original function to preserve its behavior
            await original_push_job_status_event(job_param, status, **kwargs)

        return status_events, track_status_event

    @pytest.mark.asyncio
    async def test_successful_execution(
        self, mock_app, sample_bot_files: Dict[str, Any], monkeypatch
    ):
        """Test successful execution of replace_all_files_job."""
        job = job_manager.create_job()

        # Mock the project generator methods
        mock_app.ctx.project_generator.replace_all_bot_files = AsyncMock(
            return_value="abc123"
        )

        # Mock training input and validation
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        # Setup mocks using monkeypatch
        mock_validate = AsyncMock()
        mock_train = AsyncMock()
        mock_link_model = AsyncMock()
        mock_update_agent = MagicMock()

        monkeypatch.setattr("rasa.builder.jobs.validate_project", mock_validate)
        monkeypatch.setattr("rasa.builder.job_helpers.train_and_load_agent", mock_train)
        monkeypatch.setattr(
            "rasa.builder.job_helpers.link_model_to_commit", mock_link_model
        )
        monkeypatch.setattr("rasa.builder.jobs.update_agent", mock_update_agent)

        mock_validate.return_value = None  # No validation error
        mock_agent = Mock()
        mock_train.return_value = mock_agent

        await run_replace_all_files_job(
            mock_app, job, sample_bot_files, DEFAULT_COMMIT_INFO
        )

        # Verify the flow
        mock_app.ctx.project_generator.replace_all_bot_files.assert_called_once_with(
            sample_bot_files, DEFAULT_COMMIT_INFO
        )
        mock_validate.assert_called_once_with(mock_training_input.importer)
        mock_train.assert_called_once_with(mock_training_input)
        mock_link_model.assert_called_once_with(
            mock_app.ctx.project_generator.git_service, mock_agent, "abc123"
        )
        mock_update_agent.assert_called_once_with(mock_agent, mock_app)

        # Check job status
        assert job.status == JobStatus.done.value

    @pytest.mark.asyncio
    async def test_validation_error(
        self,
        mock_app: MagicMock,
        sample_bot_files: Dict[str, Any],
        job_status_tracker: tuple[list[str], Callable],
        monkeypatch,
    ) -> None:
        """Test handling of validation errors."""
        job = job_manager.create_job()
        status_events, track_status_event = job_status_tracker

        mock_app.ctx.project_generator.replace_all_bot_files = AsyncMock(
            return_value="abc123"
        )
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        validation_error = ValidationError("Invalid configuration")
        validation_error.validation_logs = [
            {"log_level": "error", "message": "Error 1", "file": "domain.yml"},
            {"log_level": "error", "message": "Error 2", "file": "stories.yml"},
        ]

        # Setup mocks using monkeypatch
        mock_validate = AsyncMock()
        mock_validate.side_effect = validation_error
        mock_copilot_analysis = AsyncMock()
        monkeypatch.setattr(
            "rasa.builder.jobs.run_copilot_training_error_analysis_job",
            mock_copilot_analysis,
        )

        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", track_status_event
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.push_job_status_event", track_status_event
        )
        monkeypatch.setattr("rasa.builder.jobs.validate_project", mock_validate)
        monkeypatch.setattr(
            "rasa.builder.jobs.config.VALIDATION_FAIL_ON_WARNINGS", False
        )

        await run_replace_all_files_job(
            mock_app, job, sample_bot_files, DEFAULT_COMMIT_INFO
        )

        # Verify file replacement was called
        mock_app.ctx.project_generator.replace_all_bot_files.assert_called_once_with(
            sample_bot_files, DEFAULT_COMMIT_INFO
        )
        mock_validate.assert_called_once_with(mock_training_input.importer)

        # Check that the job ended with validation error status (with copilot job ID)
        assert job.status == JobStatus.validation_error.value

        # Verify the sequence:
        # last event should be validation_error (with copilot job ID)
        assert len(status_events) >= 1
        assert status_events[-1] == JobStatus.validation_error.value

    @pytest.mark.asyncio
    async def test_training_error(
        self,
        mock_app: MagicMock,
        sample_bot_files: Dict[str, Any],
        job_status_tracker: Tuple[List[str], Callable],
        monkeypatch,
    ) -> None:
        """Test handling of training errors."""
        job = job_manager.create_job()
        status_events, track_status_event = job_status_tracker

        mock_app.ctx.project_generator.replace_all_bot_files = AsyncMock(
            return_value="abc123"
        )
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        training_error = TrainingError("Training failed")

        # Setup mocks using monkeypatch
        mock_validate = AsyncMock()
        mock_train = AsyncMock()

        mock_validate.return_value = None  # No validation error
        mock_train.side_effect = training_error

        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", track_status_event
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.push_job_status_event", track_status_event
        )
        monkeypatch.setattr("rasa.builder.jobs.validate_project", mock_validate)
        monkeypatch.setattr("rasa.builder.job_helpers.train_and_load_agent", mock_train)

        await run_replace_all_files_job(
            mock_app, job, sample_bot_files, DEFAULT_COMMIT_INFO
        )

        # Verify the flow up to training
        mock_app.ctx.project_generator.replace_all_bot_files.assert_called_once_with(
            sample_bot_files, DEFAULT_COMMIT_INFO
        )
        mock_validate.assert_called_once_with(mock_training_input.importer)
        mock_train.assert_called_once_with(mock_training_input)

        # Check that the job ended with training error status (with copilot job ID)
        assert job.status == JobStatus.train_error.value

        # Verify the sequence:
        # last event should be train_error (with copilot job ID)
        assert len(status_events) >= 1
        assert status_events[-1] == JobStatus.train_error.value

    @pytest.mark.asyncio
    async def test_unexpected_error(
        self,
        mock_app: MagicMock,
        sample_bot_files: Dict[str, Any],
        job_status_tracker: Tuple[List[str], Callable],
        monkeypatch,
    ) -> None:
        """Test handling of unexpected errors."""
        job = job_manager.create_job()
        status_events, track_status_event = job_status_tracker

        # Mock an unexpected error during file replacement
        mock_app.ctx.project_generator.replace_all_bot_files.side_effect = Exception(
            "Unexpected error"
        )

        # Setup mocks using monkeypatch
        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", track_status_event
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.push_job_status_event", track_status_event
        )

        await run_replace_all_files_job(
            mock_app, job, sample_bot_files, DEFAULT_COMMIT_INFO
        )

        # Check that the job ended with error status (with copilot job ID)
        assert job.status == JobStatus.error.value

        # Verify the sequence:
        # last event should be error (with copilot job ID)
        assert len(status_events) >= 1
        assert status_events[-1] == JobStatus.error.value

    @pytest.mark.asyncio
    async def test_job_status_progression(
        self, mock_app, sample_bot_files: Dict[str, Any], monkeypatch
    ):
        job = job_manager.create_job()
        status_events = []

        original_put = JobInfo.put

        async def track_status(self, event):
            # status is stored in the event's data payload
            if status := (event.data or {}).get("status"):
                status_events.append(status)
            return await original_put(self, event)

        mock_app.ctx.project_generator.replace_all_bot_files = AsyncMock(
            return_value="abc123"
        )
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        # Setup mocks using monkeypatch
        mock_validate = AsyncMock(return_value=None)
        mock_train = AsyncMock(return_value=MagicMock())
        mock_link_model = AsyncMock()
        mock_update_agent = MagicMock()
        mock_copilot_analysis = AsyncMock()
        mock_app.ctx.project_generator.replace_all_bot_files.return_value = "abc123"
        monkeypatch.setattr("rasa.builder.job_manager.JobInfo.put", track_status)
        monkeypatch.setattr("rasa.builder.jobs.validate_project", mock_validate)
        monkeypatch.setattr("rasa.builder.job_helpers.train_and_load_agent", mock_train)
        monkeypatch.setattr(
            "rasa.builder.job_helpers.link_model_to_commit", mock_link_model
        )
        monkeypatch.setattr("rasa.builder.jobs.update_agent", mock_update_agent)
        monkeypatch.setattr(
            "rasa.builder.jobs.run_copilot_training_error_analysis_job",
            mock_copilot_analysis,
        )
        monkeypatch.setattr(
            "rasa.builder.jobs.run_copilot_training_success_job",
            AsyncMock(),
        )

        await run_replace_all_files_job(
            mock_app, job, sample_bot_files, DEFAULT_COMMIT_INFO
        )

        # Verify status progression
        expected_statuses = [
            JobStatus.received.value,
            JobStatus.validating.value,
            JobStatus.validation_success.value,
            JobStatus.training.value,
            JobStatus.train_success.value,
            JobStatus.done.value,
        ]

        assert status_events == expected_statuses

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised_exception,should_create_copilot_job",
        [
            (None, False),  # Success case
            (TrainingError("Training failed"), True),
            (ValidationError("Validation failed"), True),
            (Exception("Unexpected error"), True),
        ],
    )
    async def test_run_update_files_job_creates_copilot_analysis_job(
        self,
        mock_app: MagicMock,
        mock_job: JobInfo,
        sample_bot_files: Dict[str, Any],
        raised_exception: Optional[Exception],
        should_create_copilot_job: bool,
        monkeypatch,
    ) -> None:
        """Test run_update_files_job with different scenarios."""
        # Setup mocks
        mock_push_job_status_event = AsyncMock()
        mock_push_error_and_start_copilot = AsyncMock()
        mock_validate = AsyncMock()
        mock_train = AsyncMock()
        mock_link_model = AsyncMock()
        mock_update_agent = MagicMock()

        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", mock_push_job_status_event
        )
        monkeypatch.setattr("rasa.builder.jobs.update_agent", mock_update_agent)
        monkeypatch.setattr("rasa.builder.job_helpers.train_and_load_agent", mock_train)
        monkeypatch.setattr(
            "rasa.builder.job_helpers.link_model_to_commit", mock_link_model
        )
        monkeypatch.setattr("rasa.builder.jobs.validate_project", mock_validate)
        monkeypatch.setattr(
            "rasa.builder.jobs.push_error_and_start_copilot_analysis",
            mock_push_error_and_start_copilot,
        )

        # Given
        mock_update_agent.return_value = None

        # Set up the mock to return an AsyncMock for replace_all_bot_files
        mock_app.ctx.project_generator.replace_all_bot_files = AsyncMock(
            return_value="abc123"
        )

        if raised_exception is None:
            mock_validate.return_value = None
            mock_link_model.return_value = None
            mock_train.return_value = Mock()
        else:
            mock_validate.side_effect = raised_exception

        # When
        await run_replace_all_files_job(
            mock_app, mock_job, sample_bot_files, DEFAULT_COMMIT_INFO
        )

        # Then
        if should_create_copilot_job:
            mock_push_error_and_start_copilot.assert_called_once()
            call_args = mock_push_error_and_start_copilot.call_args
            assert call_args[0][0] == mock_app
            assert call_args[0][1] == mock_job
            assert call_args[0][2] is not None  # error_message
            assert call_args[0][3] is not None  # job_status
            assert call_args[0][4] == sample_bot_files
            assert call_args[0][5] == "abc123"  # commit_sha
        else:
            mock_push_error_and_start_copilot.assert_not_called()


class TestRunCopilotTrainingErrorAnalysisJob:
    """Test cases for run_copilot_training_error_analysis_job."""

    @pytest.fixture
    def mock_app(self) -> MagicMock:
        """Mock Sanic app."""
        return MagicMock()

    @pytest.fixture
    def mock_error_message(self) -> str:
        """Mock error message."""
        return "Training failed with error: Invalid configuration"

    @pytest.mark.asyncio
    async def test_run_copilot_training_error_analysis_job_success(
        self,
        mock_app: MagicMock,
        mock_job: JobInfo,
        sample_bot_files: Dict[str, Any],
        monkeypatch,
        mock_error_message: str,
    ) -> None:
        """Test successful copilot analysis job with content validation."""
        # Setup mocks
        mock_push_event = AsyncMock()
        mock_instantiate_copilot = MagicMock()
        mock_instantiate_handler = MagicMock()

        monkeypatch.setattr("rasa.builder.jobs.push_job_status_event", mock_push_event)
        monkeypatch.setattr(
            "rasa.builder.jobs.llm_service.instantiate_copilot",
            mock_instantiate_copilot,
        )
        monkeypatch.setattr(
            "rasa.builder.jobs.llm_service.instantiate_handler",
            mock_instantiate_handler,
        )

        # Given
        mock_copilot = MagicMock()
        mock_handler = MagicMock()
        mock_instantiate_copilot.return_value = mock_copilot
        mock_instantiate_handler.return_value = mock_handler

        mock_token = GeneratedContent(
            content="Analysis result",
            response_category=ResponseCategory.COPILOT,
            response_completeness=ResponseCompleteness.TOKEN,
        )
        mock_document = Document(
            content="Test documentation content",
            url="https://rasa.com/docs",
            title="Test Doc",
        )
        mock_reference_section = ReferenceSection(
            references=[
                ReferenceEntry(index=1, title="Test Doc", url="https://rasa.com/docs")
            ],
            response_category=ResponseCategory.REFERENCE,
            response_completeness=ResponseCompleteness.COMPLETE,
        )
        mock_generation_context = CopilotGenerationContext(
            relevant_documents=[mock_document],
            system_message={"role": "system", "content": "Test system message"},
            chat_history=[],
            last_user_message={"role": "user", "content": "Test user message"},
            tracker_event_attachments=[],
        )

        async def mock_response_stream():
            yield "text"

        async def mock_stream():
            yield mock_token

        mock_handler.handle_response.return_value = mock_stream()
        mock_handler.extract_references.return_value = mock_reference_section
        mock_copilot.generate_response = AsyncMock(
            return_value=(mock_response_stream(), mock_generation_context)
        )

        # When
        await run_copilot_training_error_analysis_job(
            mock_app, mock_job, mock_error_message, sample_bot_files
        )

        # Then
        # Updated to account for TrainingErrorLog event
        assert mock_push_event.call_count >= 3
        mock_copilot.generate_response.assert_called_once()
        mock_handler.extract_references.assert_called_once_with([mock_document])

        # Verify that generate_response was called with a context containing the
        # internal message
        call_args = mock_copilot.generate_response.call_args[0]
        context = call_args[0]  # First argument is the context

        # Verify the context has the expected structure
        assert hasattr(context, "copilot_chat_history")
        assert len(context.copilot_chat_history) == 1

        internal_message = context.copilot_chat_history[0]
        assert hasattr(internal_message, "content")
        # Should have at least log + file content blocks
        assert len(internal_message.content) >= 2

        # Verify log content block exists
        log_content = next(
            (block for block in internal_message.content if block.type == "log"), None
        )
        assert log_content is not None
        assert log_content.content == mock_error_message
        assert log_content.context == "training_error"

        # Verify file content blocks exist
        file_content_blocks = [
            block for block in internal_message.content if block.type == "file"
        ]
        assert len(file_content_blocks) == len(sample_bot_files)

        # Verify each file from sample_bot_files has a corresponding content block
        for file_path, file_content in sample_bot_files.items():
            matching_block = next(
                (
                    block
                    for block in file_content_blocks
                    if block.file_path == file_path
                ),
                None,
            )
            assert matching_block is not None
            assert matching_block.file_content == file_content

        # Verify that TrainingErrorLog was sent as part of copilot_analyzing stream
        training_error_log_calls = [
            call
            for call in mock_push_event.call_args_list
            if (
                call[0][1] == JobStatus.copilot_analyzing
                and call[1].get("payload") is not None
                and "logs" in call[1]["payload"]
            )
        ]
        assert len(training_error_log_calls) >= 1

    @pytest.mark.asyncio
    async def test_run_copilot_training_error_analysis_job_error(
        self,
        mock_app: MagicMock,
        mock_job: JobInfo,
        sample_bot_files: Dict[str, Any],
        mock_error_message: str,
        monkeypatch,
    ) -> None:
        """Test copilot analysis job with error."""
        # Setup mocks
        mock_push_event = AsyncMock()
        mock_instantiate_copilot = MagicMock()

        monkeypatch.setattr("rasa.builder.jobs.push_job_status_event", mock_push_event)
        monkeypatch.setattr(
            "rasa.builder.jobs.llm_service.instantiate_copilot",
            mock_instantiate_copilot,
        )

        # Given
        mock_instantiate_copilot.side_effect = Exception("Copilot error")

        # When
        await run_copilot_training_error_analysis_job(
            mock_app, mock_job, mock_error_message, sample_bot_files
        )

        # Then
        error_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.copilot_analysis_error
        ]
        assert len(error_calls) == 1


class TestCopilotWelcomeMessage:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        # Create mocks for welcome message job
        self.mock_push_event = AsyncMock()
        self.mock_job_manager = MagicMock()

        # Create different job mocks for template prompt and welcome jobs
        self.template_prompt_job = MagicMock()
        self.template_prompt_job.id = "template_prompt_job_123"
        self.welcome_job = MagicMock()
        self.welcome_job.id = "welcome_job_123"

        # Set up side_effect to return different jobs on successive calls
        self.mock_job_manager.create_job.side_effect = [
            self.template_prompt_job,
            self.welcome_job,
        ]

        # Create training mocks
        self.mock_train = AsyncMock(return_value=MagicMock())
        self.mock_link_model = AsyncMock()
        self.mock_load = AsyncMock(return_value=None)
        self.mock_update = MagicMock()

        # Mock history store
        self.mock_history_store = MagicMock()
        self.mock_history_store.append = AsyncMock()

        # Mock llm_service with history_store property
        mock_llm_service = MagicMock()
        mock_llm_service.history_store = self.mock_history_store

        # Patch llm_service
        monkeypatch.setattr("rasa.builder.llm_service.llm_service", mock_llm_service)
        monkeypatch.setattr("rasa.builder.jobs.llm_service", mock_llm_service)

        # Apply all mocks
        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", self.mock_push_event
        )
        monkeypatch.setattr("rasa.builder.jobs.job_manager", self.mock_job_manager)
        monkeypatch.setattr(
            "rasa.builder.job_helpers.train_and_load_agent", self.mock_train
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.link_model_to_commit", self.mock_link_model
        )
        monkeypatch.setattr("rasa.builder.jobs.try_load_existing_agent", self.mock_load)
        monkeypatch.setattr("rasa.builder.jobs.update_agent", self.mock_update)

    @pytest.fixture
    def mock_template_app(self, mock_app):
        mock_app.ctx.project_generator.get_training_input.return_value = Mock()
        mock_app.ctx.project_generator.project_folder = "/tmp/test_project"
        mock_app.add_task = MagicMock()

        mock_app.ctx.project_generator.init_from_template = AsyncMock(
            return_value="template_commit_sha"
        )
        mock_app.ctx.project_generator.get_bot_files.return_value = {
            "config.yml": "test"
        }
        return mock_app

    @pytest.fixture
    def mock_prompt_app(self, mock_app):
        mock_app.ctx.project_generator.get_training_input.return_value = Mock()
        mock_app.ctx.project_generator.project_folder = "/tmp/test_project"
        mock_app.add_task = MagicMock()

        mock_app.ctx.project_generator.generate_project_with_retries = AsyncMock(
            return_value="foobarsha"
        )
        return mock_app

    @staticmethod
    def _verify_welcome_message_call(mock_push_event, expected_content_snippets):
        welcome_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.copilot_welcome_message
        ]
        # Now expects 2 calls: welcome message + commit info
        assert len(welcome_calls) == 2

        commit_payload = welcome_calls[0][1]["payload"]
        assert "commit" in commit_payload
        assert "sha" in commit_payload["commit"]
        assert commit_payload["commit"]["sha"] == "test_sha"

        welcome_payload = welcome_calls[1][1]["payload"]
        assert "content" in welcome_payload
        assert "response_category" in welcome_payload
        assert "completeness" in welcome_payload
        assert welcome_payload["response_category"] == "copilot"
        assert welcome_payload["completeness"] == "complete"
        for snippet in expected_content_snippets:
            assert snippet in welcome_payload["content"]

    @staticmethod
    def _verify_done_event_sent(mock_push_event):
        done_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.done
        ]
        assert len(done_calls) == 1

    @pytest.mark.parametrize(
        "template_name,expected_snippets",
        [
            (
                ProjectTemplateName.FINANCE,
                ["Banking Agent template", "What's my current balance?"],
            ),
            (
                ProjectTemplateName.TELCO,
                ["Telecom Support Agent template", "Why is my internet slow?"],
            ),
            (
                ProjectTemplateName.BASIC,
                ["Starter Agent template", "What can you do?"],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_template_welcome_message(
        self, mock_app, mock_job, template_name, expected_snippets
    ):
        # Mock git_service.get_commit_info for jobs with commit_sha
        mock_app.ctx.project_generator.git_service.get_commit_info = AsyncMock(
            return_value={"sha": "test_sha", "message": "test commit"}
        )
        mock_job.commit_sha = "test_commit_sha"

        await run_copilot_welcome_message_job(mock_app, mock_job, template_name)
        self._verify_welcome_message_call(self.mock_push_event, expected_snippets)
        self._verify_done_event_sent(self.mock_push_event)

    @pytest.mark.asyncio
    async def test_prompt_to_bot_welcome_message(self, mock_app, mock_job):
        # Mock git_service.get_commit_info for jobs with commit_sha
        mock_app.ctx.project_generator.git_service.get_commit_info = AsyncMock(
            return_value={"sha": "test_sha", "message": "test commit"}
        )
        mock_job.commit_sha = None  # No commit_sha for prompt-based jobs

        await run_copilot_welcome_message_job(mock_app, mock_job)

        # For prompt-based jobs without commit_sha, only 1 event is sent
        welcome_calls = [
            call
            for call in self.mock_push_event.call_args_list
            if call[0][1] == JobStatus.copilot_welcome_message
        ]
        assert len(welcome_calls) == 1

        welcome_payload = welcome_calls[0][1]["payload"]
        assert "content" in welcome_payload
        assert "custom agent has been created" in welcome_payload["content"]

    @pytest.mark.asyncio
    async def test_welcome_message_persisted_to_history(self, mock_app, mock_job):
        await run_copilot_welcome_message_job(
            mock_app, mock_job, ProjectTemplateName.FINANCE
        )

        # Verify history store append was called
        self.mock_history_store.append.assert_called_once()

        # Verify the conversation key is correct
        call_args = self.mock_history_store.append.call_args
        conversation_key = call_args[0][0]
        assert conversation_key.chat_id == "default"

        # Verify the message content
        message = call_args[0][1]
        assert message.role == "copilot"
        assert len(message.content) == 1
        assert message.content[0].type == "text"
        assert "Banking Agent template" in message.content[0].text

    @pytest.mark.asyncio
    async def test_welcome_message_persisted_to_history_with_commit(
        self, mock_app, mock_job
    ):
        """Test that welcome message is persisted with commit info."""
        # Set up mock job with commit_sha
        mock_job.commit_sha = "test_commit_sha_123"

        # Mock git_service.get_commit_info
        mock_app.ctx.project_generator.git_service.get_commit_info = AsyncMock(
            return_value={
                "sha": "test_commit_sha_123",
                "message": "Initialize project from template",
                "author": "Test Author",
                "timestamp": 1234567890,
            }
        )

        await run_copilot_welcome_message_job(
            mock_app, mock_job, ProjectTemplateName.FINANCE
        )

        # Verify history store append was called
        self.mock_history_store.append.assert_called_once()

        # Verify the conversation key is correct
        call_args = self.mock_history_store.append.call_args
        conversation_key = call_args[0][0]
        assert conversation_key.chat_id == "default"

        # Verify the message content
        message = call_args[0][1]
        assert message.role == "copilot"

        # Should have 2 content blocks: text + commit
        assert len(message.content) == 2

        # Verify text content
        text_content = message.content[0]
        assert text_content.type == "text"
        assert "Banking Agent template" in text_content.text

        # Verify commit content
        commit_content = message.content[1]
        assert commit_content.type == "commit"
        assert commit_content.commit["sha"] == "test_commit_sha_123"
        assert commit_content.commit["message"] == "Initialize project from template"
        assert commit_content.commit["author"] == "Test Author"
        assert commit_content.commit["training_success"] is True

    @pytest.mark.asyncio
    async def test_template_job_creates_welcome_job(self, mock_template_app):
        job = MagicMock(spec=JobInfo)
        job.id = "test_job_id"
        job.put = AsyncMock()
        await run_template_to_bot_job(
            mock_template_app, job, ProjectTemplateName.FINANCE
        )

        assert self.mock_job_manager.create_job.call_count == 2
        assert mock_template_app.add_task.call_count == 2

        task_calls = mock_template_app.add_task.call_args_list

        first_call_args = task_calls[0][0][0]
        assert "run_copilot_template_prompt_job" in str(first_call_args)

        second_call_args = task_calls[1][0][0]
        assert "run_copilot_welcome_message_job" in str(second_call_args)

    @pytest.mark.asyncio
    async def test_prompt_job_creates_welcome_job(self, mock_prompt_app):
        job = MagicMock(spec=JobInfo)
        job.id = "test_job_id"
        job.put = AsyncMock()
        await run_prompt_to_bot_job(
            mock_prompt_app, job, "Build me a banking assistant"
        )

        self.mock_job_manager.create_job.assert_called_once()
        assert mock_prompt_app.add_task.called

    @pytest.mark.asyncio
    async def test_training_error_prevents_welcome_job(self, mock_template_app):
        job = MagicMock(spec=JobInfo)
        job.id = "test_job_id"
        job.put = AsyncMock()
        self.mock_train.side_effect = TrainingError("Training failed")
        await run_template_to_bot_job(
            mock_template_app, job, ProjectTemplateName.FINANCE
        )

        # Template prompt job should still be created (happens before training)
        # But welcome job should NOT be created (happens after successful training)
        assert self.mock_job_manager.create_job.call_count == 1
        assert mock_template_app.add_task.call_count == 1

        # Verify only the template prompt job was created
        task_calls = mock_template_app.add_task.call_args_list
        first_call_args = task_calls[0][0][0]
        assert "run_copilot_template_prompt_job" in str(first_call_args)

    @pytest.mark.asyncio
    async def test_done_event_includes_welcome_job_id(self, mock_template_app):
        job = MagicMock(spec=JobInfo)
        job.id = "test_job_id"
        job.put = AsyncMock()

        await run_template_to_bot_job(mock_template_app, job, ProjectTemplateName.BASIC)

        done_calls = [
            call
            for call in self.mock_push_event.call_args_list
            if call[1].get("status") == JobStatus.done
        ]
        assert len(done_calls) == 1
        assert (
            done_calls[0][1]["payload"]["copilot_welcome_job_id"] == "welcome_job_123"
        )


class TestCopilotTemplatePromptJob:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        # Mock push_job_status_event
        self.mock_push_event = AsyncMock()
        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", self.mock_push_event
        )

        # Mock job_manager
        self.mock_job_manager = MagicMock()
        template_prompt_job = MagicMock()
        template_prompt_job.id = "template_prompt_job_123"
        self.mock_job_manager.create_job.return_value = template_prompt_job
        self.mock_job_manager.mark_done = MagicMock()
        monkeypatch.setattr("rasa.builder.jobs.job_manager", self.mock_job_manager)

        # Mock history store
        self.mock_history_store = MagicMock()
        self.mock_history_store.append = AsyncMock()

        # Mock llm_service with history_store property
        mock_llm_service = MagicMock()
        mock_llm_service.history_store = self.mock_history_store

        # Patch llm_service
        monkeypatch.setattr("rasa.builder.llm_service.llm_service", mock_llm_service)
        monkeypatch.setattr("rasa.builder.jobs.llm_service", mock_llm_service)

    @staticmethod
    def _verify_template_prompt_call(mock_push_event, expected_content_snippets):
        template_prompt_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.copilot_template_prompt
        ]
        assert len(template_prompt_calls) == 1

        template_prompt_payload = template_prompt_calls[0][1]["payload"]
        assert "content" in template_prompt_payload
        assert "completeness" in template_prompt_payload
        assert template_prompt_payload["completeness"] == "complete"
        for snippet in expected_content_snippets:
            assert snippet in template_prompt_payload["content"]

    @staticmethod
    def _verify_done_event_sent(mock_push_event):
        done_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.done
        ]
        assert len(done_calls) == 1

    @pytest.mark.parametrize(
        "template_name,expected_snippets",
        [
            (
                ProjectTemplateName.FINANCE,
                ["banking agent", "account balances", "manage their cards"],
            ),
            (
                ProjectTemplateName.TELCO,
                ["telecom company", "network troubleshooting", "data plans"],
            ),
            (
                ProjectTemplateName.BASIC,
                ["customer support agent", "FAQs", "human handover"],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_template_prompt_message(
        self, mock_app, mock_job, template_name, expected_snippets
    ):
        await run_copilot_template_prompt_job(mock_app, mock_job, template_name)
        self._verify_template_prompt_call(self.mock_push_event, expected_snippets)
        self._verify_done_event_sent(self.mock_push_event)

    @pytest.mark.asyncio
    async def test_template_prompt_not_persisted_immediately(self, mock_app, mock_job):
        """Template prompt is streamed immediately but persisted later in job."""
        await run_copilot_template_prompt_job(
            mock_app, mock_job, ProjectTemplateName.FINANCE
        )

        # Verify history store append was NOT called in template prompt job
        # Persistence now happens in run_template_to_bot_job after initialization
        self.mock_history_store.append.assert_not_called()

    @pytest.mark.asyncio
    async def test_template_job_creates_template_prompt_job(self):
        # Setup mocks for template to bot job
        mock_app = MagicMock()
        project_generator = Mock(spec=ProjectGenerator)
        mock_app.ctx = SimpleNamespace()
        mock_app.ctx.project_generator = project_generator
        mock_app.add_task = MagicMock()

        project_generator.init_from_template = AsyncMock(return_value="test_commit_sha")
        project_generator.get_bot_files.return_value = {"config.yml": "test"}
        project_generator.get_training_input.return_value = Mock()
        project_generator.project_folder = "/tmp/test_project"
        project_generator.git_service = MagicMock()

        # Create a mock job using the mock job manager
        job = MagicMock()
        job.id = "test-job-123"
        job.put = AsyncMock()

        # Mock agent to be returned by training
        mock_agent = MagicMock()

        with patch(
            "rasa.builder.jobs.try_load_existing_agent", AsyncMock(return_value=None)
        ):
            with patch(
                "rasa.builder.job_helpers.train_and_load_agent",
                AsyncMock(return_value=mock_agent),
            ):
                with patch("rasa.builder.jobs.update_agent", MagicMock()):
                    with patch(
                        "rasa.builder.job_helpers.link_model_to_commit",
                        new_callable=AsyncMock,
                    ):
                        await run_template_to_bot_job(
                            mock_app, job, ProjectTemplateName.FINANCE
                        )

        # Verify template prompt job and welcome job were created (2 total)
        assert self.mock_job_manager.create_job.call_count == 2
        # Verify both jobs were added as tasks
        assert mock_app.add_task.call_count == 2

    @pytest.mark.asyncio
    async def test_template_prompt_persisted_in_template_job(self):
        """Test that template prompt is persisted after template initialization."""
        # Setup mocks for template to bot job
        mock_app = MagicMock()
        project_generator = Mock(spec=ProjectGenerator)
        mock_app.ctx = SimpleNamespace()
        mock_app.ctx.project_generator = project_generator
        mock_app.add_task = MagicMock()

        project_generator.init_from_template = AsyncMock(return_value="test_commit_sha")
        project_generator.get_bot_files.return_value = {"config.yml": "test"}
        project_generator.get_training_input.return_value = Mock()
        project_generator.project_folder = "/tmp/test_project"
        project_generator.git_service = MagicMock()

        job = MagicMock(spec=JobInfo)
        job.id = "test-job-789"
        job.put = AsyncMock()

        # Mock agent to be returned by training
        mock_agent = MagicMock()

        with patch(
            "rasa.builder.jobs.try_load_existing_agent", AsyncMock(return_value=None)
        ):
            with patch(
                "rasa.builder.job_helpers.train_and_load_agent",
                AsyncMock(return_value=mock_agent),
            ):
                with patch("rasa.builder.jobs.update_agent", MagicMock()):
                    with patch(
                        "rasa.builder.job_helpers.link_model_to_commit",
                        new_callable=AsyncMock,
                    ):
                        await run_template_to_bot_job(
                            mock_app, job, ProjectTemplateName.FINANCE
                        )

        # Verify history store append was called for template prompt persistence
        # Should be called once for the template prompt after template initialization
        self.mock_history_store.append.assert_called()

        # Verify the conversation key is correct
        call_args = self.mock_history_store.append.call_args
        conversation_key = call_args[0][0]
        assert conversation_key.chat_id == "default"

        # Verify the message content
        message = call_args[0][1]
        assert message.role == "user"
        assert len(message.content) == 1
        assert message.content[0].type == "text"
        assert "banking agent" in message.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_received_event_includes_template_prompt_job_id(self):
        # Setup mocks for template to bot job
        mock_app = MagicMock()
        project_generator = Mock(spec=ProjectGenerator)
        mock_app.ctx = SimpleNamespace()
        mock_app.ctx.project_generator = project_generator
        mock_app.add_task = MagicMock()

        project_generator.init_from_template = AsyncMock(return_value="test_commit_sha")
        project_generator.get_bot_files.return_value = {"config.yml": "test"}
        project_generator.get_training_input.return_value = Mock()
        project_generator.project_folder = "/tmp/test_project"
        project_generator.git_service = MagicMock()

        job = MagicMock(spec=JobInfo)
        job.id = "test-job-456"
        job.put = AsyncMock()

        # Mock agent to be returned by training
        mock_agent = MagicMock()

        with patch(
            "rasa.builder.jobs.try_load_existing_agent", AsyncMock(return_value=None)
        ):
            with patch(
                "rasa.builder.job_helpers.train_and_load_agent",
                AsyncMock(return_value=mock_agent),
            ):
                with patch("rasa.builder.jobs.update_agent", MagicMock()):
                    with patch(
                        "rasa.builder.job_helpers.link_model_to_commit",
                        new_callable=AsyncMock,
                    ):
                        await run_template_to_bot_job(
                            mock_app, job, ProjectTemplateName.BASIC
                        )

        # Find the received event call
        # The call args are positional, with job as first arg and status as second
        received_calls = [
            call
            for call in self.mock_push_event.call_args_list
            if len(call[0]) > 1 and call[0][1] == JobStatus.received
        ]
        assert len(received_calls) == 1
        # Check the payload keyword argument
        assert "payload" in received_calls[0][1]
        assert (
            received_calls[0][1]["payload"]["copilot_template_prompt_job_id"]
            == "template_prompt_job_123"
        )

    @pytest.mark.asyncio
    async def test_no_prompt_for_unknown_template(self, mock_app, mock_job):
        # Test with a template name that doesn't have a prompt
        with patch(
            "rasa.builder.jobs.load_copilot_template_prompts",
            return_value={},
        ):
            await run_copilot_template_prompt_job(
                mock_app, mock_job, ProjectTemplateName.FINANCE
            )

        # Should complete without sending template prompt
        template_prompt_calls = [
            call
            for call in self.mock_push_event.call_args_list
            if call[0][1] == JobStatus.copilot_template_prompt
        ]
        assert len(template_prompt_calls) == 0

        # But should still send done event
        self._verify_done_event_sent(self.mock_push_event)


class TestCopilotTrainingSuccessJob:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        # Create mocks for copilot training success message job
        self.mock_push_event = AsyncMock()
        self.mock_job_manager = MagicMock()

        # Create training success job mock
        training_success_job = MagicMock()
        training_success_job.id = "training_success_job_123"
        self.mock_job_manager.create_job.return_value = training_success_job

        # Create training mocks
        self.mock_train = AsyncMock(return_value=MagicMock())
        self.mock_link_model = AsyncMock()
        self.mock_load = AsyncMock(return_value=None)
        self.mock_update = MagicMock()

        # Mock history store
        self.mock_history_store = MagicMock()
        self.mock_history_store.append = AsyncMock()

        # Mock llm_service with history_store property
        mock_llm_service = MagicMock()
        mock_llm_service.history_store = self.mock_history_store

        # Patch llm_service
        monkeypatch.setattr("rasa.builder.llm_service.llm_service", mock_llm_service)
        monkeypatch.setattr("rasa.builder.jobs.llm_service", mock_llm_service)

        # Apply all mocks
        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", self.mock_push_event
        )
        monkeypatch.setattr("rasa.builder.jobs.job_manager", self.mock_job_manager)
        monkeypatch.setattr(
            "rasa.builder.job_helpers.job_manager", self.mock_job_manager
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.train_and_load_agent", self.mock_train
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.link_model_to_commit", self.mock_link_model
        )
        monkeypatch.setattr("rasa.builder.jobs.try_load_existing_agent", self.mock_load)
        monkeypatch.setattr("rasa.builder.jobs.update_agent", self.mock_update)

    @staticmethod
    def _verify_training_success_message_call(
        mock_push_event, expected_content_snippets
    ):
        training_success_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.train_success_message
        ]
        # Now expects 2 calls: message + commit info with training success
        assert len(training_success_calls) == 2

        commit_payload = training_success_calls[0][1]["payload"]
        assert "commit" in commit_payload
        assert "sha" in commit_payload["commit"]
        assert commit_payload["commit"]["sha"] == "test_sha"
        assert commit_payload["commit"]["training_success"] is True

        training_success_payload = training_success_calls[1][1]["payload"]
        assert "content" in training_success_payload
        assert "response_category" in training_success_payload
        assert "completeness" in training_success_payload
        assert training_success_payload["response_category"] == "copilot"
        assert training_success_payload["completeness"] == "complete"
        for snippet in expected_content_snippets:
            assert snippet in training_success_payload["content"]

    @staticmethod
    def _verify_done_event_sent(mock_push_event):
        done_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.done
        ]
        assert len(done_calls) == 1

    @pytest.mark.asyncio
    async def test_training_success_message(self, mock_app, mock_job):
        # Mock git_service.get_commit_info for jobs with commit_sha
        mock_app.ctx.project_generator.git_service.get_commit_info = AsyncMock(
            return_value={"sha": "test_sha", "message": "test commit"}
        )
        mock_job.commit_sha = "test_commit_sha"

        await run_copilot_training_success_job(mock_app, mock_job)

        expected_snippets = ["Your changes have been saved successfully."]
        self._verify_training_success_message_call(
            self.mock_push_event, expected_snippets
        )
        self._verify_done_event_sent(self.mock_push_event)

    @pytest.mark.asyncio
    async def test_training_success_message_persisted_to_history(
        self, mock_app, mock_job
    ):
        await run_copilot_training_success_job(mock_app, mock_job)

        # Verify history store append was called
        self.mock_history_store.append.assert_called_once()

        # Verify the conversation key is correct
        call_args = self.mock_history_store.append.call_args
        conversation_key = call_args[0][0]
        assert conversation_key.chat_id == "default"

        # Verify the message content
        message = call_args[0][1]
        assert message.role == "copilot"
        assert len(message.content) == 1
        assert message.content[0].type == "text"
        assert "Your changes have been saved successfully" in message.content[0].text

    @pytest.mark.asyncio
    async def test_training_error_prevents_training_success_job(
        self, mock_app, sample_bot_files
    ):
        """Test that training errors don't create training success job."""
        job = job_manager.create_job()

        # Mock the project generator methods - use AsyncMock
        mock_app.ctx.project_generator.replace_all_bot_files = AsyncMock(
            return_value="abc123"
        )
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        # Set up training to fail
        self.mock_train.side_effect = TrainingError("Training failed")

        with (
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
        ):
            mock_validate.return_value = None  # No validation error

            # Call run_replace_all_files_job which should handle the training error
            await run_replace_all_files_job(
                mock_app, job, sample_bot_files, DEFAULT_COMMIT_INFO
            )

        # Verify that create_job was called once for copilot error analysis,
        # but NOT for a training success job
        # mock_job_manager.create_job is called in
        # push_error_and_start_copilot_analysis for the copilot analysis job
        assert self.mock_job_manager.create_job.call_count == 1

        # Verify add_task was called once for the copilot error analysis job
        assert mock_app.add_task.call_count == 1


class TestCopilotRollbackSuccessJob:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        # Create mocks for copilot rollback success message job
        self.mock_push_event = AsyncMock()
        self.mock_job_manager = MagicMock()

        # Create rollback success job mock
        rollback_success_job = MagicMock()
        rollback_success_job.id = "rollback_success_job_123"
        self.mock_job_manager.create_job.return_value = rollback_success_job

        # Create rollback mocks
        self.mock_train = AsyncMock(return_value=MagicMock())
        self.mock_link_model = AsyncMock()
        self.mock_load = AsyncMock(return_value=None)
        self.mock_update = MagicMock()

        # Apply all mocks
        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", self.mock_push_event
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.push_job_status_event", self.mock_push_event
        )
        monkeypatch.setattr("rasa.builder.jobs.job_manager", self.mock_job_manager)
        monkeypatch.setattr(
            "rasa.builder.job_helpers.job_manager", self.mock_job_manager
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.train_and_load_agent", self.mock_train
        )
        monkeypatch.setattr(
            "rasa.builder.job_helpers.link_model_to_commit", self.mock_link_model
        )
        monkeypatch.setattr("rasa.builder.jobs.try_load_existing_agent", self.mock_load)
        monkeypatch.setattr("rasa.builder.jobs.update_agent", self.mock_update)

    @staticmethod
    def _verify_rollback_success_message_call(
        mock_push_event, expected_content_snippets
    ):
        rollback_success_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.rollback_success
        ]
        # Now expects 2 calls: message + commit info with rollback success
        assert len(rollback_success_calls) == 2

        commit_payload = rollback_success_calls[0][1]["payload"]
        assert "commit" in commit_payload
        assert "sha" in commit_payload["commit"]
        assert commit_payload["commit"]["sha"] == "test_sha"
        assert "message" in commit_payload["commit"]
        assert commit_payload["commit"]["message"] == "test commit"
        assert "author" in commit_payload["commit"]
        assert commit_payload["commit"]["author"] == "test_author"

        rollback_success_payload = rollback_success_calls[1][1]["payload"]
        assert "content" in rollback_success_payload
        assert "response_category" in rollback_success_payload
        assert "completeness" in rollback_success_payload
        assert rollback_success_payload["response_category"] == "copilot"
        assert rollback_success_payload["completeness"] == "complete"
        for snippet in expected_content_snippets:
            assert snippet in rollback_success_payload["content"]

    @staticmethod
    def _verify_done_event_sent(mock_push_event):
        done_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.done
        ]
        assert len(done_calls) == 1

    @pytest.mark.asyncio
    async def test_rollback_success_message(self, mock_app, mock_job):
        # Mock git_service.get_commit_info for jobs with commit_sha
        mock_app.ctx.project_generator.git_service.get_commit_info = AsyncMock(
            return_value={
                "sha": "test_sha",
                "message": "test commit",
                "author": "test_author",
            }
        )
        mock_job.commit_sha = "test_commit_sha"

        await run_copilot_go_back_in_time_success_job(
            mock_app, mock_job, "rollback_success_response", JobStatus.rollback_success
        )

        expected_snippets = ["I've restored your agent to the previous version."]
        self._verify_rollback_success_message_call(
            self.mock_push_event, expected_snippets
        )
        self._verify_done_event_sent(self.mock_push_event)

    @pytest.mark.asyncio
    async def test_rollback_error_prevents_rollback_success_job(self, mock_app):
        """Test that rollback errors don't create rollback success job."""
        from rasa.builder.jobs import run_rollback_job

        job = job_manager.create_job()
        commit_sha = "test_commit_sha"

        # Mock the git service to simulate rollback failure
        mock_app.ctx.project_generator.git_service.rollback_to_commit = AsyncMock(
            side_effect=Exception("Rollback failed")
        )
        mock_app.ctx.project_generator.project_folder = "/tmp/test_project"

        # Call run_rollback_job which should handle the rollback error
        await run_rollback_job(mock_app, job, commit_sha)

        # Verify that create_job was NOT called for a rollback success job
        # since the rollback failed
        self.mock_job_manager.create_job.assert_not_called()

        # Verify add_task was NOT called for a rollback success job
        assert mock_app.add_task.call_count == 0

        # Verify the job ended with an error status by checking the mock calls
        error_status_calls = [
            call
            for call in self.mock_push_event.call_args_list
            if call[0][1] == JobStatus.error
        ]
        assert len(error_status_calls) == 1


class TestSafeTarMembers:
    """Test _safe_tar_members function for security."""

    def test_safe_tar_members_normal_files(self, tmp_path: Path):
        # Create a test tar file with normal files
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add a normal file
            config = tarfile.TarInfo("config.yml")
            content1 = b"version: '3.1'"
            config.size = len(content1)
            tar.addfile(config, fileobj=io.BytesIO(content1))

            # Add a file in subdirectory
            nlu = tarfile.TarInfo("data/nlu.yml")
            content2 = b"nlu: []"
            nlu.size = len(content2)
            tar.addfile(nlu, fileobj=io.BytesIO(content2))

        # Test extraction
        with tarfile.open(tar_path, "r:gz") as tar:
            safe_members = _safe_tar_members(tar, tmp_path)

        assert len(safe_members) == 2
        assert safe_members[0].name == "config.yml"
        assert safe_members[1].name == "data/nlu.yml"

    def test_safe_tar_members_blocks_absolute_paths(self, tmp_path: Path):
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add file with absolute path
            info = tarfile.TarInfo("/etc/passwd")
            info.size = 10
            tar.addfile(info, fileobj=None)

        with tarfile.open(tar_path, "r:gz") as tar:
            safe_members = _safe_tar_members(tar, tmp_path)

        assert len(safe_members) == 0

    def test_safe_tar_members_blocks_path_traversal(self, tmp_path: Path):
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add file with path traversal
            info = tarfile.TarInfo("../../../etc/passwd")
            info.size = 10
            tar.addfile(info, fileobj=None)

        with tarfile.open(tar_path, "r:gz") as tar:
            safe_members = _safe_tar_members(tar, tmp_path)

        assert len(safe_members) == 0

    def test_safe_tar_members_blocks_symlinks(self, tmp_path: Path):
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add symbolic link
            info = tarfile.TarInfo("symlink")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info, fileobj=None)

        with tarfile.open(tar_path, "r:gz") as tar:
            safe_members = _safe_tar_members(tar, tmp_path)

        assert len(safe_members) == 0

    def test_safe_tar_members_blocks_hardlinks(self, tmp_path: Path):
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add hard link
            info = tarfile.TarInfo("hardlink")
            info.type = tarfile.LNKTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info, fileobj=None)

        with tarfile.open(tar_path, "r:gz") as tar:
            safe_members = _safe_tar_members(tar, tmp_path)

        assert len(safe_members) == 0


class TestBackupToBotJob:
    """Test run_backup_to_bot_job function."""

    @pytest.fixture(autouse=True)
    def setup_backup_job_mocks(self, monkeypatch):
        """Setup common mocks for backup-to-bot job tests."""
        # Create all the common mocks
        self.mock_push_event = AsyncMock()
        self.mock_job_manager = MagicMock()
        self.mock_load_or_train_agent = AsyncMock()
        self.mock_update_agent = MagicMock()
        self.mock_init_endpoints = MagicMock()
        self.mock_download_backup = AsyncMock()

        # Apply the monkeypatches
        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", self.mock_push_event
        )
        monkeypatch.setattr("rasa.builder.jobs.job_manager", self.mock_job_manager)
        monkeypatch.setattr(
            "rasa.builder.jobs.load_or_train_agent_for_commit",
            self.mock_load_or_train_agent,
        )
        monkeypatch.setattr("rasa.builder.jobs.update_agent", self.mock_update_agent)
        monkeypatch.setattr(
            "rasa.builder.jobs.download_backup_from_url", self.mock_download_backup
        )

    @pytest.fixture
    def mock_project_generator(self) -> MagicMock:
        mock_pg = MagicMock()
        mock_pg.project_folder = "/tmp/test_project"
        mock_pg.get_training_input.return_value = MagicMock()
        mock_pg.get_bot_files.return_value = {"config.yml": "version: '3.1'"}
        return mock_pg

    @pytest.fixture
    def mock_job(self) -> MagicMock:
        mock_job = MagicMock()
        mock_job.id = "test-job-123"
        mock_job.put = AsyncMock()
        return mock_job

    @staticmethod
    def create_test_backup_file(files: Dict[str, str]) -> str:
        temp_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        with tarfile.open(temp_file_path, "w:gz") as tar:
            for filename, content in files.items():
                info = tarfile.TarInfo(filename)
                info.size = len(content.encode("utf-8"))
                tar.addfile(info, fileobj=io.BytesIO(content.encode("utf-8")))

        return temp_file_path

    @staticmethod
    def create_test_backup_file_with_git(
        files: Dict[str, str], git_dir_path: Path
    ) -> str:
        """Create a test backup file with actual .git directory structure.

        Args:
            files: Dictionary of file paths to content
            git_dir_path: Path to an actual .git directory to include

        Returns:
            Path to the created backup file
        """
        temp_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        with tarfile.open(temp_file_path, "w:gz") as tar:
            # Add regular files
            for filename, content in files.items():
                info = tarfile.TarInfo(filename)
                info.size = len(content.encode("utf-8"))
                tar.addfile(info, fileobj=io.BytesIO(content.encode("utf-8")))

            # Add .git directory if it exists
            if git_dir_path.exists() and git_dir_path.is_dir():
                tar.add(git_dir_path, arcname=".git", recursive=True)

        return temp_file_path

    async def test_backup_to_bot_job_with_existing_model(
        self,
        mock_project_generator,
        mock_job,
        tmp_path: Path,
    ):
        # Setup
        mock_project_generator.project_folder = str(tmp_path)
        mock_project_generator.git_service.get_current_commit_sha = AsyncMock(
            return_value="commit123"
        )
        mock_app = MagicMock()
        mock_app.ctx.project_generator = mock_project_generator

        # Mock load_or_train_agent to return an agent (simulates existing model)
        mock_agent = MagicMock()
        self.mock_load_or_train_agent.return_value = mock_agent
        mock_endpoints = MagicMock()
        self.mock_init_endpoints.return_value.endpoints = mock_endpoints

        # Create test backup file with model file
        backup_file_path = self.create_test_backup_file(
            {
                "config.yml": "version: '3.1'",
                "domain.yml": "version: '3.1'",
                "models/model.tar.gz": "fake model data",
            }
        )

        try:
            # Mock the download function to return our test file
            self.mock_download_backup.return_value = backup_file_path

            presigned_url = "https://s3.amazonaws.com/bucket/path?signature=test"

            # Execute
            await run_backup_to_bot_job(mock_app, mock_job, presigned_url)
        finally:
            # Clean up test backup file
            try:
                Path(backup_file_path).unlink(missing_ok=True)
            except Exception:
                pass

        # Should load or train agent
        self.mock_load_or_train_agent.assert_called_once()
        self.mock_update_agent.assert_called_once_with(mock_agent, mock_app)

        # Check job events
        event_calls = self.mock_push_event.call_args_list
        statuses = [call[0][1] for call in event_calls]
        assert JobStatus.received in statuses
        assert JobStatus.generating in statuses
        assert JobStatus.generation_success in statuses
        assert JobStatus.done in statuses

        # Should NOT have training events since model exists
        # assert JobStatus.training not in statuses

    async def test_backup_to_bot_job_without_model(
        self,
        mock_project_generator,
        mock_job,
        tmp_path: Path,
    ):
        # Setup
        mock_project_generator.project_folder = str(tmp_path)
        mock_project_generator.git_service.get_current_commit_sha = AsyncMock(
            return_value="commit123"
        )
        mock_app = MagicMock()
        mock_app.ctx.project_generator = mock_project_generator

        # Mock load_or_train_agent to return an agent (simulates training new model)
        mock_agent = MagicMock()
        self.mock_load_or_train_agent.return_value = mock_agent

        # Create test backup file
        backup_file_path = self.create_test_backup_file(
            {"config.yml": "version: '3.1'", "domain.yml": "version: '3.1'"}
        )

        try:
            # Mock the download function to return our test file
            self.mock_download_backup.return_value = backup_file_path

            presigned_url = "https://s3.amazonaws.com/bucket/path?signature=test"

            # Execute
            await run_backup_to_bot_job(mock_app, mock_job, presigned_url)
        finally:
            # Clean up test backup file
            try:
                Path(backup_file_path).unlink(missing_ok=True)
            except Exception:
                pass

        # Verify
        # Should load or train agent
        self.mock_load_or_train_agent.assert_called_once()
        self.mock_update_agent.assert_called_once_with(mock_agent, mock_app)

        # Check job events
        event_calls = self.mock_push_event.call_args_list
        statuses = [call[0][1] for call in event_calls]
        assert JobStatus.received in statuses
        assert JobStatus.generating in statuses
        assert JobStatus.generation_success in statuses
        # Training status is now pushed by load_or_train_agent_for_commit
        # which is mocked, so we don't see it in the events
        assert JobStatus.train_success in statuses
        assert JobStatus.done in statuses

    @pytest.mark.asyncio
    async def test_backup_to_bot_job_restores_git_directory(
        self,
        mock_project_generator: MagicMock,
        mock_job: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that .git directory is properly restored from backup."""
        # Setup project folder
        project_folder = tmp_path / "project"
        project_folder.mkdir()
        mock_project_generator.project_folder = str(project_folder)
        mock_project_generator.git_service.get_current_commit_sha = AsyncMock(
            return_value="commit123"
        )
        mock_app = MagicMock()
        mock_app.ctx.project_generator = mock_project_generator

        # Create a mock .git directory with Git structure
        git_source_dir = tmp_path / "source_git"
        git_source_dir.mkdir()
        (git_source_dir / "config").write_text("[core]\n\trepositoryformatversion = 0")
        (git_source_dir / "HEAD").write_text("ref: refs/heads/main")
        (git_source_dir / "description").write_text("Test repository")

        # Create refs structure
        refs_dir = git_source_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123def456789012345678901234567890abcd")

        # Create objects structure
        objects_dir = git_source_dir / "objects"
        objects_dir.mkdir()
        (objects_dir / "pack").mkdir()
        (objects_dir / "info").mkdir()
        (objects_dir / "info" / "packs").write_text("")

        # Mock load_or_train_agent to return an agent
        mock_agent = MagicMock()
        self.mock_load_or_train_agent.return_value = mock_agent

        # Create test backup file with .git directory
        backup_file_path = self.create_test_backup_file_with_git(
            {
                "config.yml": "version: '3.1'",
                "domain.yml": "version: '3.1'",
                "data/nlu.yml": "version: '3.1'\nnlu: []",
            },
            git_source_dir,
        )

        try:
            # Mock the download function to return our test file
            self.mock_download_backup.return_value = backup_file_path

            presigned_url = "https://s3.amazonaws.com/bucket/path?signature=test"

            # Execute
            await run_backup_to_bot_job(mock_app, mock_job, presigned_url)

            # Verify .git directory was restored
            restored_git_dir = project_folder / ".git"
            assert restored_git_dir.exists(), ".git directory should be restored"
            assert restored_git_dir.is_dir(), ".git should be a directory"

            # Verify .git directory contents
            assert (restored_git_dir / "config").exists()
            assert (restored_git_dir / "config").read_text() == (
                "[core]\n\trepositoryformatversion = 0"
            )

            assert (restored_git_dir / "HEAD").exists()
            assert (restored_git_dir / "HEAD").read_text() == "ref: refs/heads/main"

            assert (restored_git_dir / "description").exists()
            assert (restored_git_dir / "description").read_text() == "Test repository"

            # Verify refs structure
            assert (restored_git_dir / "refs" / "heads" / "main").exists()
            assert (restored_git_dir / "refs" / "heads" / "main").read_text() == (
                "abc123def456789012345678901234567890abcd"
            )

            # Verify objects structure
            assert (restored_git_dir / "objects" / "pack").exists()
            assert (restored_git_dir / "objects" / "info").exists()
            assert (restored_git_dir / "objects" / "info" / "packs").exists()

            # Verify regular files were also restored
            assert (project_folder / "config.yml").exists()
            assert (project_folder / "domain.yml").exists()
            assert (project_folder / "data" / "nlu.yml").exists()

        finally:
            # Clean up test backup file
            try:
                Path(backup_file_path).unlink(missing_ok=True)
            except Exception:
                pass
