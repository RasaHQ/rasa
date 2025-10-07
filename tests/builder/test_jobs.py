"""Tests for rasa.builder.jobs module."""

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
    run_copilot_training_error_analysis_job,
    run_copilot_welcome_message_job,
    run_prompt_to_bot_job,
    run_replace_all_files_job,
    run_template_to_bot_job,
)
from rasa.builder.models import JobStatus
from rasa.builder.project_generator import ProjectGenerator
from rasa.cli.scaffold import ProjectTemplateName


@pytest.fixture
def mock_app() -> MagicMock:
    """Create a mock Sanic app for testing."""
    project_generator = Mock(spec=ProjectGenerator)
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
        self, mock_app, sample_bot_files: Dict[str, Any]
    ):
        """Test successful execution of replace_all_files_job."""
        job = job_manager.create_job()

        # Mock the project generator methods
        mock_app.ctx.project_generator.replace_all_bot_files = Mock()

        # Mock training input and validation
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        # Mock successful validation and training
        with (
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.jobs.train_and_load_agent", new_callable=AsyncMock
            ) as mock_train,
            patch("rasa.builder.jobs.update_agent") as mock_update_agent,
        ):
            mock_validate.return_value = None  # No validation error
            mock_agent = Mock()
            mock_train.return_value = mock_agent

            await run_replace_all_files_job(mock_app, job, sample_bot_files)

            # Verify the flow
            mock_app.ctx.project_generator.replace_all_bot_files.assert_called_once_with(
                sample_bot_files
            )
            mock_validate.assert_called_once_with(mock_training_input.importer)
            mock_train.assert_called_once_with(mock_training_input)
            mock_update_agent.assert_called_once_with(mock_agent, mock_app)

        # Check job status
        assert job.status == JobStatus.done.value

    @pytest.mark.asyncio
    async def test_validation_error(
        self,
        mock_app: MagicMock,
        sample_bot_files: Dict[str, Any],
        job_status_tracker: tuple[list[str], Callable],
    ) -> None:
        """Test handling of validation errors."""
        job = job_manager.create_job()
        status_events, track_status_event = job_status_tracker

        mock_app.ctx.project_generator.replace_all_bot_files = Mock()
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        validation_error = ValidationError("Invalid configuration")
        validation_error.validation_logs = [
            {"log_level": "error", "message": "Error 1", "file": "domain.yml"},
            {"log_level": "error", "message": "Error 2", "file": "stories.yml"},
        ]

        with (
            patch("rasa.builder.jobs.push_job_status_event", new=track_status_event),
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch("rasa.builder.jobs.config.VALIDATION_FAIL_ON_WARNINGS", False),
        ):
            mock_validate.side_effect = validation_error

            await run_replace_all_files_job(mock_app, job, sample_bot_files)

            # Verify file replacement was called
            mock_app.ctx.project_generator.replace_all_bot_files.assert_called_once_with(
                sample_bot_files
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
    ) -> None:
        """Test handling of training errors."""
        job = job_manager.create_job()
        status_events, track_status_event = job_status_tracker

        mock_app.ctx.project_generator.replace_all_bot_files = Mock()
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        training_error = TrainingError("Training failed")

        with (
            patch("rasa.builder.jobs.push_job_status_event", new=track_status_event),
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.jobs.train_and_load_agent", new_callable=AsyncMock
            ) as mock_train,
        ):
            mock_validate.return_value = None  # No validation error
            mock_train.side_effect = training_error

            await run_replace_all_files_job(mock_app, job, sample_bot_files)

            # Verify the flow up to training
            mock_app.ctx.project_generator.replace_all_bot_files.assert_called_once_with(
                sample_bot_files
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
    ) -> None:
        """Test handling of unexpected errors."""
        job = job_manager.create_job()
        status_events, track_status_event = job_status_tracker

        # Mock an unexpected error during file replacement
        mock_app.ctx.project_generator.replace_all_bot_files.side_effect = Exception(
            "Unexpected error"
        )

        with patch("rasa.builder.jobs.push_job_status_event", new=track_status_event):
            await run_replace_all_files_job(mock_app, job, sample_bot_files)

        # Check that the job ended with error status (with copilot job ID)
        assert job.status == JobStatus.error.value

        # Verify the sequence:
        # last event should be error (with copilot job ID)
        assert len(status_events) >= 1
        assert status_events[-1] == JobStatus.error.value

    @pytest.mark.asyncio
    async def test_job_status_progression(
        self, mock_app, sample_bot_files: Dict[str, Any]
    ):
        job = job_manager.create_job()
        status_events = []

        original_put = JobInfo.put

        async def track_status(self, event):
            # status is stored in the event's data payload
            if status := (event.data or {}).get("status"):
                status_events.append(status)
            return await original_put(self, event)

        mock_app.ctx.project_generator.replace_all_bot_files = Mock()
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        with (
            patch.object(JobInfo, "put", new=track_status),
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.jobs.train_and_load_agent", new_callable=AsyncMock
            ) as mock_train,
            patch("rasa.builder.jobs.update_agent"),
        ):
            mock_validate.return_value = None
            mock_train.return_value = Mock()

            await run_replace_all_files_job(mock_app, job, sample_bot_files)

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
    @patch("rasa.builder.jobs.update_agent")
    @patch("rasa.builder.jobs.train_and_load_agent", new_callable=AsyncMock)
    @patch("rasa.builder.jobs.validate_project", new_callable=AsyncMock)
    @patch(
        "rasa.builder.jobs.push_error_and_start_copilot_analysis",
        new_callable=AsyncMock,
    )
    async def test_run_update_files_job_creates_copilot_analysis_job(
        self,
        mock_push_error_and_start_copilot: AsyncMock,
        mock_validate: AsyncMock,
        mock_train: AsyncMock,
        mock_update_agent: MagicMock,
        mock_app: MagicMock,
        mock_job: JobInfo,
        sample_bot_files: Dict[str, Any],
        raised_exception: Optional[Exception],
        should_create_copilot_job: bool,
    ) -> None:
        """Test run_update_files_job with different scenarios."""
        # Given
        mock_update_agent.return_value = None

        if raised_exception is None:
            mock_validate.return_value = None
            mock_train.return_value = None
        else:
            mock_validate.side_effect = raised_exception

        # When
        await run_replace_all_files_job(mock_app, mock_job, sample_bot_files)

        # Then
        if should_create_copilot_job:
            mock_push_error_and_start_copilot.assert_called_once()
            call_args = mock_push_error_and_start_copilot.call_args
            assert call_args[0][0] == mock_app
            assert call_args[0][1] == mock_job
            assert call_args[0][2] is not None  # error_message
            assert call_args[0][3] is not None  # job_status
            assert call_args[0][4] == sample_bot_files
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
    @patch("rasa.builder.jobs.llm_service.instantiate_handler")
    @patch("rasa.builder.jobs.llm_service.instantiate_copilot")
    @patch("rasa.builder.jobs.push_job_status_event", new_callable=AsyncMock)
    async def test_run_copilot_training_error_analysis_job_success(
        self,
        mock_push_event: AsyncMock,
        mock_instantiate_copilot: MagicMock,
        mock_instantiate_handler: MagicMock,
        mock_app: MagicMock,
        mock_job: JobInfo,
        sample_bot_files: Dict[str, Any],
        mock_error_message: str,
    ) -> None:
        """Test successful copilot analysis job with content validation."""
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
    @patch("rasa.builder.jobs.llm_service.instantiate_copilot")
    @patch("rasa.builder.jobs.push_job_status_event", new_callable=AsyncMock)
    async def test_run_copilot_training_error_analysis_job_error(
        self,
        mock_push_event: AsyncMock,
        mock_instantiate_copilot: MagicMock,
        mock_app: MagicMock,
        mock_job: JobInfo,
        sample_bot_files: Dict[str, Any],
        mock_error_message: str,
    ) -> None:
        """Test copilot analysis job with error."""
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

        # Create welcome job mock
        welcome_job = MagicMock()
        welcome_job.id = "welcome_job_123"
        self.mock_job_manager.create_job.return_value = welcome_job

        # Create training mocks
        self.mock_train = AsyncMock(return_value=MagicMock())
        self.mock_load = AsyncMock(return_value=None)
        self.mock_update = MagicMock()

        # Apply all mocks
        monkeypatch.setattr(
            "rasa.builder.jobs.push_job_status_event", self.mock_push_event
        )
        monkeypatch.setattr("rasa.builder.jobs.job_manager", self.mock_job_manager)
        monkeypatch.setattr("rasa.builder.jobs.train_and_load_agent", self.mock_train)
        monkeypatch.setattr("rasa.builder.jobs.try_load_existing_agent", self.mock_load)
        monkeypatch.setattr("rasa.builder.jobs.update_agent", self.mock_update)

    @pytest.fixture
    def mock_template_app(self, mock_app):
        mock_app.ctx.project_generator.get_training_input.return_value = Mock()
        mock_app.ctx.project_generator.project_folder = "/tmp/test_project"
        mock_app.add_task = MagicMock()

        mock_app.ctx.project_generator.init_from_template = AsyncMock()
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
            return_value={"config.yml": "test"}
        )
        return mock_app

    @staticmethod
    def _verify_welcome_message_call(mock_push_event, expected_content_snippets):
        welcome_calls = [
            call
            for call in mock_push_event.call_args_list
            if call[0][1] == JobStatus.copilot_welcome_message
        ]
        assert len(welcome_calls) == 1

        welcome_payload = welcome_calls[0][1]["payload"]
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
        await run_copilot_welcome_message_job(mock_app, mock_job, template_name)
        self._verify_welcome_message_call(self.mock_push_event, expected_snippets)
        self._verify_done_event_sent(self.mock_push_event)

    @pytest.mark.asyncio
    async def test_prompt_to_bot_welcome_message(self, mock_app, mock_job):
        await run_copilot_welcome_message_job(mock_app, mock_job)
        self._verify_welcome_message_call(
            self.mock_push_event, ["custom agent has been created"]
        )

    @pytest.mark.asyncio
    async def test_template_job_creates_welcome_job(self, mock_template_app):
        job = job_manager.create_job()
        await run_template_to_bot_job(
            mock_template_app, job, ProjectTemplateName.FINANCE
        )

        self.mock_job_manager.create_job.assert_called_once()
        assert mock_template_app.add_task.called

    @pytest.mark.asyncio
    async def test_prompt_job_creates_welcome_job(self, mock_prompt_app):
        job = job_manager.create_job()
        await run_prompt_to_bot_job(
            mock_prompt_app, job, "Build me a banking assistant"
        )

        self.mock_job_manager.create_job.assert_called_once()
        assert mock_prompt_app.add_task.called

    @pytest.mark.asyncio
    async def test_training_error_prevents_welcome_job(self, mock_template_app):
        job = job_manager.create_job()
        self.mock_train.side_effect = TrainingError("Training failed")
        await run_template_to_bot_job(
            mock_template_app, job, ProjectTemplateName.FINANCE
        )

        self.mock_job_manager.create_job.assert_not_called()
        mock_template_app.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_done_event_includes_welcome_job_id(self, mock_template_app):
        job = MagicMock(spec=JobInfo)
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
