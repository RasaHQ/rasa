"""Tests for rasa.builder.jobs module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from rasa.builder.exceptions import TrainingError, ValidationError
from rasa.builder.job_manager import JobInfo, job_manager
from rasa.builder.jobs import run_replace_all_files_job
from rasa.builder.models import JobStatus
from rasa.builder.project_generator import ProjectGenerator


@pytest.fixture
def mock_app():
    """Create a mock Sanic app for testing."""
    project_generator = Mock(spec=ProjectGenerator)
    app = SimpleNamespace()
    app.ctx = SimpleNamespace()
    app.ctx.project_generator = project_generator
    return app


@pytest.fixture
def sample_bot_files():
    """Sample bot files for testing."""
    return {
        "config.yml": "version: '3.1'\npipeline: []",
        "domain.yml": "version: '3.1'\nintents: []",
        "data/nlu.yml": "version: '3.1'\nnlu: []",
    }


class TestRunReplaceAllFilesJob:
    """Test the run_replace_all_files_job function."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, mock_app, sample_bot_files):
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
    async def test_validation_error(self, mock_app, sample_bot_files):
        """Test handling of validation errors."""
        job = job_manager.create_job()

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

        # Check job ended with validation error
        assert job.status == JobStatus.validation_error.value

    @pytest.mark.asyncio
    async def test_training_error(self, mock_app, sample_bot_files):
        """Test handling of training errors."""
        job = job_manager.create_job()

        mock_app.ctx.project_generator.replace_all_bot_files = Mock()
        mock_training_input = Mock()
        mock_app.ctx.project_generator.get_training_input.return_value = (
            mock_training_input
        )

        training_error = TrainingError("Training failed")

        with (
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

        # Check job ended with training error
        assert job.status == JobStatus.train_error.value

    @pytest.mark.asyncio
    async def test_unexpected_error(self, mock_app, sample_bot_files):
        """Test handling of unexpected errors."""
        job = job_manager.create_job()

        # Mock an unexpected error during file replacement
        mock_app.ctx.project_generator.replace_all_bot_files.side_effect = Exception(
            "Unexpected error"
        )

        await run_replace_all_files_job(mock_app, job, sample_bot_files)

        # Check job ended with error
        assert job.status == JobStatus.error.value

    @pytest.mark.asyncio
    async def test_job_status_progression(self, mock_app, sample_bot_files):
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
