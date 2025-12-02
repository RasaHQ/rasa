"""Tests for Git integration with the job system."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from sanic import Sanic

from rasa.builder.git_service import GitService
from rasa.builder.job_manager import job_manager
from rasa.builder.jobs import (
    run_replace_all_files_job,
    run_rollback_job,
)
from rasa.builder.models import JobStatus
from rasa.builder.project_generator.project_generator import (
    DEFAULT_COMMIT_INFO,
    ProjectGenerator,
)


class TestGitIntegration:
    """Test Git integration with existing job system."""

    @pytest.fixture
    def temp_project_dir(self, tmp_path: Path) -> Path:
        """Create a temporary project directory."""
        return tmp_path / "project"

    @pytest.fixture
    def mock_app(self, temp_project_dir: Path) -> Sanic:
        """Create a mock Sanic app with project generator."""
        app = MagicMock(spec=Sanic)
        app.ctx.project_generator = ProjectGenerator(str(temp_project_dir))
        return app

    @pytest.fixture
    def sample_bot_files(self) -> dict:
        """Sample bot files for testing."""
        return {
            "domain.yml": "version: '3.1'\\nintents:\\n  - greet",
            "config.yml": "version: '3.1'\\npipeline: []",
            "data/nlu.yml": "version: '3.1'\\nnlu: []",
        }

    @pytest.mark.asyncio
    async def test_replace_all_files_job_creates_commit(
        self, mock_app: Sanic, sample_bot_files: dict
    ) -> None:
        """Test that replace_all_files_job creates a Git commit before training."""
        job = job_manager.create_job()

        with (
            patch("rasa.builder.jobs.push_job_status_event", new_callable=AsyncMock),
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.job_helpers.train_and_load_agent",
                new_callable=AsyncMock,
            ) as mock_train,
            patch(
                "rasa.builder.job_helpers.link_model_to_commit",
                new_callable=AsyncMock,
            ),
            patch("rasa.builder.jobs.update_agent"),
            patch.object(
                mock_app.ctx.project_generator,
                "replace_all_bot_files",
                new_callable=AsyncMock,
            ) as mock_replace_files,
            patch.object(
                mock_app.ctx.project_generator,
                "get_training_input",
            ) as mock_get_training_input,
            patch.object(
                mock_app.ctx.project_generator.git_service,
                "commit_changes",
                new_callable=AsyncMock,
            ) as mock_commit,
            patch.object(
                mock_app.ctx.project_generator.git_service,
                "create_model_tag",
                new_callable=AsyncMock,
            ),
        ):
            mock_validate.return_value = None
            mock_train.return_value = MagicMock()
            mock_commit.return_value = "abc123def456"
            mock_replace_files.return_value = "abc123def456"
            mock_get_training_input.return_value = Mock()

            await run_replace_all_files_job(
                mock_app, job, sample_bot_files, DEFAULT_COMMIT_INFO
            )

            # Verify replace_all_bot_files was called with correct parameters
            mock_app.ctx.project_generator.replace_all_bot_files.assert_called_once_with(
                sample_bot_files, DEFAULT_COMMIT_INFO
            )

            # Verify training was called (only takes training_input)
            mock_train.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_job_follows_existing_patterns(
        self, mock_app: Sanic
    ) -> None:
        """Test rollback job follows existing job patterns."""
        job = job_manager.create_job()
        commit_sha = "abc123def456"

        with (
            patch(
                "rasa.builder.job_helpers.push_job_status_event", new_callable=AsyncMock
            ) as mock_push_event,
            patch(
                "rasa.builder.jobs.push_job_status_event", new_callable=AsyncMock
            ) as mock_push_event_jobs,
            patch("rasa.builder.jobs.update_agent"),
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch.object(
                mock_app.ctx.project_generator,
                "get_training_input",
            ) as mock_get_training_input,
            patch.object(
                mock_app.ctx.project_generator.git_service,
                "rollback_to_commit",
                new_callable=AsyncMock,
            ) as mock_rollback,
            patch.object(
                mock_app.ctx.project_generator.git_service,
                "load_model_for_commit",
                new_callable=AsyncMock,
            ) as mock_load_model,
            patch(
                "rasa.builder.job_helpers.train_and_load_agent",
                new_callable=AsyncMock,
            ) as mock_train_and_load,
            patch.object(
                mock_app.ctx.project_generator.git_service,
                "get_commit_info",
                new_callable=AsyncMock,
            ) as mock_get_commit_info,
            patch("rasa.builder.jobs.job_manager.create_job") as mock_create_job,
            patch("rasa.builder.jobs.job_manager.mark_done"),
            patch.object(
                mock_app,
                "add_task",
            ) as mock_add_task,
            patch(
                "rasa.builder.jobs.run_copilot_rollback_message_job",
                new_callable=AsyncMock,
            ),
        ):
            mock_agent = MagicMock()
            mock_validate.return_value = None
            mock_train_and_load.return_value = mock_agent
            mock_get_training_input.return_value = Mock()
            mock_rollback.return_value = "new_rollback_commit_sha"
            mock_load_model.return_value = None  # Force retraining
            mock_get_commit_info.return_value = {
                "sha": "new_rollback_commit_sha",
                "message": "Rollback commit",
                "author": "test_user",
            }
            copilot_job = MagicMock()
            copilot_job.id = "copilot_job_123"
            mock_create_job.return_value = copilot_job

            await run_rollback_job(mock_app, job, commit_sha)

            # Verify rollback flow
            mock_rollback.assert_called_once_with(commit_sha)
            mock_load_model.assert_called_once_with(commit_sha)
            mock_train_and_load.assert_called_once()
            mock_create_job.assert_called_once_with(
                commit_sha="new_rollback_commit_sha"
            )
            mock_add_task.assert_called_once()

            # Verify job status events - combine calls from both mocks
            status_calls_jobs = [
                call[0][1] for call in mock_push_event_jobs.call_args_list
            ]
            status_calls_helpers = [
                call[0][1] for call in mock_push_event.call_args_list
            ]

            # The events come from jobs module
            assert JobStatus.received in status_calls_jobs
            assert JobStatus.validating in status_calls_jobs
            assert JobStatus.validation_success in status_calls_jobs
            assert JobStatus.done in status_calls_jobs

            # The rollback and training events come from helpers module
            assert JobStatus.rolling_back in status_calls_helpers
            assert JobStatus.rollback_success in status_calls_helpers
            assert JobStatus.training in status_calls_helpers

            # Verify all expected statuses are present
            all_statuses = status_calls_jobs + status_calls_helpers
            expected_statuses = [
                JobStatus.received,
                JobStatus.rolling_back,
                JobStatus.rollback_success,
                JobStatus.validating,
                JobStatus.validation_success,
                JobStatus.training,
                JobStatus.train_success,
                JobStatus.done,
            ]
            for status in expected_statuses:
                assert (
                    status in all_statuses
                ), f"Expected status {status} not found in {all_statuses}"

    @pytest.mark.asyncio
    async def test_rollback_job_uses_existing_model(self, mock_app: Sanic) -> None:
        """Test rollback job uses existing model when available."""
        job = job_manager.create_job()
        commit_sha = "abc123def456"
        existing_agent = MagicMock()

        with (
            patch("rasa.builder.jobs.push_job_status_event", new_callable=AsyncMock),
            patch(
                "rasa.builder.job_helpers.push_job_status_event", new_callable=AsyncMock
            ),
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.jobs.load_or_train_agent_for_commit",
                new_callable=AsyncMock,
            ) as mock_load_or_train,
            patch("rasa.builder.jobs.update_agent"),
            patch.object(
                mock_app.ctx.project_generator,
                "get_training_input",
            ) as mock_get_training_input,
            patch.object(
                mock_app.ctx.project_generator.git_service,
                "rollback_to_commit",
                new_callable=AsyncMock,
            ),
            patch("rasa.builder.jobs.job_manager.create_job") as mock_create_job,
            patch("rasa.builder.jobs.job_manager.mark_done"),
            patch.object(mock_app, "add_task"),
            patch(
                "rasa.builder.jobs.run_copilot_rollback_message_job",
                new_callable=AsyncMock,
            ),
        ):
            mock_validate.return_value = None
            mock_load_or_train.return_value = existing_agent
            mock_get_training_input.return_value = Mock()
            copilot_job = MagicMock()
            copilot_job.id = "copilot_job_123"
            mock_create_job.return_value = copilot_job

            await run_rollback_job(mock_app, job, commit_sha)

            # Should validate project
            mock_validate.assert_called_once()

            # Should load or train agent
            mock_load_or_train.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_for_commit_existing_metadata(
        self, temp_project_dir: Path
    ) -> None:
        """Test loading model for commit with existing metadata."""
        temp_project_dir.mkdir()
        models_dir = temp_project_dir / "models"
        models_dir.mkdir(parents=True)

        # Create model file
        commit_sha = "abc123def456"
        model_path = models_dir / "model.tar.gz"
        model_path.write_text("dummy model")

        with (
            patch(
                "rasa.builder.training_service.try_load_existing_agent",
                new_callable=AsyncMock,
            ) as mock_load,
            patch.object(
                GitService,
                "get_model_for_commit",
                new_callable=AsyncMock,
            ) as mock_get_model,
        ):
            mock_agent = MagicMock()
            mock_load.return_value = mock_agent
            mock_get_model.return_value = "model/model"

            git_service = GitService(str(temp_project_dir))
            result = await git_service.load_model_for_commit(commit_sha)

            assert result == mock_agent
            mock_get_model.assert_called_once_with(commit_sha)
            mock_load.assert_called_once_with(temp_project_dir, model_path)

    @pytest.mark.asyncio
    async def test_load_model_for_commit_no_metadata(
        self, temp_project_dir: Path
    ) -> None:
        """Test loading model for commit with no metadata."""
        temp_project_dir.mkdir()
        commit_sha = "abc123def456"

        git_service = GitService(str(temp_project_dir))
        result = await git_service.load_model_for_commit(commit_sha)

        assert result is None

    @pytest.mark.asyncio
    async def test_load_model_for_commit_missing_model_file(
        self, temp_project_dir: Path
    ) -> None:
        """Test loading model for commit when model file is missing."""
        temp_project_dir.mkdir()
        metadata_dir = temp_project_dir / "models" / ".metadata"
        metadata_dir.mkdir(parents=True)

        commit_sha = "abc123def456"
        metadata = {
            "commit_sha": commit_sha,
            "model_path": "/nonexistent/model.tar.gz",
            "created_at": 1640995200,
            "author": "user",
        }
        metadata_file = metadata_dir / f"{commit_sha}.json"
        metadata_file.write_text(json.dumps(metadata))

        git_service = GitService(str(temp_project_dir))
        result = await git_service.load_model_for_commit(commit_sha)

        assert result is None


class TestProjectGeneratorGitIntegration:
    """Test ProjectGenerator integration with GitService."""

    @pytest.fixture
    def temp_project_dir(self, tmp_path: Path) -> Path:
        """Create a temporary project directory."""
        return tmp_path / "project"

    @pytest.fixture
    def project_generator(self, temp_project_dir: Path) -> ProjectGenerator:
        """Create ProjectGenerator instance."""
        temp_project_dir.mkdir()
        return ProjectGenerator(str(temp_project_dir))

    def test_project_generator_has_git_service(
        self, project_generator: ProjectGenerator
    ) -> None:
        """Test that ProjectGenerator has GitService instance."""
        assert isinstance(project_generator.git_service, GitService)
        assert (
            project_generator.git_service.project_folder
            == project_generator.project_folder
        )

    @pytest.mark.asyncio
    async def test_project_generator_uses_git_service_for_branches(
        self, project_generator: ProjectGenerator
    ) -> None:
        """Test that ProjectGenerator uses GitService for branch operations."""
        with patch.object(
            project_generator.git_service, "get_current_branch", return_value="main"
        ) as mock_branch:
            branch = await project_generator._get_current_branch()
            assert branch == "main"
            mock_branch.assert_called_once()

    @pytest.mark.asyncio
    async def test_project_generator_uses_git_service_for_checkout(
        self, project_generator: ProjectGenerator
    ) -> None:
        """Test that ProjectGenerator uses GitService for checkout operations."""
        with patch.object(
            project_generator.git_service, "checkout_branch"
        ) as mock_checkout:
            await project_generator.checkout_branch("feature-branch", True)
            mock_checkout.assert_called_once_with("feature-branch", True)
