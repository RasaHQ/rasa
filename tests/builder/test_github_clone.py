"""Tests for GitHub repository cloning functionality."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import ValidationError
from sanic import Request

from rasa.builder.github_clone import (
    GitCloneError,
    _cleanup_temp_directory,
    _move_files_to_target,
    clone_public_repo,
)
from rasa.builder.jobs import run_github_to_bot_job
from rasa.builder.models import GitHubToBotRequest
from rasa.builder.service import handle_github_to_bot


class TestGitHubToBotRequestValidation:
    """Test validation logic for GitHubToBotRequest model."""

    def test_valid_github_url(self) -> None:
        """Test that valid GitHub URLs are accepted."""
        valid_urls = [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo.git",
            "https://github.com/owner-name/repo-name",
            "https://github.com/owner.name/repo.name",
            "https://github.com/owner_name/repo_name",
        ]
        for url in valid_urls:
            request = GitHubToBotRequest(repo_url=url, branch=None)
            assert request.repo_url == url

    def test_github_url_with_trailing_slash(self) -> None:
        """Test that trailing slashes are stripped from URLs."""
        request = GitHubToBotRequest(
            repo_url="https://github.com/owner/repo/", branch=None
        )
        assert request.repo_url == "https://github.com/owner/repo"

        request = GitHubToBotRequest(
            repo_url="https://github.com/owner/repo.git/", branch=None
        )
        assert request.repo_url == "https://github.com/owner/repo.git"

    def test_github_url_with_whitespace(self) -> None:
        """Test that whitespace is stripped from URLs."""
        request = GitHubToBotRequest(
            repo_url="  https://github.com/owner/repo  ", branch=None
        )
        assert request.repo_url == "https://github.com/owner/repo"

    def test_invalid_non_https_url(self) -> None:
        """Test that non-HTTPS URLs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GitHubToBotRequest(repo_url="http://github.com/owner/repo", branch=None)
        assert "HTTPS" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GitHubToBotRequest(repo_url="git@github.com:owner/repo.git", branch=None)
        assert "HTTPS" in str(exc_info.value)

    def test_invalid_non_github_url(self) -> None:
        """Test that non-GitHub URLs are rejected (SSRF prevention)."""
        invalid_urls = [
            "https://gitlab.com/owner/repo",
            "https://evil.com/malicious",
            "https://internal-git.company.com/repo",
            "https://github.evil.com/owner/repo",
            "https://sub.github.com/owner/repo",
        ]
        for url in invalid_urls:
            with pytest.raises(ValidationError) as exc_info:
                GitHubToBotRequest(repo_url=url, branch=None)
            assert "GitHub URL" in str(exc_info.value)

    def test_invalid_github_url_format(self) -> None:
        """Test that malformed GitHub URLs are rejected."""
        invalid_urls = [
            "https://github.com/",
            "https://github.com/owner",
            "https://github.com/owner/",
            "https://github.com/owner/repo/extra/path",
        ]
        for url in invalid_urls:
            with pytest.raises(ValidationError) as exc_info:
                GitHubToBotRequest(repo_url=url, branch=None)
            assert "GitHub URL" in str(exc_info.value)

    def test_valid_branch_names(self) -> None:
        """Test that valid branch names are accepted."""
        valid_branches = [
            "main",
            "develop",
            "feature/new-feature",
            "bugfix/fix-123",
            "release-1.0",
            "hotfix_urgent",
            "feature.name",
        ]
        for branch in valid_branches:
            request = GitHubToBotRequest(
                repo_url="https://github.com/owner/repo", branch=branch
            )
            assert request.branch == branch

    def test_none_branch(self) -> None:
        """Test that None branch is accepted (uses default branch)."""
        request = GitHubToBotRequest(
            repo_url="https://github.com/owner/repo", branch=None
        )
        assert request.branch is None

    def test_empty_branch_normalized_to_none(self) -> None:
        """Test that empty/whitespace branch is normalized to None."""
        request = GitHubToBotRequest(
            repo_url="https://github.com/owner/repo", branch="  "
        )
        assert request.branch is None

    def test_invalid_branch_with_dangerous_characters(self) -> None:
        """Test that branch names with dangerous characters are rejected."""
        invalid_branches = [
            "feature/../../../etc/passwd",
            "branch~1",
            "branch^HEAD",
            "branch:colon",
            "branch?question",
            "branch*star",
            "branch[bracket",
            "branch\\backslash",
            "branch@{",
            "branch with spaces",
        ]
        for branch in invalid_branches:
            with pytest.raises(ValidationError) as exc_info:
                GitHubToBotRequest(
                    repo_url="https://github.com/owner/repo", branch=branch
                )
            # Should have a validation error about invalid characters
            assert exc_info.value.errors()[0]["type"] == "value_error"

    def test_invalid_branch_starting_with_dot(self) -> None:
        """Test that branch names starting with dot are rejected."""
        with pytest.raises(ValidationError):
            GitHubToBotRequest(
                repo_url="https://github.com/owner/repo", branch=".hidden"
            )

    def test_invalid_branch_ending_with_lock(self) -> None:
        """Test that branch names ending with .lock are rejected."""
        with pytest.raises(ValidationError):
            GitHubToBotRequest(
                repo_url="https://github.com/owner/repo", branch="branch.lock"
            )

    def test_invalid_branch_ending_with_slash(self) -> None:
        """Test that branch names ending with slash are rejected."""
        with pytest.raises(ValidationError):
            GitHubToBotRequest(
                repo_url="https://github.com/owner/repo", branch="feature/"
            )


class TestGitHubCloneHelpers:
    """Test helper functions for GitHub cloning."""

    def test_cleanup_temp_directory(self, tmp_path: Path) -> None:
        """Test cleanup of temporary directory."""
        temp_dir = tmp_path / "temp_clone"
        temp_dir.mkdir()
        (temp_dir / "file.txt").write_text("test")

        _cleanup_temp_directory(temp_dir)
        assert not temp_dir.exists()

    def test_cleanup_temp_directory_nonexistent(self, tmp_path: Path) -> None:
        """Test cleanup of nonexistent directory doesn't raise."""
        temp_dir = tmp_path / "nonexistent"
        _cleanup_temp_directory(temp_dir)  # Should not raise

    def test_move_files_to_target(self, tmp_path: Path) -> None:
        """Test moving files from temp to target directory."""
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        target.mkdir()

        # Create some files in source
        (source / "file1.txt").write_text("content1")
        (source / "file2.txt").write_text("content2")
        (source / "subdir").mkdir()
        (source / "subdir" / "file3.txt").write_text("content3")

        _move_files_to_target(source, target)

        # Check files were moved
        assert (target / "file1.txt").read_text() == "content1"
        assert (target / "file2.txt").read_text() == "content2"
        assert (target / "subdir" / "file3.txt").read_text() == "content3"

        # Source should be empty
        assert not list(source.iterdir())

    def test_move_files_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that moving files overwrites existing files."""
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        target.mkdir()

        # Create file in both directories
        (source / "file.txt").write_text("new content")
        (target / "file.txt").write_text("old content")

        _move_files_to_target(source, target)

        # Target should have new content
        assert (target / "file.txt").read_text() == "new content"


class TestClonePublicRepo:
    """Test clone_public_repo function."""

    @pytest.mark.asyncio
    async def test_clone_success_default_branch(self, tmp_path: Path) -> None:
        """Test successful clone with default branch."""
        target = tmp_path / "target"
        target.mkdir()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            # Mock file moving since we're not actually cloning
            with patch("rasa.builder.github_clone._move_files_to_target"):
                await clone_public_repo(
                    repo_url="https://github.com/owner/repo",
                    target_path=str(target),
                    branch=None,
                )

            # Verify git clone was called correctly
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert call_args[0][0] == "git"
            assert call_args[0][1] == "clone"
            assert "https://github.com/owner/repo" in call_args[0]

    @pytest.mark.asyncio
    async def test_clone_success_specific_branch(self, tmp_path: Path) -> None:
        """Test successful clone with specific branch."""
        target = tmp_path / "target"
        target.mkdir()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            # Mock file moving since we're not actually cloning
            with patch("rasa.builder.github_clone._move_files_to_target"):
                await clone_public_repo(
                    repo_url="https://github.com/owner/repo",
                    target_path=str(target),
                    branch="develop",
                )

            # Verify git clone was called with branch
            call_args = mock_exec.call_args
            assert "--branch" in call_args[0]
            assert "develop" in call_args[0]
            assert "--single-branch" in call_args[0]

    @pytest.mark.asyncio
    async def test_clone_failure_raises_error(self, tmp_path: Path) -> None:
        """Test that clone failure raises GitCloneError."""
        target = tmp_path / "target"
        target.mkdir()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"fatal: repository not found")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(GitCloneError) as exc_info:
                await clone_public_repo(
                    repo_url="https://github.com/owner/nonexistent",
                    target_path=str(target),
                )
            assert "Git clone failed" in str(exc_info.value)
            assert "repository not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_clone_creates_target_directory(self, tmp_path: Path) -> None:
        """Test that clone creates target directory if it doesn't exist."""
        target = tmp_path / "nonexistent" / "nested" / "target"

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # Mock the move operation since we're not actually cloning
            with patch("rasa.builder.github_clone._move_files_to_target"):
                await clone_public_repo(
                    repo_url="https://github.com/owner/repo",
                    target_path=str(target),
                )

        assert target.exists()

    @pytest.mark.asyncio
    async def test_clone_cleanup_on_error(self, tmp_path: Path) -> None:
        """Test that temp directory is cleaned up on error."""
        target = tmp_path / "target"
        target.mkdir()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(GitCloneError):
                await clone_public_repo(
                    repo_url="https://github.com/owner/repo",
                    target_path=str(target),
                )

        # Check that no temp directories remain
        temp_dirs = list(target.glob(".tmp_clone_*"))
        assert len(temp_dirs) == 0

    @pytest.mark.asyncio
    async def test_clone_cleanup_on_unexpected_error(self, tmp_path: Path) -> None:
        """Test cleanup on unexpected exception."""
        target = tmp_path / "target"
        target.mkdir()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=RuntimeError("Unexpected error"),
        ):
            with pytest.raises(GitCloneError) as exc_info:
                await clone_public_repo(
                    repo_url="https://github.com/owner/repo",
                    target_path=str(target),
                )
            assert "Unexpected error" in str(exc_info.value)

        # Check that no temp directories remain
        temp_dirs = list(target.glob(".tmp_clone_*"))
        assert len(temp_dirs) == 0


class TestRunGitHubToBotJob:
    """Test run_github_to_bot_job function."""

    @pytest.mark.asyncio
    async def test_github_to_bot_job_success(self) -> None:
        """Test successful github-to-bot job execution."""
        # Create mocks
        mock_app = MagicMock()
        mock_app.ctx.project_generator = MagicMock()
        mock_app.ctx.project_generator.project_folder = "/test/project"
        mock_app.ctx.project_generator.cleanup = Mock()
        mock_app.ctx.project_generator.get_bot_files = Mock(return_value={})

        # Mock git_service methods
        mock_git_service = MagicMock()
        mock_git_service.init_repo = Mock()
        mock_git_service.ensure_builder_gitignore_entries = Mock()
        mock_git_service.get_current_commit_sha = AsyncMock(return_value="abc123")
        mock_app.ctx.project_generator.git_service = mock_git_service

        # Mock get_training_input
        mock_training_input = MagicMock()
        mock_app.ctx.project_generator.get_training_input = Mock(
            return_value=mock_training_input
        )

        mock_job = MagicMock()
        mock_job.id = "test-job-id"

        mock_job_manager = MagicMock()
        mock_job_manager.create_job = Mock(return_value=MagicMock(id="welcome-job-id"))

        # Mock the clone function
        with patch("rasa.builder.jobs.clone_public_repo", new=AsyncMock()):
            # Mock ensure_first_used
            with patch("rasa.builder.jobs.ensure_first_used", new=Mock()):
                # Mock validate_project
                with patch(
                    "rasa.builder.jobs.validate_project",
                    new=Mock(return_value=None),
                ):
                    # Mock train_and_load_and_link_agent
                    with patch(
                        "rasa.builder.jobs.train_and_load_and_link_agent",
                        new=AsyncMock(return_value=MagicMock()),
                    ):
                        # Mock update_agent
                        with patch("rasa.builder.jobs.update_agent", new=Mock()):
                            # Mock push_job_status_event
                            with patch(
                                "rasa.builder.jobs.push_job_status_event",
                                new=AsyncMock(),
                            ):
                                # Mock send_heartbeat
                                with patch(
                                    "rasa.builder.jobs.send_heartbeat",
                                    new=AsyncMock(),
                                ):
                                    # Mock job_manager
                                    with patch(
                                        "rasa.builder.jobs.job_manager",
                                        mock_job_manager,
                                    ):
                                        await run_github_to_bot_job(
                                            app=mock_app,
                                            job=mock_job,
                                            repo_url="https://github.com/owner/repo",
                                            branch="main",
                                        )

                                        # Verify cleanup was called
                                        cleanup = mock_app.ctx.project_generator.cleanup
                                        cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_to_bot_job_clone_error(self) -> None:
        """Test github-to-bot job handling clone errors."""
        mock_app = MagicMock()
        mock_app.ctx.project_generator = MagicMock()
        mock_app.ctx.project_generator.cleanup = Mock()

        mock_job = MagicMock()
        mock_job.id = "test-job-id"
        mock_job_manager = MagicMock()

        # Mock clone to raise an error
        with patch(
            "rasa.builder.jobs.clone_public_repo",
            new=AsyncMock(side_effect=GitCloneError("Clone failed")),
        ):
            with patch("rasa.builder.jobs.push_job_status_event", new=AsyncMock()):
                with patch("rasa.builder.jobs.send_heartbeat", new=AsyncMock()):
                    with patch("rasa.builder.jobs.job_manager", mock_job_manager):
                        await run_github_to_bot_job(
                            app=mock_app,
                            job=mock_job,
                            repo_url="https://github.com/owner/repo",
                            branch="main",
                        )

                        # Verify error status was pushed
                        mock_job_manager.mark_done.assert_called_once()
                        call_kwargs = mock_job_manager.mark_done.call_args[1]
                        assert "error" in call_kwargs


class TestHandleGitHubToBotEndpoint:
    """Test handle_github_to_bot endpoint handler."""

    @pytest.mark.asyncio
    async def test_handle_github_to_bot_success(self) -> None:
        """Test successful request to github-to-bot endpoint."""
        mock_request = MagicMock(spec=Request)
        mock_request.json = {
            "repo_url": "https://github.com/owner/repo",
            "branch": "main",
        }
        mock_request.app = MagicMock()
        mock_request.app.add_task = Mock()

        mock_job = MagicMock()
        mock_job.id = "test-job-id"

        with patch("rasa.builder.service.job_manager") as mock_job_manager:
            mock_job_manager.create_job.return_value = mock_job

            result: Any = await handle_github_to_bot(mock_request)

            # Verify job was created
            mock_job_manager.create_job.assert_called_once()

            # Verify task was added
            mock_request.app.add_task.assert_called_once()

            # Verify response
            assert result.status == 200

    @pytest.mark.asyncio
    async def test_handle_github_to_bot_invalid_payload(self) -> None:
        """Test github-to-bot endpoint with invalid payload."""
        mock_request = MagicMock(spec=Request)
        mock_request.json = {
            "repo_url": "not-a-valid-url",  # Invalid URL
        }

        result: Any = await handle_github_to_bot(mock_request)

        # Verify error response
        assert result.status == 400
        assert "Invalid request" in str(result.body)

    @pytest.mark.asyncio
    async def test_handle_github_to_bot_missing_repo_url(self) -> None:
        """Test github-to-bot endpoint with missing repo_url."""
        mock_request = MagicMock(spec=Request)
        mock_request.json = {}  # Missing required field

        result: Any = await handle_github_to_bot(mock_request)

        # Verify error response
        assert result.status == 400
