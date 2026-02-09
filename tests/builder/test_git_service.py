"""Tests for GitService class."""

import asyncio
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder.git_service import (
    CommitNotFoundError,
    GitOperationInProgressError,
    GitService,
)
from rasa.builder.models import CommitFileContents, GitCommitInfo


class TestGitService:
    """Test GitService class methods."""

    @pytest.fixture
    def git_service(self, tmp_path: Path) -> GitService:
        """Create a GitService instance for testing."""
        return GitService(str(tmp_path))

    @pytest.fixture
    def mock_git_command_sync(self) -> Generator[MagicMock, None, None]:
        """Mock the run_git_command_sync method."""
        with patch.object(GitService, "run_git_command_sync") as mock:
            yield mock

    @pytest.fixture
    def mock_git_command_async(self) -> Generator[MagicMock, None, None]:
        """Mock the run_git_command method and bypass lock for legacy tests."""

        # Create a no-op async context manager for the lock
        @asynccontextmanager
        async def mock_git_operation(self) -> AsyncIterator[None]:
            yield

        with (
            patch.object(GitService, "run_git_command") as mock_cmd,
            patch.object(GitService, "git_operation", mock_git_operation),
        ):
            yield mock_cmd

    def test_init(self, tmp_path: Path) -> None:
        """Test GitService initialization."""
        service = GitService(str(tmp_path))
        assert service.project_folder == tmp_path
        assert service.git_dir == tmp_path / ".git"

    def test_init_repo_new_repository(
        self, git_service: GitService, mock_git_command_sync: MagicMock
    ) -> None:
        """Test initializing a new Git repository."""
        # Mock that .git doesn't exist by patching exists()
        with patch.object(Path, "exists", return_value=False):
            git_service.init_repo()

            # Check that git commands were called (allowing for config commands)
            assert mock_git_command_sync.call_count >= 2
            mock_git_command_sync.assert_any_call(["init"])
            mock_git_command_sync.assert_any_call(["checkout", "-b", "main"])

    def test_init_repo_existing_repository(
        self, git_service: GitService, mock_git_command_sync: MagicMock
    ) -> None:
        """Test initializing when .git already exists."""
        # Create .git directory
        git_service.git_dir.mkdir()

        git_service.init_repo()

        # Should not call any git commands
        mock_git_command_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_changes_with_changes(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test committing changes when there are changes."""
        # Mock that there are changes (diff --cached --exit-code fails)
        mock_git_command_async.side_effect = [
            None,  # add .
            subprocess.CalledProcessError(
                1, "git"
            ),  # diff --cached --exit-code (changes exist)
            None,  # commit
            "abc123def456",  # rev-parse HEAD
        ]

        commit_sha = await git_service.commit_changes(
            GitCommitInfo(
                message="Test commit", author="user", email="user@example.com"
            )
        )

        assert commit_sha == "abc123def456"
        mock_git_command_async.assert_any_call(["add", "."])
        mock_git_command_async.assert_any_call(
            ["diff", "--cached", "--exit-code"], skip_error_logging=True
        )
        mock_git_command_async.assert_any_call(
            [
                "commit",
                "-m",
                "Test commit",
                "--allow-empty",
                "--author",
                "user <user@example.com>",
            ],
            env={
                "GIT_COMMITTER_NAME": "user",
                "GIT_COMMITTER_EMAIL": "user@example.com",
            },
        )

    @pytest.mark.asyncio
    async def test_commit_changes_no_changes(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test committing when there are no changes."""
        # Mock that there are no changes (diff --cached --exit-code succeeds)
        mock_git_command_async.side_effect = [
            None,  # add .
            None,  # diff --cached --exit-code (no changes)
            "abc123def456",  # rev-parse HEAD
        ]

        commit_sha = await git_service.commit_changes(
            GitCommitInfo(
                message="Test commit", author="user", email="user@example.com"
            )
        )

        assert commit_sha == "abc123def456"
        # Should not call commit
        assert not any(
            "commit" in str(call) for call in mock_git_command_async.call_args_list[:-1]
        )

    async def test_get_current_commit_sha(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting current commit SHA."""
        mock_git_command_async.return_value = "abc123def456\n"

        sha = await git_service.get_current_commit_sha()

        assert sha == "abc123def456"
        mock_git_command_async.assert_called_once_with(
            ["rev-parse", "HEAD"], check_output=True
        )

    @pytest.mark.asyncio
    async def test_get_commit_history(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting commit history."""
        mock_output = (
            "abc123def456|user|user@example.com|1640995200|Initial commit\n"
            "def456abc123|copilot|copilot@rasa.com|1640995300|Update bot files"
        )
        mock_git_command_async.return_value = mock_output

        commits = await git_service.get_commit_history(50)

        assert len(commits) == 2
        assert commits[0] == {
            "sha": "abc123def456",
            "short_sha": "abc123d",
            "author": "user",
            "email": "user@example.com",
            "timestamp": 1640995200,
            "message": "Initial commit",
        }
        assert commits[1] == {
            "sha": "def456abc123",
            "short_sha": "def456a",
            "author": "copilot",
            "email": "copilot@rasa.com",
            "timestamp": 1640995300,
            "message": "Update bot files",
        }

    @pytest.mark.asyncio
    async def test_get_commit_history_no_commits(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting commit history when there are no commits."""
        mock_git_command_async.side_effect = subprocess.CalledProcessError(128, "git")

        commits = await git_service.get_commit_history(50)

        assert commits == []

    @pytest.mark.asyncio
    async def test_rollback_to_commit(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test rollback creates a new commit by reverting to target commit tree."""
        mock_git_command_async.side_effect = [
            None,  # reset --hard HEAD
            None,  # clean -fdx -e .rasa
            None,  # revert --no-commit <sha>..HEAD
            "abc123def456|user|user@example.com|1640995200|Initial commit\n",
            None,  # commit
            "newcommitsha\n",  # rev-parse HEAD
        ]

        result = await git_service.rollback_to_commit("abc123def456")

        mock_git_command_async.assert_any_call(["reset", "--hard", "HEAD"])
        mock_git_command_async.assert_any_call(["clean", "-fdx", "-e", ".rasa"])
        mock_git_command_async.assert_any_call(
            [
                "revert",
                "--no-commit",
                "abc123def456..HEAD",
            ]
        )
        assert result == "newcommitsha"

    @pytest.mark.asyncio
    async def test_commit_exists_returns_true(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test _commit_exists returns True when commit exists."""
        mock_git_command_async.return_value = None

        result = await git_service._commit_exists("abc123def456")

        assert result is True
        mock_git_command_async.assert_called_once_with(
            ["cat-file", "-e", "abc123def456"], check_output=False
        )

    @pytest.mark.asyncio
    async def test_commit_exists_returns_false(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test _commit_exists returns False when commit doesn't exist."""
        mock_git_command_async.side_effect = subprocess.CalledProcessError(1, "git")

        result = await git_service._commit_exists("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_parent_sha_returns_empty_string(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting parent SHA when commit has no parent."""
        mock_git_command_async.side_effect = subprocess.CalledProcessError(1, "git")

        result = await git_service._get_parent_sha("abc123def456")

        assert result == ""

        assert mock_git_command_async.call_count == 1

    @pytest.mark.asyncio
    async def test_get_parent_sha_returns_empty_string2(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting parent SHA when commit has no parent."""
        mock_git_command_async.return_value = None

        result = await git_service._get_parent_sha("abc123def456")

        assert result == ""

        assert mock_git_command_async.call_count == 1

    @pytest.mark.asyncio
    async def test_get_parent_sha_returns_parent_sha(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting parent SHA when commit has a parent."""
        mock_parent_sha = "abc123def456"
        mock_git_command_async.return_value = mock_parent_sha

        result = await git_service._get_parent_sha("12312312331")

        assert result == mock_parent_sha

        assert mock_git_command_async.call_count == 1

    @pytest.mark.asyncio
    async def test_get_changes_files_when_output_is_none(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting changed files when output is empty."""
        mock_git_command_async.return_value = None

        result = await git_service._get_changed_files("asdasdsdad")

        assert result == []

        assert mock_git_command_async.call_count == 1

    @pytest.mark.asyncio
    async def test_get_changes_files_when_output_is_rich(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting changed files when output is rich in scenarios M, Rx, A, D."""
        mock_git_command_async.return_value = """D\tREADME.md
R054\tscrpt.sh\tscrpt-renamed.sh
M\tdemo.html
A\ttest.added.ts"""

        result = await git_service._get_changed_files("asdasdsdad")

        assert result == [
            ("D", "README.md", None),
            ("R054", "scrpt.sh", "scrpt-renamed.sh"),
            ("M", "demo.html", None),
            ("A", "test.added.ts", None),
        ]

        assert mock_git_command_async.call_count == 1

    @pytest.mark.asyncio
    async def test_build_file_diffs_when_files_are_empty(
        self, git_service: GitService
    ) -> None:
        """Test building file diffs when files are empty."""
        result = await git_service._build_file_diffs([], "abc123123123", "abc123def456")

        assert result == {}

    @pytest.mark.asyncio
    async def test_build_file_diffs_when_parent_sha_is_empty(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test building file diffs when parent SHA is empty.

        This is the case of an initial commit, when all files are added.
        """
        files = [
            ("A", "file.txt", None),
            ("A", "path/to/file2.txt", None),
            ("A", "path/to/file3.txt", None),
        ]
        mock_git_command_async.side_effect = [
            "New file content",
            "New file2 content",
            "New file3 content",
        ]
        mock_commit_sha = "abc123def456"
        result = await git_service._build_file_diffs(files, "", mock_commit_sha)

        assert result == {
            "file.txt": CommitFileContents(
                status="A",
                content_original="",
                content_modified="New file content",
            ),
            "path/to/file2.txt": CommitFileContents(
                status="A",
                content_original="",
                content_modified="New file2 content",
            ),
            "path/to/file3.txt": CommitFileContents(
                status="A",
                content_original="",
                content_modified="New file3 content",
            ),
        }

        assert mock_git_command_async.call_count == 3
        assert mock_git_command_async.call_args_list[0].args[0] == [
            "show",
            f"{mock_commit_sha}:file.txt",
        ]
        assert mock_git_command_async.call_args_list[1].args[0] == [
            "show",
            f"{mock_commit_sha}:path/to/file2.txt",
        ]
        assert mock_git_command_async.call_args_list[2].args[0] == [
            "show",
            f"{mock_commit_sha}:path/to/file3.txt",
        ]

    @pytest.mark.asyncio
    async def test_build_file_diffs_when_commit_sha_is_empty(
        self, git_service: GitService
    ) -> None:
        """Test building file diffs when commit SHA is empty."""
        files = [("A", "file.txt", None)]
        result = await git_service._build_file_diffs(files, "abc123123123", "")

        assert result == {
            "file.txt": CommitFileContents(
                status="A",
                content_original="",
                content_modified="",
            ),
        }

    @pytest.mark.asyncio
    async def test_build_file_diffs_when_files_are_unsupported(
        self, git_service: GitService
    ) -> None:
        """Test building file diffs when files are unsupported."""
        # Unsupported status is X
        files = [("X", "file.txt", None)]

        mock_commit_sha = "abc123def456"

        with pytest.raises(ValueError) as exc_info:
            await git_service._build_file_diffs(files, "abc123123123", mock_commit_sha)

        assert (
            f"Unsupported status: {files[0][0]} "
            f"for file file.txt in commit {mock_commit_sha}" in str(exc_info.value)
        )

    @pytest.mark.asyncio
    async def test_build_file_diffs_when_files_are_rich(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test building file diffs when files are rich.

        These tests should have R, A, M, D statuses to cover for
        file rename, add, modify and delete scenarios.
        """
        files = [
            ("R100", "file.txt", "file-renamed.txt"),
            ("A", "file-added.txt", None),
            ("M", "file-modified.txt", None),
            ("D", "file-deleted.txt", None),
        ]
        mock_parent_sha = "abc123123123"
        mock_commit_sha = "abc123def456"

        mock_git_command_async.side_effect = [
            # rename mock
            "Original file content",
            # add mock
            "Added file content",
            # modify mock
            "Before modified file content",
            "After modified file content",
            # delete mock
            "Deleted file content",
        ]

        result = await git_service._build_file_diffs(
            files, mock_parent_sha, mock_commit_sha
        )

        assert result == {
            "file-renamed.txt": CommitFileContents(
                status="R",
                content_original="Original file content",
                content_modified="Original file content",
                path_original="file.txt",
                path_modified="file-renamed.txt",
            ),
            "file-added.txt": CommitFileContents(
                status="A",
                content_original="",
                content_modified="Added file content",
                path_original=None,
                path_modified=None,
            ),
            "file-modified.txt": CommitFileContents(
                status="M",
                content_original="Before modified file content",
                content_modified="After modified file content",
                path_original=None,
                path_modified=None,
            ),
            "file-deleted.txt": CommitFileContents(
                status="D",
                content_original="Deleted file content",
                content_modified="",
                path_original=None,
                path_modified=None,
            ),
        }

        assert mock_git_command_async.call_count == 5
        assert mock_git_command_async.call_args_list[0].args[0] == [
            "show",
            f"{mock_parent_sha}:file.txt",
        ]
        assert mock_git_command_async.call_args_list[1].args[0] == [
            "show",
            f"{mock_commit_sha}:file-added.txt",
        ]
        assert mock_git_command_async.call_args_list[2].args[0] == [
            "show",
            f"{mock_parent_sha}:file-modified.txt",
        ]
        assert mock_git_command_async.call_args_list[3].args[0] == [
            "show",
            f"{mock_commit_sha}:file-modified.txt",
        ]
        assert mock_git_command_async.call_args_list[4].args[0] == [
            "show",
            f"{mock_parent_sha}:file-deleted.txt",
        ]

    @pytest.mark.asyncio
    async def test_build_file_diffs_when_rename_is_not_100(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test building file diffs when files are rich.

        This test should have R less 100% similarity status
        to cover for file rename with file modifications.
        """
        files = [("R75", "domain/file.txt", "domain/file-renamed.txt")]
        mock_parent_sha = "abc123123123"
        mock_commit_sha = "abc123def456"

        # For renames other than R100 we check diff for new and old path
        # therefore now we have to calls to git show
        mock_git_command_async.side_effect = [
            "Original file content",
            "Modified file content",
        ]
        result = await git_service._build_file_diffs(
            files, mock_parent_sha, mock_commit_sha
        )

        assert result == {
            "domain/file-renamed.txt": CommitFileContents(
                status="R",
                content_original="Original file content",
                content_modified="Modified file content",
                path_original="domain/file.txt",
                path_modified="domain/file-renamed.txt",
            ),
        }

        assert mock_git_command_async.call_count == 2
        assert mock_git_command_async.call_args_list[0].args[0] == [
            "show",
            f"{mock_parent_sha}:domain/file.txt",
        ]
        assert mock_git_command_async.call_args_list[1].args[0] == [
            "show",
            f"{mock_commit_sha}:domain/file-renamed.txt",
        ]

    @pytest.mark.asyncio
    async def test_build_file_diffs_with_binary_files(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test building file diffs with binary files."""
        files = [("A", "file.bin", None), ("D", "file2.bin", None)]
        mock_parent_sha = "abc123123123"
        mock_commit_sha = "abc123def456"

        mock_git_command_async.side_effect = [None, None]
        result = await git_service._build_file_diffs(
            files, mock_parent_sha, mock_commit_sha
        )

        assert mock_git_command_async.call_count == 2
        assert mock_git_command_async.call_args_list[0].args[0] == [
            "show",
            f"{mock_commit_sha}:file.bin",
        ]
        assert mock_git_command_async.call_args_list[1].args[0] == [
            "show",
            f"{mock_parent_sha}:file2.bin",
        ]

        assert result == {
            "file.bin": CommitFileContents(
                status="A",
                content_original=None,
                content_modified=None,
            ),
            "file2.bin": CommitFileContents(
                status="D",
                content_original=None,
                content_modified=None,
            ),
        }

    @pytest.mark.asyncio
    async def test_build_file_diffs_with_binary_files_when_parent_sha_is_none(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test building file diffs with binary files."""
        files = [("A", "file.bin", None)]
        mock_parent_sha = None
        mock_commit_sha = "abc123def456"

        mock_git_command_async.side_effect = [None, None]
        result = await git_service._build_file_diffs(
            files, mock_parent_sha, mock_commit_sha
        )

        assert mock_git_command_async.call_count == 1
        assert mock_git_command_async.call_args[0][0] == [
            "show",
            "abc123def456:file.bin",
        ]

        assert result == {
            "file.bin": CommitFileContents(
                status="A",
                content_original=None,
                content_modified=None,
            ),
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "commit_sha, path, git_command_side_effect, expected_result",
        [
            (
                None,
                "file.txt",
                [UnicodeDecodeError("utf-8", b"", 0, 0, "utf-8")],
                "",
            ),
            (
                "abc123def456",
                None,
                [UnicodeDecodeError("utf-8", b"", 0, 0, "utf-8")],
                "",
            ),
            (
                "abc123def456",
                "file.txt",
                [UnicodeDecodeError("utf-8", b"", 0, 0, "utf-8")],
                None,
            ),
            (
                "abc123def456",
                "file.txt",
                ["valid utf-8 string"],
                "valid utf-8 string",
            ),
        ],
    )
    async def test_git_show(
        self,
        git_service: GitService,
        mock_git_command_async: MagicMock,
        commit_sha: str,
        path: str,
        git_command_side_effect: object,
        expected_result: str | None,
    ) -> None:
        """Test git show."""
        mock_git_command_async.side_effect = git_command_side_effect
        result = await git_service._git_show(commit_sha, path)
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_get_commit_diff_with_contents_when_commit_does_not_exist(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting commit diff with contents."""
        mock_git_command_async.side_effect = subprocess.CalledProcessError(1, "git")

        commit_sha = "abc123def456"

        with pytest.raises(CommitNotFoundError) as exc_info:
            await git_service.get_commit_diff_with_contents(commit_sha)

        assert commit_sha in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)

        assert mock_git_command_async.call_count == 1
        assert mock_git_command_async.call_args[0][0] == ["cat-file", "-e", commit_sha]

    @pytest.mark.asyncio
    async def test_get_commit_diff_with_contents_when_build_diff_fails(
        self, git_service: GitService
    ) -> None:
        """Test getting commit diff with contents when build diff fails."""
        commit_sha = "abc123def456"

        # mock tested functions
        with (
            patch.object(
                git_service, "_commit_exists", return_value=True
            ) as mock_commit_exists,
            patch.object(
                git_service, "_get_parent_sha", return_value="abc123123123"
            ) as mock_get_parent_sha,
            patch.object(
                git_service, "_get_changed_files", side_effect=ValueError("Test error")
            ) as mock_get_changed_files,
        ):
            with pytest.raises(ValueError) as exc_info:
                await git_service.get_commit_diff_with_contents(commit_sha)

            assert "Test error" in str(exc_info.value)
            assert mock_commit_exists.call_count == 1
            assert mock_commit_exists.call_args[0][0] == commit_sha
            assert mock_get_parent_sha.call_count == 1
            assert mock_get_parent_sha.call_args[0][0] == commit_sha
            assert mock_get_changed_files.call_count == 1
            assert mock_get_changed_files.call_args[0][0] == commit_sha

    @pytest.mark.asyncio
    async def test_get_commit_info(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting commit info."""
        mock_git_command_async.side_effect = [
            "abc123def456|user|user@example.com|1640995200|Test commit",
        ]

        diff_data = await git_service.get_commit_info("abc123def456")

        assert mock_git_command_async.call_count == 1

        assert diff_data["sha"] == "abc123def456"
        assert diff_data["author"] == "user"
        assert diff_data["message"] == "Test commit"

        # Call again to test caching
        diff_data_2 = await git_service.get_commit_info("abc123def456")

        assert mock_git_command_async.call_count == 1

        assert diff_data_2["sha"] == "abc123def456"
        assert diff_data_2["author"] == "user"
        assert diff_data_2["message"] == "Test commit"

    @pytest.mark.asyncio
    async def test_get_current_branch(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test getting current branch."""
        mock_git_command_async.return_value = "main\n"

        branch = await git_service.get_current_branch()

        assert branch == "main"
        mock_git_command_async.assert_called_once_with(
            ["branch", "--show-current"], check_output=True
        )

    @pytest.mark.asyncio
    async def test_has_uncommitted_changes_true(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test has_uncommitted_changes when changes exist."""
        mock_git_command_async.side_effect = [
            subprocess.CalledProcessError(1, "git"),  # diff --exit-code fails
        ]

        has_changes = await git_service.has_uncommitted_changes()

        assert has_changes is True

    @pytest.mark.asyncio
    async def test_has_uncommitted_changes_false(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test has_uncommitted_changes when no changes exist."""
        mock_git_command_async.side_effect = [
            None,  # diff --exit-code succeeds
            None,  # diff --cached --exit-code succeeds
        ]

        has_changes = await git_service.has_uncommitted_changes()

        assert has_changes is False

    @pytest.mark.asyncio
    async def test_checkout_branch_existing(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test checking out existing branch."""
        mock_git_command_async.side_effect = [None]  # checkout succeeds

        await git_service.checkout_branch("feature-branch", False)

        mock_git_command_async.assert_called_once_with(["checkout", "feature-branch"])

    @pytest.mark.asyncio
    async def test_checkout_branch_create_new(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test creating and checking out new branch."""
        mock_git_command_async.side_effect = [
            subprocess.CalledProcessError(1, "git"),  # checkout fails
            None,  # reset --hard succeeds
            None,  # clean -fdx succeeds
            None,  # checkout -b succeeds
        ]

        await git_service.checkout_branch("new-branch", True)

        mock_git_command_async.assert_any_call(["checkout", "new-branch"])
        mock_git_command_async.assert_any_call(["reset", "--hard", "HEAD"])
        mock_git_command_async.assert_any_call(["clean", "-fdx"])
        mock_git_command_async.assert_any_call(["checkout", "-b", "new-branch"])

    @pytest.mark.asyncio
    async def test_checkout_branch_existing_with_dirty_workspace(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test checking out existing branch with dirty workspace."""
        mock_git_command_async.side_effect = [
            subprocess.CalledProcessError(1, "git"),  # initial checkout fails
            None,  # reset --hard succeeds
            None,  # clean -fdx succeeds
            None,  # checkout succeeds after cleanup
        ]

        await git_service.checkout_branch("existing-branch", False)

        mock_git_command_async.assert_any_call(["checkout", "existing-branch"])
        mock_git_command_async.assert_any_call(["reset", "--hard", "HEAD"])
        mock_git_command_async.assert_any_call(["clean", "-fdx"])

    @pytest.mark.asyncio
    async def test_create_model_tag(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test creating model tag."""
        mock_git_command_async.side_effect = [
            "user",  # show --format=%an
            "1640995200",  # show --format=%at
            None,  # tag command
        ]

        tag_name = await git_service.create_model_tag(
            "abc123def456", "/path/to/model.tar.gz"
        )

        assert tag_name == "model/model"
        # Verify tag command was called with -f flag to handle existing tags
        mock_git_command_async.assert_any_call(
            [
                "tag",
                "-f",
                "-a",
                "model/model",
                "abc123def456",
                "-m",
                (
                    "Model trained on commit abc123de\n"
                    "Model file: /path/to/model.tar.gz\n"
                    "Author: user\n"
                    "Timestamp: 1640995200"
                ),
            ],
            env={
                "GIT_COMMITTER_NAME": "Rasa Bot Builder",
                "GIT_COMMITTER_EMAIL": "noreply@rasa.com",
            },
        )

    @pytest.mark.asyncio
    async def test_create_model_tag_with_existing_tag(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test creating model tag when tag already exists (force update)."""
        mock_git_command_async.side_effect = [
            "user",  # show --format=%an
            "1640995200",  # show --format=%at
            None,  # tag command (with -f flag should succeed)
        ]

        # Create tag for first training
        tag_name = await git_service.create_model_tag(
            "abc123def456", "/path/to/model.tar.gz"
        )

        assert tag_name == "model/model"
        # The -f flag ensures this succeeds even if tag exists
        mock_git_command_async.assert_any_call(
            [
                "tag",
                "-f",
                "-a",
                "model/model",
                "abc123def456",
                "-m",
                (
                    "Model trained on commit abc123de\n"
                    "Model file: /path/to/model.tar.gz\n"
                    "Author: user\n"
                    "Timestamp: 1640995200"
                ),
            ],
            env={
                "GIT_COMMITTER_NAME": "Rasa Bot Builder",
                "GIT_COMMITTER_EMAIL": "noreply@rasa.com",
            },
        )

    @pytest.mark.asyncio
    async def test_run_git_command_success(self, git_service: GitService) -> None:
        """Test successful git command execution."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(b"success output", b""))
            mock_proc.returncode = 0
            mock_subprocess.return_value = mock_proc

            result = await git_service.run_git_command(["status"], check_output=True)

            assert result == "success output"
            mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_git_command_failure(self, git_service: GitService) -> None:
        """Test git command execution failure."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"error output"))
            mock_proc.returncode = 1
            mock_subprocess.return_value = mock_proc

            # Use a readonly command to bypass lock check
            with pytest.raises(subprocess.CalledProcessError):
                await git_service.run_git_command(["show", "invalid-sha"])

    @pytest.mark.asyncio
    async def test_parallel_operations_blocked(self, tmp_path: Path) -> None:
        """Test that parallel git operations are blocked by the lock."""
        # Create a real git_service without mocks to test actual locking
        real_git_service = GitService(str(tmp_path))

        call_count = 0

        # Create async side effect that delays on the commit command
        async def git_command_side_effect(args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # add .
                return None
            elif call_count == 2:  # diff --cached --exit-code
                raise subprocess.CalledProcessError(1, "git")
            elif call_count == 3:  # commit (make it slow)
                await asyncio.sleep(0.1)
                return None
            elif call_count == 4:  # rev-parse HEAD
                return "abc123def456"

        # Mock git commands
        mock_run_git = AsyncMock(side_effect=git_command_side_effect)
        with patch.object(GitService, "run_git_command", new=mock_run_git):
            # Start first operation
            task1 = asyncio.create_task(
                real_git_service.commit_changes(
                    GitCommitInfo(
                        message="First", author="user", email="user@example.com"
                    )
                )
            )

            # Give task1 time to acquire the lock
            await asyncio.sleep(0.01)

            # Try to start second operation - should raise immediately
            with pytest.raises(GitOperationInProgressError) as exc_info:
                await real_git_service.commit_changes(
                    GitCommitInfo(
                        message="Second", author="user", email="user@example.com"
                    )
                )

            assert "git operation is already in progress" in str(exc_info.value)

            # Wait for first task to complete
            await task1

    @pytest.mark.asyncio
    async def test_rollback_operation_blocked_during_commit(
        self, tmp_path: Path
    ) -> None:
        """Test that rollback is blocked when commit is in progress."""
        # Create a real git_service without mocks to test actual locking
        real_git_service = GitService(str(tmp_path))

        call_count = 0

        # Create async side effect that delays on the commit command
        async def git_command_side_effect(args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # add .
                return None
            elif call_count == 2:  # diff --cached --exit-code
                raise subprocess.CalledProcessError(1, "git")
            elif call_count == 3:  # commit (make it slow)
                await asyncio.sleep(0.1)
                return None
            elif call_count == 4:  # rev-parse HEAD
                return "abc123def456"

        # Mock git commands for commit
        mock_run_git = AsyncMock(side_effect=git_command_side_effect)
        with patch.object(GitService, "run_git_command", new=mock_run_git):
            # Start commit operation
            task1 = asyncio.create_task(
                real_git_service.commit_changes(
                    GitCommitInfo(
                        message="Test", author="user", email="user@example.com"
                    )
                )
            )

            # Give task1 time to acquire the lock
            await asyncio.sleep(0.01)

            # Try rollback - should raise immediately
            with pytest.raises(GitOperationInProgressError) as exc_info:
                await real_git_service.rollback_to_commit("oldcommit123")

            assert "git operation is already in progress" in str(exc_info.value)

            # Wait for first task to complete
            await task1

    @pytest.mark.asyncio
    async def test_sequential_operations_succeed(
        self, git_service: GitService, mock_git_command_async: MagicMock
    ) -> None:
        """Test that sequential git operations succeed."""
        # Mock git commands for two sequential commits
        mock_git_command_async.side_effect = [
            # First commit
            None,  # add .
            subprocess.CalledProcessError(1, "git"),  # diff shows changes
            None,  # commit
            "commit1",  # rev-parse HEAD
            # Second commit
            None,  # add .
            subprocess.CalledProcessError(1, "git"),  # diff shows changes
            None,  # commit
            "commit2",  # rev-parse HEAD
        ]

        # First operation
        sha1 = await git_service.commit_changes(
            GitCommitInfo(message="First", author="user", email="user@example.com")
        )
        assert sha1 == "commit1"

        # Second operation should succeed after first completes
        sha2 = await git_service.commit_changes(
            GitCommitInfo(message="Second", author="user", email="user@example.com")
        )
        assert sha2 == "commit2"

    @pytest.mark.asyncio
    async def test_run_git_command_non_readonly_without_lock_fails(
        self, git_service: GitService
    ) -> None:
        """Test that non-whitelisted commands without lock raise error."""
        # Try to run a non-readonly command without acquiring the lock
        with pytest.raises(GitOperationInProgressError) as exc_info:
            await git_service.run_git_command(["add", "."])

        assert "without acquiring lock" in str(exc_info.value)
        assert "readonly_commands whitelist" in str(exc_info.value)
        assert "This is a bug" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_git_command_readonly_without_lock_succeeds(
        self, git_service: GitService
    ) -> None:
        """Test that read-only commands work without lock."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(b"abc123", b""))
            mock_proc.returncode = 0
            mock_subprocess.return_value = mock_proc

            # Read-only command should work without lock
            result = await git_service.run_git_command(
                ["rev-parse", "HEAD"], check_output=True
            )

            assert result == "abc123"
            mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_git_command_unknown_command_requires_lock(
        self, git_service: GitService
    ) -> None:
        """Test that unknown/new commands are blocked without lock (safe default)."""
        # Try to run a hypothetical new git command that's not in the whitelist
        with pytest.raises(GitOperationInProgressError) as exc_info:
            await git_service.run_git_command(["hypothetical-new-command", "arg"])

        assert "hypothetical-new-command" in str(exc_info.value)
        assert "without acquiring lock" in str(exc_info.value)
        # Helpful message to add it to whitelist if it's actually read-only
        assert "readonly_commands whitelist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_readonly_tag_commands_work_without_lock(
        self, git_service: GitService
    ) -> None:
        """Test that read-only tag commands work without lock."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(b"tag1\ntag2", b""))
            mock_proc.returncode = 0
            mock_subprocess.return_value = mock_proc

            # tag -l should work without lock
            result = await git_service.run_git_command(
                ["tag", "-l", "model/*"], check_output=True
            )
            assert result == "tag1\ntag2"

            # tag --points-at should work without lock
            result = await git_service.run_git_command(
                ["tag", "--points-at", "abc123"], check_output=True
            )
            assert result == "tag1\ntag2"

            # tag --list should work without lock
            result = await git_service.run_git_command(
                ["tag", "--list", "model/*"], check_output=True
            )
            assert result == "tag1\ntag2"

    @pytest.mark.asyncio
    async def test_write_tag_commands_require_lock(
        self, git_service: GitService
    ) -> None:
        """Test that write tag commands require lock."""
        # tag -a (annotate) should require lock
        with pytest.raises(GitOperationInProgressError) as exc_info:
            await git_service.run_git_command(["tag", "-a", "v1.0", "-m", "Release"])
        assert "without acquiring lock" in str(exc_info.value)

        # tag -f (force) should require lock
        with pytest.raises(GitOperationInProgressError) as exc_info:
            await git_service.run_git_command(
                ["tag", "-f", "-a", "v1.0", "abc123", "-m", "Release"]
            )
        assert "without acquiring lock" in str(exc_info.value)

        # tag -d (delete) should require lock
        with pytest.raises(GitOperationInProgressError) as exc_info:
            await git_service.run_git_command(["tag", "-d", "v1.0"])
        assert "without acquiring lock" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tag_without_flags_requires_lock(
        self, git_service: GitService
    ) -> None:
        """Test that tag command without flags requires lock (safe default)."""
        # tag without any flags should require lock for safety
        with pytest.raises(GitOperationInProgressError) as exc_info:
            await git_service.run_git_command(["tag", "v1.0"])
        assert "without acquiring lock" in str(exc_info.value)
