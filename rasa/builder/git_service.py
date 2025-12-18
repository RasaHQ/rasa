"""Git service for handling version control operations in Rasa builder."""

import asyncio
import subprocess
import textwrap
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog

from rasa.builder.models import (
    CommitDiffWithContentsResponse,
    CommitFileContents,
    GitCommitInfo,
)

structlogger = structlog.get_logger()

MODEL_TAG_PREFIX = "model/"


DEFAULT_COMMIT_INFO = GitCommitInfo(
    author="Rasa Bot Builder", email="noreply@rasa.com", message="Bot data updated"
)


class GitOperationInProgressError(Exception):
    """Raised when a git operation is requested while another is in progress."""

    pass


class CommitNotFoundError(Exception):
    """Raised when a commit is not found in the repository."""

    pass


class GitService:
    """Service for handling Git operations within the Rasa builder architecture."""

    def __init__(self, project_folder: str) -> None:
        """Initialize GitService with project folder.

        Args:
            project_folder: Path to the project folder
        """
        self.project_folder = Path(project_folder)
        self.git_dir = self.project_folder / ".git"
        # Cache for commit info since commits are immutable
        self._commit_info_cache: Dict[str, Dict[str, Any]] = {}
        # Lock to prevent parallel conflicting operations
        self._operation_lock = asyncio.Lock()

    @asynccontextmanager
    async def git_operation(self) -> AsyncIterator[None]:
        """Context manager for git operations that modify the filesystem.

        Ensures only one operation runs at a time. Use this to protect operations
        that involve both file system changes and git commits as an atomic unit.

        Raises:
            GitOperationInProgressError: If another operation is in progress

        Yields:
            None

        Example:
            async with git_service.git_operation():
                # Write files
                # Commit changes
        """
        # Use wait_for with short timeout to atomically check and acquire the lock
        # This prevents the race condition where two coroutines both pass the
        # locked() check before either acquires the lock
        try:
            await asyncio.wait_for(self._operation_lock.acquire(), timeout=0.001)
        except asyncio.TimeoutError:
            raise GitOperationInProgressError(
                "A git operation is already in progress. "
                "Please wait for it to complete before starting another operation."
            )

        try:
            yield
        finally:
            self._operation_lock.release()

    def git_exists(self) -> bool:
        """Check if Git is initialized."""
        return self.git_dir.exists()

    def init_repo(self) -> None:
        """Initialize Git repository in project folder."""
        if self.git_dir.exists():
            return

        structlogger.info(
            "git_service.init_repository", project_folder=str(self.project_folder)
        )

        self._initialize_git_repository()
        self._setup_git_configuration()
        self._create_gitignore()

        structlogger.info(
            "git_service.repo_initialized", project_folder=str(self.project_folder)
        )

    def _initialize_git_repository(self) -> None:
        """Initialize the Git repository and create main branch."""
        self.run_git_command_sync(["init"])
        self.run_git_command_sync(["checkout", "-b", "main"])

    def _setup_git_configuration(self) -> None:
        """Set up basic Git configuration if not already set."""
        self._set_git_config_if_missing("user.name", "Rasa Bot Builder")
        self._set_git_config_if_missing("user.email", "noreply@rasa.com")

    def _set_git_config_if_missing(self, config_key: str, default_value: str) -> None:
        """Set Git configuration if not already set."""
        try:
            self.run_git_command_sync(["config", config_key])
        except subprocess.CalledProcessError:
            self.run_git_command_sync(["config", config_key, default_value])

    def _create_gitignore(self) -> None:
        """Create .gitignore file for the project."""
        gitignore = self.project_folder / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                textwrap.dedent("""
                    models/
                    *.pyc
                    __pycache__/
                    .rasa/
                    """)
            )

    async def commit_changes(self, commit_info: GitCommitInfo) -> str:
        """Create Git commit and return commit SHA.

        Args:
            commit_info: Information about the commit

        Returns:
            Commit SHA of the created commit

        Raises:
            GitOperationInProgressError: If another operation is in progress
        """
        async with self.git_operation():
            return await self._commit_changes_internal(commit_info)

    async def _commit_changes_internal(self, commit_info: GitCommitInfo) -> str:
        """Internal commit implementation (assumes lock is already held).

        Args:
            commit_info: Information about the commit

        Returns:
            Commit SHA of the created commit
        """
        await self._stage_all_changes()

        if not await self._has_staged_changes():
            structlogger.info("git_service.no_changes_to_commit")
            return await self.get_current_commit_sha()

        await self._create_commit(commit_info)
        commit_sha = await self.get_current_commit_sha()

        structlogger.info(
            "git_service.commit_created",
            commit_sha=commit_sha,
            commit_info=commit_info,
        )
        return commit_sha

    async def _stage_all_changes(self) -> None:
        """Stage all changes for commit."""
        await self.run_git_command(["add", "."])

    async def _has_staged_changes(self) -> bool:
        """Check if there are staged changes to commit."""
        try:
            # in this case it is ok if the exit code is not 0
            await self.run_git_command(
                ["diff", "--cached", "--exit-code"], skip_error_logging=True
            )
            return False
        except subprocess.CalledProcessError:
            # in case of an error we assume there are changes not yet committed
            return True

    async def _create_commit(self, commit_info: GitCommitInfo) -> None:
        """Create a commit with author attribution."""
        message = commit_info.message or ""
        await self.run_git_command(
            [
                "commit",
                "-m",
                message,
                "--allow-empty",
                "--author",
                f"{commit_info.author} <{commit_info.email}>",
            ]
        )

    async def get_current_commit_sha(self) -> str:
        """Get current HEAD commit SHA.

        Returns:
            Current HEAD commit SHA, empty string if no commits exist
        """
        try:
            result = await self.run_git_command(
                ["rev-parse", "HEAD"], check_output=True
            )
            return result.strip() if result else ""
        except subprocess.CalledProcessError:
            # No commits yet
            return ""

    async def get_commit_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get Git commit history.

        Args:
            limit: Maximum number of commits to return

        Returns:
            List of commit information dictionaries
        """
        git_log_output = await self._get_git_log_output(limit)
        if not git_log_output:
            return []

        return self._parse_commit_log(git_log_output)

    async def _get_git_log_output(self, limit: int) -> Optional[str]:
        """Get raw git log output."""
        try:
            return await self.run_git_command(
                ["log", "--oneline", "--format=%H|%an|%ae|%at|%s", f"-{limit}"],
                check_output=True,
            )
        except subprocess.CalledProcessError:
            # No commits yet
            return None

    def _parse_commit_log(self, git_log_output: str) -> List[Dict[str, Any]]:
        """Parse git log output into commit dictionaries."""
        commits = []
        for line in git_log_output.strip().split("\n"):
            if not line:
                continue

            commit_data = self._parse_commit_line(line)
            if commit_data:
                commits.append(commit_data)

        return commits

    def _parse_commit_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single commit line into a commit dictionary."""
        try:
            sha, author, email, timestamp, message = line.split("|", 4)
            return {
                "sha": sha,
                "short_sha": sha[:7],
                "author": author,
                "email": email,
                "timestamp": int(timestamp),
                "message": message,
            }
        except ValueError:
            # Skip malformed lines
            return None

    async def rollback_to_commit(self, commit_sha: str) -> str:
        """Create a new commit that restores the repository to the given commit.

        Instead of moving HEAD or checking out the commit, this computes the
        inverse of all changes from the target commit to the current HEAD and
        applies them as a new commit. After this operation, the working tree
        matches the state referenced by ``commit_sha`` while preserving history.

        Args:
            commit_sha: SHA of the commit to restore to
            author: Author of the rollback commit
            message: Optional commit message; if not provided, a default is used

        Returns:
            The SHA of the new rollback commit

        Raises:
            GitOperationInProgressError: If another operation is in progress
        """
        async with self.git_operation():
            # Ensure a clean workspace so revert can proceed without conflicts
            await self.run_git_command(["reset", "--hard", "HEAD"])  # discard changes
            # Clean untracked files (including ignored files with -x) but preserve .rasa
            await self.run_git_command(["clean", "-fdx", "-e", ".rasa"])

            # Revert all commits from target (exclusive) to HEAD, staging the inverse
            # changes. This results in the tree matching the target commit after
            # we finalize the commit.
            await self.run_git_command(["revert", "--no-commit", f"{commit_sha}..HEAD"])

            # Get commit info
            commit_info = await self.get_commit_info(commit_sha)

            # Create the rollback commit
            new_commit_info = DEFAULT_COMMIT_INFO.model_copy(
                update={"message": f"Restore '{commit_info['message']}'"}
            )
            await self._create_commit(new_commit_info)
            new_sha = await self.get_current_commit_sha()

            structlogger.info(
                "git_service.rollback_completed",
                target_commit_sha=commit_sha,
                new_commit_sha=new_sha,
                project_folder=self.project_folder.as_posix(),
            )
            return new_sha

    async def get_commit_diff_with_contents(
        self, commit_sha: str
    ) -> CommitDiffWithContentsResponse:
        """Get commit diff with contents."""
        # First verify the commit exists
        try:
            if not await self._commit_exists(commit_sha):
                structlogger.error(
                    "git_service.get_commit_diff_with_content_not_found",
                    commit_sha=commit_sha,
                )
                raise CommitNotFoundError(
                    f"Commit {commit_sha} does not exist in this repository"
                )
            parent_sha = await self._get_parent_sha(commit_sha)
            files = await self._get_changed_files(commit_sha)
            file_diffs: dict[str, CommitFileContents] = await self._build_file_diffs(
                files, parent_sha, commit_sha
            )
            return CommitDiffWithContentsResponse(
                files=file_diffs,
            )
        except subprocess.CalledProcessError as e:
            structlogger.error(
                "git_service.get_commit_diff_with_content_failed", error=str(e)
            )
            raise

    async def _build_file_diffs(
        self, files: list[tuple[str, str, str | None]], parent_sha: str, commit_sha: str
    ) -> dict[str, CommitFileContents]:
        file_diffs: dict[str, CommitFileContents] = {}
        for status, old_path, new_path in files:
            path = new_path or old_path

            original_path = None
            modified_path = None
            original = ""
            modified = ""

            # If no parent (initial commit), treat all files as added
            if not parent_sha:
                original = ""
                modified = await self._git_show(commit_sha, path) or ""
            elif status.startswith("R"):  # rename
                if status == "R100":
                    original = await self._git_show(parent_sha, old_path) or ""
                    modified = original
                else:
                    original = await self._git_show(parent_sha, old_path) or ""
                    modified = await self._git_show(commit_sha, new_path) or ""
                original_path = old_path
                modified_path = new_path
                status = "R"
            elif status == "D":  # deleted
                original = await self._git_show(parent_sha, old_path) or ""
                modified = ""
            elif status == "A":  # added
                original = ""
                modified = await self._git_show(commit_sha, path) or ""
            elif status == "M":  # modified
                original = await self._git_show(parent_sha, old_path) or ""
                modified = await self._git_show(commit_sha, path) or ""
            else:  # others are not supported
                structlogger.error(
                    "git_service.get_commit_diff_with_contents_unsupported_status",
                    status=status,
                    commit_sha=commit_sha,
                    parent_sha=parent_sha,
                    old_path=old_path,
                    new_path=new_path,
                )
                raise ValueError(
                    f"Unsupported status: {status} for file {path} "
                    f"in commit {commit_sha}"
                )
            file_diffs[path] = CommitFileContents(
                status=status,
                content_original=original,
                content_modified=modified,
                path_original=original_path,
                path_modified=modified_path,
            )
        return file_diffs

    async def _commit_exists(self, commit_sha: str) -> bool:
        """Check if a commit exists in the repository."""
        try:
            await self.run_git_command(
                ["cat-file", "-e", commit_sha], check_output=False
            )
            return True
        except subprocess.CalledProcessError:
            return False

    async def _get_parent_sha(self, commit_sha: str) -> str:
        """Get parent SHA of a commit. Returns empty string if no parent exists."""
        try:
            result = await self.run_git_command(
                ["rev-parse", f"{commit_sha}^1"], check_output=True
            )
            return (result or "").strip()
        except subprocess.CalledProcessError:
            # Commit has no parent (initial commit)
            return ""

    async def _get_changed_files(
        self, commit_sha: str
    ) -> List[tuple[str, str, str | None]]:
        """Get changed files for a commit."""
        diff_tree_result = await self.run_git_command(
            [
                "diff-tree",
                "--root",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-M",
                commit_sha,
            ],
            check_output=True,
        )
        if not diff_tree_result:
            return []

        files: list[tuple[str, str, str | None]] = []
        for line in diff_tree_result.strip().splitlines():
            parts = line.split("\t")
            # R0-100, A, M, D
            status = parts[0]
            if status.startswith("R"):  # R0-100 old new
                files.append((status, parts[1], parts[2]))
            else:
                files.append((status, parts[1], None))
        return files

    async def _git_show(self, sha: str, path: str) -> Optional[str]:
        if not sha or not path:
            return ""
        return await self.run_git_command(["show", f"{sha}:{path}"], check_output=True)

    async def _get_commit_info(self, commit_sha: str) -> Dict[str, Any]:
        """Get commit information (author, timestamp, message)."""
        commit_info_output = await self.run_git_command(
            ["show", "--format=%H|%an|%ae|%at|%s", "--no-patch", commit_sha],
            check_output=True,
        )

        if not commit_info_output:
            return {}

        return self._parse_commit_line(commit_info_output.strip()) or {}

    async def get_commit_info(self, commit_sha: str) -> Dict[str, Any]:
        """Get info for a specific commit (cached).

        Args:
            commit_sha: SHA of the commit

        Returns:
            Dictionary with commit information copy
        """
        # Check cache first
        if commit_sha in self._commit_info_cache:
            return self._commit_info_cache[commit_sha].copy()

        try:
            commit_info = await self._get_commit_info(commit_sha)
            # Cache the result
            self._commit_info_cache[commit_sha] = commit_info
            return commit_info.copy()
        except subprocess.CalledProcessError as e:
            structlogger.error(
                "git_service.get_commit_info_failed",
                error=str(e),
                commit_sha=commit_sha,
            )
            raise

    async def get_current_branch(self) -> str:
        """Get the current Git branch name.

        Returns:
            Current branch name
        """
        result = await self.run_git_command(
            ["branch", "--show-current"], check_output=True
        )
        return result.strip() if result else "main"

    async def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes.

        Returns:
            True if there are uncommitted changes, False otherwise
        """
        try:
            # in this case it is ok if the exit code is not 0
            await self.run_git_command(["diff", "--exit-code"], skip_error_logging=True)
            await self.run_git_command(
                ["diff", "--cached", "--exit-code"], skip_error_logging=True
            )
            return False
        except subprocess.CalledProcessError:
            # in case of any error, we assume that there are still
            # uncommitted changes
            return True

    async def checkout_branch(
        self, branch_name: str, create_if_not_exists: bool = False
    ) -> None:
        """Checkout a Git branch, handling dirty workspace.

        This method will discard any uncommitted changes to ensure
        the checkout succeeds regardless of workspace state.

        Args:
            branch_name: Name of the branch to checkout
            create_if_not_exists: Whether to create the branch if it doesn't exist

        Raises:
            GitOperationInProgressError: If another operation is in progress
        """
        async with self.git_operation():
            try:
                # First try a regular checkout
                await self.run_git_command(["checkout", branch_name])
                structlogger.info(
                    "git_service.checkout_success",
                    branch_name=branch_name,
                    project_folder=self.project_folder.as_posix(),
                )
            except subprocess.CalledProcessError:
                if create_if_not_exists:
                    # For new branch creation, discard changes and create
                    await self._discard_changes_and_checkout_new(branch_name)
                else:
                    # For existing branch, try discarding changes first
                    await self._discard_changes_and_checkout_existing(branch_name)

    async def _discard_changes_and_checkout_new(self, branch_name: str) -> None:
        """Discard all changes and create a new branch."""
        try:
            # Reset any staged changes
            await self.run_git_command(["reset", "--hard", "HEAD"])
            # Clean untracked files
            await self.run_git_command(["clean", "-fdx"])
            # Create and checkout new branch
            await self.run_git_command(["checkout", "-b", branch_name])
            structlogger.info(
                "git_service.checkout_created_after_cleanup",
                branch_name=branch_name,
                project_folder=self.project_folder.as_posix(),
            )
        except subprocess.CalledProcessError as e:
            structlogger.error(
                "git_service.checkout_new_failed",
                branch_name=branch_name,
                error=str(e),
            )
            raise

    async def _discard_changes_and_checkout_existing(self, branch_name: str) -> None:
        """Discard all changes and checkout existing branch."""
        try:
            # Reset any staged changes
            await self.run_git_command(["reset", "--hard", "HEAD"])
            # Clean untracked files
            await self.run_git_command(["clean", "-fdx"])
            # Try checkout again
            await self.run_git_command(["checkout", branch_name])
            structlogger.info(
                "git_service.checkout_success_after_cleanup",
                branch_name=branch_name,
                project_folder=self.project_folder.as_posix(),
            )
        except subprocess.CalledProcessError as e:
            structlogger.error(
                "git_service.checkout_existing_failed",
                branch_name=branch_name,
                error=str(e),
            )
            raise

    def _is_readonly_command(self, args: List[str]) -> bool:
        """Check if a git command is read-only and safe to run without lock.

        Args:
            args: Git command arguments (without 'git')

        Returns:
            True if the command is read-only and doesn't require a lock
        """
        if not args:
            return False

        command = args[0]

        # Whitelist of read-only commands that are safe without lock
        # Any command NOT in this list will require the lock
        readonly_commands = {
            "show",
            "log",
            "diff",
            "diff-tree",
            "status",
            "branch",
            "rev-parse",
            "ls-files",
            "cat-file",
            "rev-list",
            "config",  # For reading config (writes during init are sync)
        }

        if command in readonly_commands:
            return True

        # Special case: 'tag' command can be read-only or write depending on flags
        if command == "tag":
            # Read-only tag operations: -l/--list, --points-at
            # Write operations: -a, -f, -d, -m
            readonly_tag_flags = {"-l", "--list", "--points-at"}
            write_tag_flags = {
                "-a",
                "--annotate",
                "-f",
                "--force",
                "-d",
                "--delete",
                "-m",
            }

            # Check if any flag is present in args
            for arg in args[1:]:
                if arg in readonly_tag_flags:
                    return True
                if arg in write_tag_flags:
                    return False

            # If called with no flags or only value arguments (not in either set),
            # default to requiring lock for safety
            return False

        return False

    async def run_git_command(
        self,
        args: List[str],
        check_output: bool = False,
        skip_error_logging: bool = False,
    ) -> Optional[str]:
        """Run a Git command in the project folder.

        Args:
            args: Git command arguments (without 'git')
            check_output: Whether to return the command output
            skip_error_logging: Whether to ignore errors

        Returns:
            Command output if check_output is True, None otherwise

        Raises:
            subprocess.CalledProcessError: If the Git command fails
            GitOperationInProgressError: If a non-readonly command is run without lock
        """
        # Ensure lock is held for all commands except known read-only ones
        if not self._is_readonly_command(args) and not self._operation_lock.locked():
            raise GitOperationInProgressError(
                f"Cannot run git command '{args[0]}' without acquiring lock. "
                "If this is a read-only command, add it to the "
                "readonly_commands whitelist. This is a bug in the implementation."
            )

        cmd = ["git"] + args

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,  # Unpack the command list
                cwd=self.project_folder,
                # create_subprocess_exec returns bytes it does NOT have a text parameter
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            # Check return code and raise exception if failed
            if proc.returncode != 0:
                error = subprocess.CalledProcessError(
                    proc.returncode, cmd, stdout.decode("utf-8")
                )
                error.stderr = stderr.decode("utf-8")
                raise error

            if check_output:
                return stdout.decode("utf-8").strip()
            return None
        except subprocess.CalledProcessError as e:
            if not skip_error_logging:
                structlogger.error(
                    "git_service.command_failed",
                    error=str(e),
                    project_folder=self.project_folder.as_posix(),
                )
            raise

    def run_git_command_sync(
        self, args: List[str], check_output: bool = False
    ) -> Optional[str]:
        """Run a Git command synchronously (for use in __init__).

        Note: This method is only used during initialization and migration,
        so it bypasses the lock check.

        Args:
            args: Git command arguments (without 'git')
            check_output: Whether to return the command output

        Returns:
            Command output if check_output is True, None otherwise

        Raises:
            subprocess.CalledProcessError: If the Git command fails
        """
        cmd = ["git"] + args

        try:
            if check_output:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_folder,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout
            else:
                subprocess.run(
                    cmd,
                    cwd=self.project_folder,
                    check=True,
                    capture_output=True,
                )
                return None
        except subprocess.CalledProcessError as e:
            structlogger.error(
                "git_service.command_failed",
                error=str(e),
                project_folder=self.project_folder.as_posix(),
            )
            raise

    async def create_model_tag(self, commit_sha: str, model_file_path: str) -> str:
        """Create a git tag linking model to commit.

        Args:
            commit_sha: SHA of the commit the model was trained on
            model_file_path: Path to the trained model file

        Returns:
            The created tag name

        Raises:
            GitOperationInProgressError: If another operation is in progress
        """
        # Extract model name from file path (remove .tar.gz extension)
        model_name = Path(model_file_path).name
        if model_name.endswith(".tar.gz"):
            model_name = model_name[:-7]  # Remove .tar.gz
        elif model_name.endswith(".tar"):
            model_name = model_name[:-4]  # Remove .tar
        else:
            model_name = Path(model_file_path).stem  # Fallback to stem
        tag_name = f"{MODEL_TAG_PREFIX}{model_name}"

        # Create annotated tag with model metadata
        author = await self._get_commit_author(commit_sha)
        timestamp = await self._get_commit_timestamp(commit_sha)

        tag_message = (
            f"Model trained on commit {commit_sha[:8]}\n"
            f"Model file: {model_file_path}\n"
            f"Author: {author}\n"
            f"Timestamp: {timestamp}"
        )

        # Acquire lock for tag creation (modifies repository)
        async with self.git_operation():
            # Use -f flag to force update if tag already exists (e.g., from retraining)
            await self.run_git_command(
                ["tag", "-f", "-a", tag_name, commit_sha, "-m", tag_message]
            )

        structlogger.info(
            "git_service.model_tag_created",
            commit_sha=commit_sha,
            model_path=model_file_path,
            tag_name=tag_name,
        )

        return tag_name

    async def get_model_tags(self) -> List[Dict[str, str]]:
        """Get all model tags in the repository.

        Returns:
            List of dictionaries with tag info
        """
        try:
            # Get all tags with the model/ prefix
            result = await self.run_git_command(
                [
                    "tag",
                    "-l",
                    f"{MODEL_TAG_PREFIX}*",
                    "--format=%(refname:short)|%(objectname)|%(contents:subject)",
                ],
                check_output=True,
            )

            if not result:
                return []

            tags = []
            for line in result.strip().split("\n"):
                if not line:
                    continue
                try:
                    tag_name, commit_sha, subject = line.split("|", 2)
                    tags.append(
                        {
                            "tag_name": tag_name,
                            "commit_sha": commit_sha,
                            "model_name": tag_name.replace(MODEL_TAG_PREFIX, ""),
                            "subject": subject,
                        }
                    )
                except ValueError:
                    continue

            return tags
        except subprocess.CalledProcessError:
            return []

    async def get_model_for_commit(self, commit_sha: str) -> Optional[str]:
        """Get model tag for a specific commit.

        Args:
            commit_sha: SHA of the commit

        Returns:
            Model tag name if found, None otherwise
        """
        try:
            # Get tags pointing to this commit
            result = await self.run_git_command(
                ["tag", "--points-at", commit_sha, "--list", f"{MODEL_TAG_PREFIX}*"],
                check_output=True,
            )

            if not result:
                return None

            # Return the first model tag found
            tags = result.strip().split("\n")
            return tags[0] if tags and tags[0] else None

        except subprocess.CalledProcessError:
            return None

    async def _get_commit_timestamp(self, commit_sha: str) -> int:
        """Get timestamp for a commit.

        Args:
            commit_sha: SHA of the commit

        Returns:
            Unix timestamp of the commit
        """
        try:
            result = await self.run_git_command(
                ["show", "--format=%at", "--no-patch", commit_sha], check_output=True
            )
            return int(result.strip()) if result else 0
        except (subprocess.CalledProcessError, ValueError):
            return 0

    async def _get_commit_author(self, commit_sha: str) -> str:
        """Get author for a commit.

        Args:
            commit_sha: SHA of the commit

        Returns:
            Author name of the commit
        """
        try:
            result = await self.run_git_command(
                ["show", "--format=%an", "--no-patch", commit_sha], check_output=True
            )
            return result.strip() if result else "unknown"
        except subprocess.CalledProcessError:
            return "unknown"

    async def load_model_for_commit(self, commit_sha: str) -> Optional[Any]:
        """Try to load an existing model for a specific commit using git tags.

        Args:
            commit_sha: SHA of the commit

        Returns:
            Loaded agent if model exists, None otherwise
        """
        try:
            # Avoid circular import
            from rasa.builder.training_service import try_load_existing_agent

            # Get model tag for this commit
            model_tag = await self.get_model_for_commit(commit_sha)
            if not model_tag:
                return None

            # Extract model name from tag
            model_name = model_tag.replace(MODEL_TAG_PREFIX, "")

            # Look for the model file in the models directory
            models_dir = self.project_folder / "models"
            if not models_dir.exists():
                return None

            # Find model files with this name
            model_files = list(models_dir.glob(f"{model_name}*.tar.gz"))
            if not model_files:
                return None

            # Use the most recent model file with this name
            model_file = max(model_files, key=lambda f: f.stat().st_mtime)

            # Try to load the specific model file
            return await try_load_existing_agent(self.project_folder, model_file)

        except Exception as e:
            structlogger.warning(
                "load_model_for_commit.failed", commit_sha=commit_sha, error=str(e)
            )
            return None


async def link_model_to_commit(
    git_service: GitService, agent: Any, commit_sha: str
) -> None:
    """Link a trained model to its commit SHA using git tags.

    Args:
        git_service: The GitService instance
        agent: The trained agent
        commit_sha: SHA of the commit to link to
    """
    model_file = extract_model_file_path(git_service.project_folder, agent)
    if model_file:
        await git_service.create_model_tag(commit_sha, model_file)


def extract_model_file_path(project_folder: Path, agent: Any) -> Optional[str]:
    """Extract model file path from agent or find latest model.

    Args:
        project_folder: Path to the project folder
        agent: The agent to extract model path from

    Returns:
        Path to the model file if found, None otherwise
    """
    if (
        agent
        and hasattr(agent, "model_metadata")
        and isinstance(agent.model_metadata, dict)
        and agent.model_metadata.get("model_file")
    ):
        return agent.model_metadata["model_file"]
    elif (
        agent
        and hasattr(agent, "_model_metadata")
        and isinstance(agent._model_metadata, dict)
        and agent._model_metadata.get("model_file")
    ):
        return agent._model_metadata["model_file"]
    else:
        # Try to get the latest model file
        from rasa.model import get_latest_model

        return get_latest_model(str(project_folder / "models"))
