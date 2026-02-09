"""GitHub repository cloning for public repositories.

This module provides functionality for cloning public GitHub repositories.
"""

import asyncio
import shutil
from pathlib import Path
from typing import Optional

import structlog

structlogger = structlog.get_logger()


class GitCloneError(Exception):
    """Exception raised when git clone fails."""

    pass


def _cleanup_temp_directory(temp_dir: Path) -> None:
    """Clean up temporary clone directory.

    Args:
        temp_dir: Path to temporary directory to remove.
    """
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def _move_files_to_target(temp_dir: Path, target_dir: Path) -> None:
    """Move all files from temporary directory to target directory.

    Args:
        temp_dir: Source directory with cloned files.
        target_dir: Destination directory.
    """
    for item in temp_dir.iterdir():
        dest = target_dir / item.name
        # Remove destination if it exists
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        # Move from temp to target
        shutil.move(str(item), str(dest))


async def clone_public_repo(
    repo_url: str,
    target_path: str,
    branch: Optional[str] = None,
) -> None:
    """Clone a public GitHub repository.

    This function clones into a temporary subdirectory and then moves the
    contents to the target directory. This approach is more robust than
    trying to clone into the current directory.

    Args:
        repo_url: Repository URL (HTTPS format).
        target_path: Directory to clone into.
        branch: Branch to checkout. If None, uses default branch.

    Raises:
        GitCloneError: If cloning fails.
    """
    import uuid

    structlogger.info(
        "github_clone.starting",
        repo_url=repo_url,
        target_path=target_path,
        branch=branch,
    )

    target_dir = Path(target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique temp directory name for cloning
    temp_clone_dir = target_dir / f".tmp_clone_{uuid.uuid4().hex[:8]}"

    try:
        # Build git clone command
        cmd = ["git", "clone"]
        if branch:
            cmd.extend(["--branch", branch, "--single-branch"])
        cmd.extend([repo_url, str(temp_clone_dir)])

        # Execute git clone
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/tmp",
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            stderr_str = stderr.decode("utf-8")
            structlogger.error(
                "github_clone.failed",
                returncode=proc.returncode,
                stderr=stderr_str,
            )
            raise GitCloneError(f"Git clone failed: {stderr_str}")

        # Move files and cleanup
        _move_files_to_target(temp_clone_dir, target_dir)
        _cleanup_temp_directory(temp_clone_dir)

        structlogger.info(
            "github_clone.success",
            target_path=target_path,
            branch=branch,
        )

    except (asyncio.CancelledError, GitCloneError):
        # Cleanup and re-raise
        _cleanup_temp_directory(temp_clone_dir)
        raise
    except Exception as e:
        # Cleanup temp directory on unexpected error
        _cleanup_temp_directory(temp_clone_dir)
        structlogger.exception(
            "github_clone.unexpected_error",
            error=str(e),
        )
        raise GitCloneError(f"Unexpected error during clone: {e}")
