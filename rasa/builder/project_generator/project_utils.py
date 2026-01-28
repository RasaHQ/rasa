from pathlib import Path
from typing import Dict, Generator, List, Optional

import structlog

from rasa.builder.exceptions import InvalidFileContentError
from rasa.builder.models import BotFiles
from rasa.shared.constants import (
    DEFAULT_MODELS_PATH,
)
from rasa.utils.io import InvalidPathException, subpath

DEFAULT_COMMIT_MESSAGE = "Update files"
structlogger = structlog.get_logger()


def get_bot_files(
    project_folder: Path,
    allowed_file_extensions: Optional[List[str]] = None,
    exclude_docs_directory: bool = False,
    exclude_models_directory: bool = True,
) -> BotFiles:
    """Get the current bot files by reading from disk.

    Args:
        project_folder: Path to the project folder
        allowed_file_extensions: Optional list of file extensions to include.
            If None, fetch all files. If provided, only fetch files with matching
            extensions. Use `""` empty string to allow files with no extensions.
        exclude_docs_directory: Optional boolean indicating whether to exclude.
        exclude_models_directory: Optional boolean indicating whether to exclude.

    Returns:
        Dictionary of file contents with relative paths as keys
    """
    bot_files: BotFiles = {}

    for file in bot_file_paths(project_folder, exclude_models_directory):
        relative_path = file.relative_to(project_folder)

        # Exclude the docs directory if specified
        if exclude_docs_directory and relative_path.parts[0] == "docs":
            continue

        # Exclude the files by file extensions if specified
        if allowed_file_extensions is not None:
            allowed_file_extensions = [ext.lower() for ext in allowed_file_extensions]
            if file.suffix.lstrip(".").lower() not in allowed_file_extensions:
                continue
        # Read file content and store with relative path as key
        try:
            bot_files[relative_path.as_posix()] = file.read_text(encoding="utf-8")
        except Exception as e:
            structlogger.debug(
                "project_generator.get_bot_files.error",
                error=str(e),
                file_path=file.as_posix(),
            )
            bot_files[relative_path.as_posix()] = None
    return bot_files


def bot_file_paths(
    project_folder: Path, exclude_models_directory: bool = True
) -> Generator[Path, None, None]:
    """Get the paths of all bot files.

    Args:
        project_folder: Path to the project folder
        exclude_models_directory: Optional boolean indicating whether to exclude.

    Returns:
        Generator of file paths
    """
    for file in project_folder.glob("**/*"):
        # Skip directories
        if not file.is_file() or is_restricted_path(
            project_folder, file, exclude_models_directory
        ):
            continue

        yield file


def is_restricted_path(
    project_folder: Path, path: Path, exclude_models_directory: bool = True
) -> bool:
    """Check if the path is restricted.

    These paths are excluded from deletion and editing by the user.

    Args:
        project_folder: Path to the project folder
        path: Path to the file or directory
        exclude_models_directory: Optional boolean indicating whether to exclude.

    Returns:
        True if the path is restricted, False otherwise
    """
    relative_path = path.relative_to(project_folder)

    # Skip hidden files and directories (any path component starting with '.')
    # as well as `__pycache__` folders
    if any(part.startswith(".") for part in relative_path.parts):
        return True

    if "__pycache__" in relative_path.parts:
        return True

    # exclude the project_folder / models folder if specified
    if exclude_models_directory and relative_path.parts[0] == DEFAULT_MODELS_PATH:
        return True

    return False


def unsafe_write_to_bot_files(
    project_folder: Path,
    files: Dict[str, Optional[str]],
    fail_on_restricted_path: bool = True,
) -> None:
    """Write content to bot project files.

    This does NOT acquire a git operation lock, make sure the caller
    has a lock on files to prevent concurrent writes.

    Args:
        project_folder: Path to the project folder
        files: Dictionary mapping file names to their content
        fail_on_restricted_path: Optional boolean indicating whether to fail
            on restricted paths.
    """
    for filename, content in files.items():
        file_path = path_relative_to_project(project_folder, filename)
        # Disallow updates inside .rasa project metadata directory
        if any(
            part.startswith(".") for part in file_path.relative_to(project_folder).parts
        ):
            # silently ignore hidden paths
            if fail_on_restricted_path:
                raise InvalidPathException(
                    f"This file or folder is restricted from editing: {file_path}"
                )
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content or "", encoding="utf-8")


def path_relative_to_project(project_folder: Path, filename: str) -> Path:
    """Get the relative path of a file or directory to the project folder."""
    return Path(subpath(str(project_folder), filename))


def validate_file_content_transitions(
    new_files: Dict[str, Optional[str]],
    current_files: BotFiles,
    project_folder: Path,
) -> None:
    """Validate file content state transitions.

    Rules:
    - binary (null) -> string: REJECT - cannot modify binary content
    - binary (null) -> null: OK - preserve file unchanged
    - text (string) -> null: REJECT - use "" for empty file
    - text (string) -> string: OK - write new content
    - doesn't exist -> null: REJECT - cannot create binary via API
    - doesn't exist -> string: OK - create new file

    Args:
        new_files: Dictionary mapping file names to their new content
        current_files: Dictionary of current file contents
        project_folder: Path to the project folder for path normalization

    Raises:
        InvalidFileContentError: If a state transition is not allowed
    """
    abs_project_folder = project_folder.resolve()

    for filename, new_content in new_files.items():
        # Normalize the filename to match the format used in current_files
        # (relative POSIX paths like "image.png" instead of "./image.png")
        normalized_path = path_relative_to_project(abs_project_folder, filename)
        normalized_filename = normalized_path.relative_to(abs_project_folder).as_posix()

        current_content: str | None = current_files.get(normalized_filename)
        file_exists: bool = normalized_filename in current_files

        if new_content is None:
            # New content is null - only valid for existing binary files
            if not file_exists:
                raise InvalidFileContentError(
                    f"Cannot create binary file '{filename}' through this API. "
                )
            if current_content is not None:
                # Current file is text, new content is null
                raise InvalidFileContentError(
                    f"Cannot set text file '{filename}' to null. "
                    f'Use an empty string ("") if you want an empty file, '
                    f"or omit the file from the request to delete it."
                )
            # current_content is None (binary) and new_content is None -> OK
        else:
            # New content is string
            if file_exists and current_content is None:
                # Current file is binary, new content is text
                raise InvalidFileContentError(
                    f"Cannot modify binary file '{filename}' through this API. "
                    f"Binary files can only be preserved (send with content: null) "
                    f"or deleted (omit from request)."
                )
