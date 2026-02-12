"""File operation tools for bot project - Hello Rasa supplementary tools.

These tools provide Rasa project-aware file operations with security checks.
They are used by Hello Rasa / Agent Copilot only and are NOT exposed via
the local MCP server, as external IDEs (like Cursor) have their own
built-in file operation capabilities.

This module contains:
1. Implementation functions that take project_folder as a parameter
2. @function_tool decorated wrappers that get project_folder from environment
3. FILE_OPERATION_TOOLS list for injection into the Agent
"""

import json
import os
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import structlog
from agents import function_tool
from pydantic import Field, ValidationError

from rasa.builder.copilot.agent_sdk.tools.constants import (
    TOOL_GET_PROJECT_FILE,
    TOOL_LIST_PROJECT_FILES,
    TOOL_READ_PROJECT_FILES,
    TOOL_UPDATE_MULTIPLE_FILES,
    TOOL_WRITE_PROJECT_FILE,
)
from rasa.builder.copilot.mcp_server.models import (
    FileContentResponse,
    FileListResponse,
    FileUpdateFailure,
    MultiFileUpdate,
    ReadFilesResponse,
    UpdateFilesResponse,
    WriteFileResponse,
)
from rasa.builder.project_generator import (
    bot_file_paths,
    get_bot_files,
    is_restricted_path,
    unsafe_write_to_bot_files,
)
from rasa.builder.telemetry.langfuse.langfuse_compat import observe
from rasa.shared.exceptions import RasaException
from rasa.utils.io import InvalidPathException

structlogger = structlog.get_logger()


def _get_project_folder() -> str:
    """Get the project folder from environment.

    Returns:
        Project folder path as string

    Raises:
        RasaException: If RASA_PROJECT_FOLDER is not set
    """
    project_folder = os.getenv("RASA_PROJECT_FOLDER")
    if project_folder is None:
        raise RasaException(
            "Project folder not configured. The RASA_PROJECT_FOLDER environment "
            "variable must be set."
        )
    return project_folder


# ============================================================================
# IMPLEMENTATION FUNCTIONS
# These are the core implementations that take project_folder as a parameter.
# Used by MCP server resources and can be tested independently.
# ============================================================================


async def list_files(project_folder: str) -> FileListResponse:
    """List all bot project files with directory structure.

    Args:
        project_folder: Path to the project folder

    Returns:
        FileListResponse containing files organized by directory with a tree structure.
    """
    try:
        project_path = Path(project_folder)

        # Collect all allowed files
        all_files = list(bot_file_paths(project_path))
        all_file_paths = [
            str(file_path.relative_to(project_path)) for file_path in all_files
        ]

        # Organize files by directory
        file_tree: Dict[str, List[str]] = {}
        root_files: List[str] = []

        for file_path in sorted(all_file_paths):
            if "/" in file_path:
                # File in a subdirectory
                parts = file_path.split("/")
                directory = parts[0]
                filename = "/".join(parts[1:])

                if directory not in file_tree:
                    file_tree[directory] = []
                file_tree[directory].append(filename)
            else:
                # Root level file
                root_files.append(file_path)

        # Build a human-readable tree structure
        tree_lines = []
        tree_lines.append("Project Structure:")
        tree_lines.append(".")

        # Add root files
        for f in root_files:
            tree_lines.append(f"├── {f}")

        # Add directories and their contents
        dirs = sorted(file_tree.keys())
        for i, directory in enumerate(dirs):
            is_last_dir = i == len(dirs) - 1
            prefix = "└──" if is_last_dir else "├──"
            tree_lines.append(f"{prefix} {directory}/")

            files_in_dir = sorted(file_tree[directory])
            for j, f in enumerate(files_in_dir):
                is_last_file = j == len(files_in_dir) - 1
                if is_last_dir:
                    file_prefix = "    └──" if is_last_file else "    ├──"
                else:
                    file_prefix = "│   └──" if is_last_file else "│   ├──"
                tree_lines.append(f"{file_prefix} {f}")

        tree_view = "\n".join(tree_lines)

        return FileListResponse(
            success=True,
            tree=tree_view,
            files=sorted(all_file_paths),
            count=len(all_file_paths),
            directories=list(file_tree.keys()),
        )

    except Exception as e:
        structlogger.error(
            "agent_sdk.tools.file_operations.list_files.error",
            event_info="Failed to list files",
            error=str(e),
        )
        return FileListResponse(
            success=False,
            files=[],
            count=0,
            directories=[],
            error=f"Failed to list files: {e!s}",
        )


async def read_assistant_files(
    project_folder: str,
    exclude_docs: bool = True,
    allowed_extensions: Optional[str] = None,
) -> ReadFilesResponse:
    """Read all bot project files.

    Args:
        project_folder: Path to the project folder
        exclude_docs: Whether to exclude the docs directory (default: True)
        allowed_extensions: Comma-separated list of file extensions to include
                          (e.g., "yaml,yml,py"). If not provided, includes all files.

    Returns:
        ReadFilesResponse containing a mapping of file paths to their contents.
    """
    try:
        project_path = Path(project_folder)

        # Parse allowed extensions if provided
        extensions = None
        if allowed_extensions:
            extensions = [ext.strip().lower() for ext in allowed_extensions.split(",")]

        bot_files: Dict[str, Optional[str]] = get_bot_files(
            project_path, extensions, exclude_docs
        )

        return ReadFilesResponse(files=bot_files, count=len(bot_files))

    except Exception as e:
        structlogger.error(
            "agent_sdk.tools.file_operations.read_assistant_files.error",
            event_info="Failed to read bot files",
            error=str(e),
        )
        return ReadFilesResponse(
            files={},
            count=0,
            error=f"Failed to read bot files: {e!s}",
        )


async def get_file_content(project_folder: str, file_path: str) -> FileContentResponse:
    """Get the content of a specific bot project file.

    Args:
        project_folder: Path to the project folder
        file_path: Relative path to the file within the project

    Returns:
        FileContentResponse containing the file content and metadata.

    Raises:
        InvalidPathException: If file path is invalid or access is denied
    """
    try:
        project_path = Path(project_folder)
        full_path = project_path / file_path

        # Security check
        if is_restricted_path(project_path, full_path):
            raise InvalidPathException(
                f"Access denied to '{file_path}'. Cannot access hidden files, "
                "__pycache__, or files outside the project directory."
            )

        if not full_path.exists():
            return FileContentResponse(
                file_path=file_path,
                content=None,
                exists=False,
                error=f"File not found: {file_path}",
            )

        content = full_path.read_text(encoding="utf-8")
        return FileContentResponse(
            file_path=file_path,
            content=content,
            exists=True,
        )

    except Exception as e:
        structlogger.error(
            "agent_sdk.tools.file_operations.get_file_content.error",
            event_info="Failed to get file content",
            file_path=file_path,
            error=str(e),
        )
        return FileContentResponse(
            file_path=file_path,
            content=None,
            exists=False,
            error=f"Failed to get file content: {e!s}",
        )


async def write_file(
    project_folder: str, file_path: str, content: str
) -> WriteFileResponse:
    """Write content to a bot project file.

    This will create or overwrite the file at the specified path.

    Args:
        project_folder: Path to the project folder
        file_path: Relative path to the file within the project (e.g., "domain.yml")
        content: The content to write to the file

    Returns:
        WriteFileResponse indicating success or failure

    """
    try:
        project_path = Path(project_folder)
        unsafe_write_to_bot_files(
            project_path, {file_path: content}, fail_on_restricted_path=True
        )
        return WriteFileResponse(
            success=True,
            file_path=file_path,
            message=f"Successfully wrote {len(content)} characters to {file_path}",
        )

    except Exception as e:
        structlogger.error(
            "agent_sdk.tools.file_operations.write_file.error",
            event_info="Failed to write file",
            file_path=file_path,
            error=str(e),
        )
        return WriteFileResponse(
            success=False,
            file_path=file_path,
            message=f"Failed to write file: {e!s}",
        )


async def update_files(
    project_folder: str, files: Dict[str, str]
) -> UpdateFilesResponse:
    """Update multiple bot project files at once.

    This is useful for making coordinated changes across multiple files.

    Args:
        project_folder: Path to the project folder
        files: Dictionary mapping file paths to their new contents

    Returns:
        UpdateFilesResponse with results for each file
    """
    try:
        updated = []
        failed = []

        for file_path, content in files.items():
            result = await write_file(project_folder, file_path, content)

            if result.success:
                updated.append(file_path)
            else:
                failed.append(
                    FileUpdateFailure(
                        file_path=file_path,
                        error=result.message,
                    )
                )

        success = len(failed) == 0
        message = f"Updated {len(updated)} file(s)"
        if failed:
            message += f", {len(failed)} failed"

        return UpdateFilesResponse(
            success=success,
            updated=updated,
            failed=failed,
            message=message,
        )

    except Exception as e:
        structlogger.error(
            "agent_sdk.tools.file_operations.update_files.error",
            event_info="Failed to update files",
            error=str(e),
        )
        return UpdateFilesResponse(
            success=False,
            updated=[],
            failed=[FileUpdateFailure(error=str(e))],
            message=f"Failed to update files: {e!s}",
        )


# ============================================================================
# FUNCTION TOOL WRAPPERS
# These are @function_tool decorated wrappers that the Agent SDK can call.
# They get the project folder from environment and delegate to implementations.
# ============================================================================


@function_tool(
    name_override=TOOL_LIST_PROJECT_FILES,
    description_override="List all files in the bot project as a directory tree",
)
@observe(name=f"file_operation_tool.{TOOL_LIST_PROJECT_FILES}", as_type="generation")
async def list_project_files() -> FileListResponse:
    """List all files in the bot project as a directory tree.

    Shows the complete directory structure with all files organized by folder.
    Use this first to understand the project layout before reading specific files.
    """
    project_folder = _get_project_folder()
    return await list_files(project_folder)


@function_tool(
    name_override=TOOL_READ_PROJECT_FILES,
    description_override="Read all bot project files with their complete contents",
)
@observe(name=f"file_operation_tool.{TOOL_READ_PROJECT_FILES}", as_type="generation")
async def read_project_files(
    exclude_docs: Annotated[
        bool, Field(description="Whether to exclude the docs directory from results")
    ] = True,
    allowed_extensions: Annotated[
        str,
        Field(
            description=(
                "Comma-separated list of file extensions to "
                "include (e.g., 'yaml,yml,py')"
            )
        ),
    ] = "yaml,yml,py,jinja,jinja2",
) -> ReadFilesResponse:
    """Read all bot project files with their complete contents.

    Returns all files in the project that match the allowed extensions,
    organized as a mapping of file paths to their contents.
    """
    project_folder = _get_project_folder()
    return await read_assistant_files(project_folder, exclude_docs, allowed_extensions)


@function_tool(
    name_override=TOOL_GET_PROJECT_FILE,
    description_override="Get the content of a specific file in the bot project",
)
@observe(name=f"file_operation_tool.{TOOL_GET_PROJECT_FILE}", as_type="generation")
async def get_project_file(
    file_path: Annotated[
        str,
        Field(
            description=(
                "Relative path to the file "
                "(e.g., 'domain.yml', 'flows/greeting.yml')"
            )
        ),
    ],
) -> FileContentResponse:
    """Get the content of a specific file in the bot project.

    Use this to read individual files when you know the exact path.
    More efficient than reading all files when you only need one.
    """
    project_folder = _get_project_folder()
    return await get_file_content(project_folder, file_path)


@function_tool(
    name_override=TOOL_WRITE_PROJECT_FILE,
    description_override="Write or overwrite a file in the bot project",
)
@observe(name=f"file_operation_tool.{TOOL_WRITE_PROJECT_FILE}", as_type="generation")
async def write_project_file(
    file_path: Annotated[
        str,
        Field(
            description=(
                "Relative path to the file "
                "(e.g., 'domain.yml', 'domain/booking.yml', 'flows/greeting.yml')"
            )
        ),
    ],
    content: Annotated[str, Field(description="Complete content to write to the file")],
) -> WriteFileResponse:
    """Write or overwrite a file in the bot project.

    Creates new files or completely replaces existing file contents.
    Always validate the project after making changes.
    """
    project_folder = _get_project_folder()
    return await write_file(project_folder, file_path, content)


@function_tool(
    name_override=TOOL_UPDATE_MULTIPLE_FILES,
    description_override="Write multiple files in a single coordinated operation",
)
@observe(name=f"file_operation_tool.{TOOL_UPDATE_MULTIPLE_FILES}", as_type="generation")
async def update_multiple_files(
    files_json: Annotated[
        str,
        Field(
            description=(
                "JSON string mapping file paths to their new contents. "
                'Example: {"domain/booking.yml": "slots:\\n  name:\\n    '
                'type: text\\nresponses:\\n  utter_ok:\\n    - text: OK"}'
            ),
        ),
    ],
) -> UpdateFilesResponse:
    """Write multiple files in a single coordinated operation.

    Use this for making related changes across multiple files atomically.
    More efficient than multiple individual write operations.
    Automatically validates all file paths for security.
    """
    try:
        files = json.loads(files_json)
    except json.JSONDecodeError as e:
        return UpdateFilesResponse(
            success=False,
            updated=[],
            failed=[FileUpdateFailure(error=f"Invalid JSON: {e}")],
            message=f"Failed to parse files JSON: {e}",
        )

    # Validate input using Pydantic model
    try:
        validated = MultiFileUpdate(files=files)
    except ValidationError as e:
        return UpdateFilesResponse(
            success=False,
            updated=[],
            failed=[FileUpdateFailure(error=f"Invalid input format: {e}")],
            message=(
                "Expected a dictionary mapping file paths (strings) to contents "
                "(strings)"
            ),
        )

    project_folder = _get_project_folder()
    return await update_files(project_folder, validated.files)


# Export the function tools for use in the Agent
FILE_OPERATION_TOOLS = [
    list_project_files,
    read_project_files,
    get_project_file,
    write_project_file,
    update_multiple_files,
]
