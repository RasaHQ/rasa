"""Shared utility helpers for MCP tools."""

from pathlib import Path

from rasa.utils.io import InvalidPathException


def validate_subfolder_path(project_path: Path, subfolder: str) -> Path:
    """Validate that a subfolder path stays within the project directory.

    Resolves the combined path and checks it is still inside the project,
    preventing absolute-path replacement (``/etc``) and relative escapes
    (``../../``).

    Args:
        project_path: Resolved project root.
        subfolder: The caller-supplied folder name (e.g. ``data``, ``domain``).

    Returns:
        The resolved subfolder ``Path``.

    Raises:
        InvalidPathException: If the resolved path escapes the project root.
    """
    resolved_project = project_path.resolve()
    resolved_subfolder = (project_path / subfolder).resolve()

    if not resolved_subfolder.is_relative_to(resolved_project):
        raise InvalidPathException(
            f"Path traversal detected: '{subfolder}' resolves outside the "
            f"project directory."
        )
    return resolved_subfolder
