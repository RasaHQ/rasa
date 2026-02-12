"""Agent SDK tools implementations."""

from rasa.builder.copilot.agent_sdk.tools.planning_tools import PLANNING_TOOLS


def get_file_operation_tools() -> list:
    """Get file operation tools lazily to avoid circular imports.

    Returns:
        List of file operation function tools for the Agent.
    """
    from rasa.builder.copilot.agent_sdk.tools.file_operations import (
        FILE_OPERATION_TOOLS,
    )

    return FILE_OPERATION_TOOLS


def get_supplementary_tools() -> list:
    """Get all supplementary tools for the Hello Rasa Agent.

    This includes both planning tools and file operation tools.
    File operations are loaded lazily to avoid circular imports.

    Returns:
        List of all function tools for the Agent.
    """
    return PLANNING_TOOLS + get_file_operation_tools()


__all__ = [
    "get_supplementary_tools",
    "get_file_operation_tools",
]
