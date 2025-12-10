"""MCP resources for project context information."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import structlog

from rasa.builder.logging_utils import get_recent_logs
from rasa.builder.project_generator import get_bot_files

structlogger = structlog.get_logger()


async def get_assistant_logs() -> str:
    """Get recent assistant logs.

    Returns:
        JSON string containing recent log entries from the assistant.
    """
    try:
        return get_recent_logs()
    except Exception as e:
        structlogger.error(
            "mcp_server.resources.project_context.get_assistant_logs.error",
            event_info="MCP resource failed to get assistant logs",
            error=str(e),
        )
        return f"Failed to get logs: {e!s}"


async def get_flows_definitions(project_folder: str) -> str:
    """Get only flow definition files from the data/ directory.

    Args:
        project_folder: Path to the project folder

    Returns:
        JSON string containing flow files and their contents.
    """
    try:
        flow_files = _get_subfolder_files(
            project_folder, subfolder="data", allowed_file_extensions=["yml", "yaml"]
        )

        return json.dumps(
            {"flows": flow_files, "count": len(flow_files)},
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.resources.project_context.get_flows_definitions.error",
            event_info="MCP resource failed to get flows",
            error=str(e),
        )
        return json.dumps({"error": f"Failed to get flows: {e!s}", "flows": {}})


async def get_domain_definitions(project_folder: str) -> str:
    """Get only domain definition files from the domain/ directory.

    Args:
        project_folder: Path to the project folder

    Returns:
        JSON string containing domain files and their contents.
    """
    try:
        domain_files = _get_subfolder_files(
            project_folder,
            subfolder="domain",
            allowed_file_extensions=["yml", "yaml"],
        )

        return json.dumps(
            {"domain": domain_files, "count": len(domain_files)},
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.resources.project_context.get_domain_definitions.error",
            event_info="MCP resource failed to get domain",
            error=str(e),
        )
        return json.dumps({"error": f"Failed to get domain: {e!s}", "domain": {}})


async def get_custom_actions_code(project_folder: str) -> str:
    """Get only custom action Python files from the actions/ directory.

    Args:
        project_folder: Path to the project folder

    Returns:
        JSON string containing action files and their contents.
    """
    try:
        action_files = _get_subfolder_files(
            project_folder, subfolder="actions", allowed_file_extensions=["py"]
        )

        return json.dumps(
            {"actions": action_files, "count": len(action_files)},
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.resources.project_context.get_custom_actions_code.error",
            event_info="MCP resource failed to get actions",
            error=str(e),
        )
        return json.dumps({"error": f"Failed to get actions: {e!s}", "actions": {}})


def _get_subfolder_files(
    project_folder: str,
    subfolder: str,
    allowed_file_extensions: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    """Get the files from a subfolder.

    Args:
        project_folder: Path to the project folder
        subfolder: The subfolder to get the files from
        allowed_file_extensions: The file extensions to get the files from
    """
    project_path = Path(project_folder) / subfolder
    files = get_bot_files(project_path, allowed_file_extensions=allowed_file_extensions)
    return {
        (Path(subfolder) / file_path).as_posix(): file_content
        for file_path, file_content in files.items()
    }
