"""MCP tool for listing custom action implementations."""

import ast
from pathlib import Path
from typing import List, Optional

import structlog

from rasa.builder.copilot.mcp_server.models import (
    CustomActionImplementationInfo,
    CustomActionsResponse,
)
from rasa.builder.copilot.mcp_server.tools.project_context import _glob_yaml_files
from rasa.builder.copilot.mcp_server.tools.utils import validate_subfolder_path
from rasa.utils.endpoints import read_endpoint_config
from rasa.utils.io import InvalidPathException

structlogger = structlog.get_logger()

# Default actions folder name (matches rasa_sdk convention)
DEFAULT_ACTIONS_FOLDER = "actions"

# Common endpoint file names to match
ENDPOINT_FILE_NAMES = {
    "endpoints.yml",
    "endpoints.yaml",
}


def _find_endpoint_files(project_path: Path) -> List[Path]:
    """Find all endpoint configuration files in the project.

    Uses _glob_yaml_files to find all YAML files, then filters for files
    matching common endpoint file names.

    Args:
        project_path: Path to the project folder

    Returns:
        List of Path objects for endpoint files found
    """
    try:
        # Get all YAML files in the project (case-insensitive)
        yaml_files = list(_glob_yaml_files(project_path))

        # Filter for files that match common endpoint file names
        return [
            yaml_file
            for yaml_file in yaml_files
            if yaml_file.name.lower() in ENDPOINT_FILE_NAMES
        ]
    except Exception as e:
        structlogger.debug(
            "mcp_server.tools.custom_actions.find_endpoints_error",
            event_info="Failed to find endpoint files",
            error=str(e),
        )
        return []


def _detect_actions_module_from_endpoints(project_path: Path) -> Optional[str]:
    """Try to detect the actions_module from the project's endpoints config.

    Searches for endpoint files and extracts the action_endpoint.actions_module
    setting from the first valid configuration found.

    Args:
        project_path: Path to the project folder

    Returns:
        The actions_module path (converted from Python module notation to
        filesystem path) if found, None otherwise
    """
    endpoint_files = _find_endpoint_files(project_path)

    for endpoint_path in endpoint_files:
        try:
            endpoint_config = read_endpoint_config(
                str(endpoint_path), "action_endpoint"
            )
            if not endpoint_config or not endpoint_config.actions_module:
                continue

            actions_module = endpoint_config.actions_module
            # Handle both string and module type
            if isinstance(actions_module, str):
                # Convert Python module path to filesystem path
                folder_path = actions_module.replace(".", "/")
                structlogger.debug(
                    "mcp_server.tools.custom_actions.detected_module",
                    event_info="Detected actions_module from endpoints",
                    endpoint_file=str(endpoint_path.relative_to(project_path)),
                    actions_module=actions_module,
                    folder_path=folder_path,
                )
                return folder_path
        except Exception as e:
            structlogger.debug(
                "mcp_server.tools.custom_actions.endpoint_parse_error",
                event_info="Failed to parse endpoint file",
                endpoint_file=str(endpoint_path.relative_to(project_path)),
                error=str(e),
            )
            continue

    return None


def _string_literal_from_ast_expr(expr: ast.expr) -> Optional[str]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


def _extract_string_from_name_function(name_fn: ast.FunctionDef) -> Optional[str]:
    for stmt in ast.walk(name_fn):
        if not isinstance(stmt, ast.Return) or not stmt.value:
            continue
        s = _string_literal_from_ast_expr(stmt.value)
        if s is not None:
            return s
    return None


def _extract_action_name_from_method(class_node: ast.ClassDef) -> Optional[str]:
    """Try to extract the action name from the name() method.

    Looks for a simple return statement with a string literal in the name() method.

    Args:
        class_node: The AST node for the class

    Returns:
        The action name if found, None otherwise
    """
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "name":
            return _extract_string_from_name_function(item)
    return None


def _is_action_subclass(class_node: ast.ClassDef) -> bool:
    """Check if a class inherits from Action or a known action base class.

    Args:
        class_node: The AST node for the class

    Returns:
        True if the class appears to be an Action subclass
    """
    action_base_names = {
        "Action",
        "ActionQueryKnowledgeBase",
        "FormValidationAction",
        "ValidationAction",
    }

    for base in class_node.bases:
        # Handle simple name: class MyAction(Action)
        if isinstance(base, ast.Name) and base.id in action_base_names:
            return True
        # Handle attribute access: class MyAction(rasa_sdk.Action)
        if isinstance(base, ast.Attribute) and base.attr in action_base_names:
            return True

    return False


def _scan_python_file(
    file_path: Path, project_path: Path
) -> List[CustomActionImplementationInfo]:
    """Scan a Python file for custom action class definitions.

    Args:
        file_path: Path to the Python file
        project_path: Path to the project root (for relative path calculation)

    Returns:
        List of CustomActionImplementationInfo for each action class found
    """
    actions = []

    try:
        source_code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source_code, filename=str(file_path))

        relative_path = str(file_path.relative_to(project_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_action_subclass(node):
                action_name = _extract_action_name_from_method(node)

                actions.append(
                    CustomActionImplementationInfo(
                        name=action_name,
                        class_name=node.name,
                        file_path=relative_path,
                    )
                )

    except SyntaxError as e:
        structlogger.warning(
            "mcp_server.tools.custom_actions.syntax_error",
            event_info="Failed to parse Python file due to syntax error",
            file_path=str(file_path),
            error=str(e),
        )
    except Exception as e:
        structlogger.warning(
            "mcp_server.tools.custom_actions.parse_error",
            event_info="Failed to parse Python file",
            file_path=str(file_path),
            error=str(e),
        )

    return actions


async def list_custom_action_implementations(
    project_folder: str,
    actions_folder: Optional[str] = None,
) -> CustomActionsResponse:
    """List all custom action implementations in the actions folder.

    Scans Python files in the actions folder and extracts information about
    classes that inherit from rasa_sdk.Action or similar base classes.

    If actions_folder is not specified, attempts to detect it from the project's
    endpoints.yml file. Falls back to "actions" if not found.

    Args:
        project_folder: Path to the project folder
        actions_folder: Path to the actions folder/package, relative to
                       project root. Can be a simple folder name (e.g., "actions")
                       or a nested path (e.g., "my_package/actions"). If not provided,
                       auto-detected from endpoints.yml or defaults to "actions".
                       Absolute paths and path traversal (..) are not allowed.

    Returns:
        CustomActionsResponse containing information about all found actions
    """
    project_path = Path(project_folder)

    # Determine the actions folder: explicit param > auto-detect > default
    folder_name = (
        actions_folder
        or _detect_actions_module_from_endpoints(project_path)
        or DEFAULT_ACTIONS_FOLDER
    )

    try:
        # Validate that the actions folder path stays within the project
        try:
            actions_path = validate_subfolder_path(project_path, folder_name)
        except InvalidPathException as e:
            structlogger.warning(
                "mcp_server.tools.custom_actions.path_traversal_detected",
                event_info="Path traversal attempt detected",
                actions_folder=folder_name,
                error=str(e),
            )
            return CustomActionsResponse(
                actions=[],
                actions_folder=folder_name,
                error=str(e),
            )

        if not actions_path.exists():
            structlogger.info(
                "mcp_server.tools.custom_actions.folder_not_found",
                event_info="Actions folder does not exist",
                actions_folder=str(actions_path),
            )
            return CustomActionsResponse(
                actions=[],
                actions_folder=folder_name,
                error=(
                    f"Actions folder '{folder_name}' not found in project. "
                    f"Please specify the correct actions_folder parameter or configure "
                    f"'action_endpoint.actions_module' in endpoints.yml."
                ),
            )

        if not actions_path.is_dir():
            structlogger.info(
                "mcp_server.tools.custom_actions.not_a_directory",
                event_info="Actions path is not a directory",
                actions_folder=str(actions_path),
            )
            return CustomActionsResponse(
                actions=[],
                actions_folder=folder_name,
                error=(
                    f"Path '{folder_name}' exists but is not a directory. "
                    f"Please specify the correct actions_folder parameter."
                ),
            )

        # Find all Python files in the actions folder (recursively)
        python_files = list(actions_path.rglob("*.py"))

        # Filter out __pycache__ and hidden files
        python_files = [
            f
            for f in python_files
            if "__pycache__" not in str(f) and not f.name.startswith(".")
        ]

        all_actions: List[CustomActionImplementationInfo] = []

        for py_file in python_files:
            file_actions = _scan_python_file(py_file, project_path)
            all_actions.extend(file_actions)

        structlogger.debug(
            "mcp_server.tools.custom_actions.scan_complete",
            event_info="Finished scanning for custom actions",
            actions_folder=folder_name,
            files_scanned=len(python_files),
            actions_found=len(all_actions),
        )

        return CustomActionsResponse(
            actions=all_actions,
            actions_folder=folder_name,
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.custom_actions.error",
            event_info="Failed to list custom actions",
            error=str(e),
            actions_folder=folder_name,
        )
        return CustomActionsResponse(
            actions=[],
            actions_folder=folder_name,
            error=f"Failed to list custom actions: {e!s}",
        )
