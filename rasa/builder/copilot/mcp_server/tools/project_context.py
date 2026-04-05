"""MCP tools for project context information (flows, slots, responses, actions)."""

from itertools import chain
from pathlib import Path
from typing import Iterator, List, Optional

import structlog

from rasa.builder.copilot.mcp_server.models import (
    CustomActionInfo,
    FlowInfo,
    GetFlowResponse,
    GetResponseResponse,
    GetSlotResponse,
    ListCustomActionsResponse,
    ListFlowsResponse,
    ListResponsesResponse,
    ListSlotsResponse,
    ResponseInfo,
    SlotInfo,
)
from rasa.builder.copilot.mcp_server.tools.utils import validate_subfolder_path
from rasa.builder.logging_utils import get_recent_logs
from rasa.shared.constants import UTTER_PREFIX
from rasa.shared.core.constants import DEFAULT_ACTION_NAMES
from rasa.shared.core.domain import KEY_ACTIONS, KEY_RESPONSES, KEY_SLOTS
from rasa.shared.core.flows.constants import KEY_NAME
from rasa.shared.core.flows.yaml_flows_io import KEY_FLOWS
from rasa.shared.core.slots import AnySlot
from rasa.shared.utils.yaml import read_yaml_file

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
            "mcp_server.tools.get_assistant_logs.error",
            event_info="MCP tool failed to get assistant logs",
            error=str(e),
        )
        return f"Failed to get logs: {e!s}"


async def list_project_flows(
    project_folder: str, data_folder: Optional[str] = "data"
) -> ListFlowsResponse:
    """List all flows in the project with structured information.

    Args:
        project_folder: Path to the project folder
        data_folder: Name of the data folder (None to search entire project)

    Returns:
        ListFlowsResponse with structured flow information.
    """
    try:
        project_path = Path(project_folder).resolve()
        flows: List[FlowInfo] = []

        for yaml_file in _get_flow_yaml_files(project_path, data_folder):
            try:
                content = read_yaml_file(yaml_file)
                if (
                    not content
                    or not isinstance(content, dict)
                    or KEY_FLOWS not in content
                ):
                    continue

                flows_data = content.get(KEY_FLOWS, {})
                if not isinstance(flows_data, dict):
                    continue

                for flow_id, flow_data in flows_data.items():
                    if not isinstance(flow_data, dict):
                        continue

                    flow_info = FlowInfo(
                        id=flow_id,
                        name=flow_data.get(KEY_NAME),
                        file_path=str(yaml_file.relative_to(project_path)),
                    )
                    flows.append(flow_info)

            except Exception as file_error:
                structlogger.warning(
                    "mcp_server.tools.list_project_flows.file_parse_error",
                    file=str(yaml_file),
                    error=str(file_error),
                )
                continue

        return ListFlowsResponse(
            success=True,
            flows=flows,
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.list_project_flows.error",
            event_info="MCP tool failed to list flows",
            error=str(e),
        )
        return ListFlowsResponse(
            success=False,
            flows=[],
            error=f"Failed to list flows: {e!s}",
        )


async def list_project_slots(
    project_folder: str, domain_folder: Optional[str] = "domain"
) -> ListSlotsResponse:
    """List all slots defined in the domain.

    Args:
        project_folder: Path to the project folder
        domain_folder: Name of the domain folder (None to use domain.yml directly)

    Returns:
        ListSlotsResponse with structured slot information.
    """
    try:
        project_path = Path(project_folder).resolve()
        slots: List[SlotInfo] = []
        for yaml_file in _get_domain_yaml_files(project_path, domain_folder):
            slots.extend(_extract_slots_from_file(project_path, yaml_file))

        return ListSlotsResponse(
            success=True,
            slots=slots,
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.list_project_slots.error",
            event_info="MCP tool failed to list slots",
            error=str(e),
        )
        return ListSlotsResponse(
            success=False,
            slots=[],
            error=f"Failed to list slots: {e!s}",
        )


def _extract_slots_from_file(project_path: Path, yaml_file: Path) -> List[SlotInfo]:
    """Extract slot information from a domain YAML file.

    Args:
        project_path: Resolved project root for computing relative paths.
        yaml_file: Path to the YAML file

    Returns:
        List of SlotInfo objects
    """
    slots: List[SlotInfo] = []
    try:
        content = read_yaml_file(yaml_file)
        if not isinstance(content, dict) or KEY_SLOTS not in content:
            return slots

        slots_data = content.get(KEY_SLOTS, {})
        if not isinstance(slots_data, dict):
            return slots

        file_path = str(yaml_file.relative_to(project_path))
        for slot_name, slot_config in slots_data.items():
            if not isinstance(slot_config, dict):
                continue

            slot_info = SlotInfo(
                name=slot_name,
                # "type" has no constant; matches rasa.shared.core.domain usage
                type=slot_config.get("type", AnySlot.type_name),
                file_path=file_path,
            )
            slots.append(slot_info)

    except Exception as file_error:
        structlogger.warning(
            "mcp_server.tools._extract_slots_from_file.parse_error",
            file=str(yaml_file),
            error=str(file_error),
        )
    return slots


async def list_project_responses(
    project_folder: str, domain_folder: Optional[str] = "domain"
) -> ListResponsesResponse:
    """List all responses defined in the domain.

    Args:
        project_folder: Path to the project folder
        domain_folder: Name of the domain folder (None to use domain.yml directly)

    Returns:
        ListResponsesResponse with structured response information.
    """
    try:
        project_path = Path(project_folder).resolve()
        responses: List[ResponseInfo] = []
        for yaml_file in _get_domain_yaml_files(project_path, domain_folder):
            responses.extend(_extract_responses_from_file(project_path, yaml_file))

        return ListResponsesResponse(
            success=True,
            responses=responses,
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.list_project_responses.error",
            event_info="MCP tool failed to list responses",
            error=str(e),
        )
        return ListResponsesResponse(
            success=False,
            responses=[],
            error=f"Failed to list responses: {e!s}",
        )


def _extract_responses_from_file(
    project_path: Path, yaml_file: Path
) -> List[ResponseInfo]:
    """Extract response information from a domain YAML file.

    Args:
        project_path: Resolved project root for computing relative paths.
        yaml_file: Path to the YAML file

    Returns:
        List of ResponseInfo objects
    """
    responses: List[ResponseInfo] = []
    try:
        content = read_yaml_file(yaml_file)
        if not isinstance(content, dict) or KEY_RESPONSES not in content:
            return responses

        responses_data = content.get(KEY_RESPONSES, {})
        if not isinstance(responses_data, dict):
            return responses

        file_path = str(yaml_file.relative_to(project_path))
        for response_name in responses_data.keys():
            response_info = ResponseInfo(
                name=response_name,
                file_path=file_path,
            )
            responses.append(response_info)

    except Exception as file_error:
        structlogger.warning(
            "mcp_server.tools._extract_responses_from_file.parse_error",
            file=str(yaml_file),
            error=str(file_error),
        )
    return responses


async def list_project_custom_actions(
    project_folder: str, domain_folder: Optional[str] = "domain"
) -> ListCustomActionsResponse:
    """List all custom action names declared in the domain YAML.

    Note: This returns action names from domain configuration, not the Python
    implementation code. Use this to discover which custom actions are registered.

    Args:
        project_folder: Path to the project folder
        domain_folder: Name of the domain folder (None to use domain.yml directly)

    Returns:
        ListCustomActionsResponse with structured action information.
    """
    try:
        project_path = Path(project_folder).resolve()
        actions: List[CustomActionInfo] = []
        for yaml_file in _get_domain_yaml_files(project_path, domain_folder):
            actions.extend(_extract_actions_from_file(project_path, yaml_file))

        return ListCustomActionsResponse(
            success=True,
            actions=actions,
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.list_project_custom_actions.error",
            event_info="MCP tool failed to list custom actions",
            error=str(e),
        )
        return ListCustomActionsResponse(
            success=False,
            actions=[],
            error=f"Failed to list custom actions: {e!s}",
        )


def _action_name_from_domain_entry(action_entry: object) -> Optional[str]:
    """Resolve an action name from a domain.yml actions list entry.

    Entries may be plain strings or single-key dicts (e.g. mapping to channel config).

    Args:
        action_entry: One element from the domain ``actions`` list.

    Returns:
        The action name, or ``None`` if the entry cannot be interpreted as a name.
    """
    if isinstance(action_entry, str):
        return action_entry
    if isinstance(action_entry, dict) and action_entry:
        return next(iter(action_entry.keys()))
    return None


def _is_user_registered_custom_action(action_name: str) -> bool:
    """Return True if ``action_name`` is a user custom action, not built-in or utter.

    Args:
        action_name: Declared action name from domain configuration.

    Returns:
        ``True`` if the name should be listed as a custom action for the project.
    """
    if action_name in DEFAULT_ACTION_NAMES:
        return False
    if action_name.startswith(UTTER_PREFIX):
        return False
    return True


def _extract_actions_from_file(
    project_path: Path, yaml_file: Path
) -> List[CustomActionInfo]:
    """Extract custom action information from a domain YAML file.

    Args:
        project_path: Resolved project root for computing relative paths.
        yaml_file: Path to the YAML file

    Returns:
        List of CustomActionInfo objects
    """
    actions: List[CustomActionInfo] = []

    try:
        content = read_yaml_file(yaml_file)
        if not isinstance(content, dict) or KEY_ACTIONS not in content:
            return actions

        actions_data = content.get(KEY_ACTIONS, [])
        if not isinstance(actions_data, list):
            return actions

        file_path = str(yaml_file.relative_to(project_path))
        for action_entry in actions_data:
            action_name = _action_name_from_domain_entry(action_entry)
            if action_name and _is_user_registered_custom_action(action_name):
                actions.append(
                    CustomActionInfo(
                        name=action_name,
                        file_path=file_path,
                    )
                )

    except Exception as file_error:
        structlogger.warning(
            "mcp_server.tools._extract_actions_from_file.parse_error",
            file=str(yaml_file),
            error=str(file_error),
        )
    return actions


def _find_flow_in_file(
    yaml_file: Path, project_path: Path, lookup: str
) -> Optional[FlowInfo]:
    """Load one YAML file and return flow info if ``lookup`` matches id or flow name.

    Args:
        yaml_file: Path to a flows YAML file under the project.
        project_path: Resolved project root (for relative ``file_path`` in the result).
        lookup: Flow id (YAML key) or trimmed flow name to match.

    Returns:
        ``FlowInfo`` when a match exists, otherwise ``None``. Parse errors are logged
        and treated as no match.
    """
    try:
        content = read_yaml_file(yaml_file)
        if not content or not isinstance(content, dict) or KEY_FLOWS not in content:
            return None

        flows_data = content.get(KEY_FLOWS, {})
        if not isinstance(flows_data, dict):
            return None

        for flow_id_key, flow_data in flows_data.items():
            if not isinstance(flow_data, dict):
                continue

            flow_name_stripped = (flow_data.get(KEY_NAME) or "").strip()
            if lookup == flow_id_key or lookup == flow_name_stripped:
                return FlowInfo(
                    id=flow_id_key,
                    name=flow_data.get(KEY_NAME),
                    file_path=str(yaml_file.relative_to(project_path)),
                    definition=flow_data,
                )
    except Exception as file_error:
        structlogger.warning(
            "mcp_server.tools.get_project_flow.file_parse_error",
            file=str(yaml_file),
            error=str(file_error),
        )
    return None


async def get_project_flow(
    project_folder: str,
    flow_id: str,
    data_folder: Optional[str] = "data",
) -> GetFlowResponse:
    """Get a single flow by flow ID (YAML key) or flow name.

    Args:
        project_folder: Path to the project folder
        flow_id: Flow ID (YAML key) or flow name
        data_folder: Name of the data folder (None to search entire project)

    Returns:
        GetFlowResponse with structured flow information.
    """
    if not (flow_id and flow_id.strip()):
        return GetFlowResponse(
            success=False,
            error="flow_id must be a non-empty string",
        )

    lookup = flow_id.strip()
    try:
        project_path = Path(project_folder).resolve()
        for yaml_file in _get_flow_yaml_files(project_path, data_folder):
            if flow := _find_flow_in_file(yaml_file, project_path, lookup):
                return GetFlowResponse(success=True, flow=flow)

        return GetFlowResponse(
            success=False,
            error=f"Flow not found: {flow_id!r}",
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.get_project_flow.error",
            event_info="MCP tool failed to get flow",
            error=str(e),
        )
        return GetFlowResponse(
            success=False,
            error=f"Failed to get flow: {e!s}",
        )


def _find_slot_in_file(
    yaml_file: Path, project_path: Path, lookup: str
) -> Optional[SlotInfo]:
    """Load one domain YAML file and return slot info if ``lookup`` matches a slot key.

    Args:
        yaml_file: Path to a domain YAML file.
        project_path: Resolved project root (for relative ``file_path`` in the result).
        lookup: Slot name to find.

    Returns:
        ``SlotInfo`` when a matching slot exists, otherwise ``None``. Parse errors are
        logged and treated as no match.
    """
    try:
        content = read_yaml_file(yaml_file)
        if not isinstance(content, dict) or KEY_SLOTS not in content:
            return None

        slots_data = content.get(KEY_SLOTS, {})
        if not isinstance(slots_data, dict):
            return None

        for slot_key, slot_config in slots_data.items():
            if not isinstance(slot_config, dict):
                continue

            if lookup == slot_key:
                return SlotInfo(
                    name=slot_key,
                    type=slot_config.get("type", AnySlot.type_name),
                    file_path=str(yaml_file.relative_to(project_path)),
                    definition=slot_config,
                )
    except Exception as file_error:
        structlogger.warning(
            "mcp_server.tools.get_project_slot.file_parse_error",
            file=str(yaml_file),
            error=str(file_error),
        )
    return None


async def get_project_slot(
    project_folder: str,
    slot_name: str,
    domain_folder: Optional[str] = "domain",
) -> GetSlotResponse:
    """Get a single slot by name.

    Args:
        project_folder: Path to the project folder
        slot_name: Slot name
        domain_folder: Name of the domain folder (None to use domain.yml directly)

    Returns:
        GetSlotResponse with structured slot information.
    """
    if not (slot_name and slot_name.strip()):
        return GetSlotResponse(
            success=False,
            error="slot_name must be a non-empty string",
        )

    lookup = slot_name.strip()
    try:
        project_path = Path(project_folder).resolve()
        for yaml_file in _get_domain_yaml_files(project_path, domain_folder):
            if slot := _find_slot_in_file(yaml_file, project_path, lookup):
                return GetSlotResponse(success=True, slot=slot)

        return GetSlotResponse(
            success=False,
            error=f"Slot not found: {lookup!r}",
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.get_project_slot.error",
            event_info="MCP tool failed to get slot",
            error=str(e),
        )
        return GetSlotResponse(
            success=False,
            error=f"Failed to get slot: {e!s}",
        )


async def get_project_response(
    project_folder: str,
    response_name: str,
    domain_folder: Optional[str] = "domain",
) -> GetResponseResponse:
    """Get a single response by name.

    Args:
        project_folder: Path to the project folder
        response_name: Response name
        domain_folder: Name of the domain folder (None to use domain.yml directly)

    Returns:
        GetResponseResponse with structured response information.
    """
    if not (response_name and response_name.strip()):
        return GetResponseResponse(
            success=False,
            error="response_name must be a non-empty string",
        )

    lookup = response_name.strip()
    try:
        # Iterate over the domain YAML files
        project_path = Path(project_folder).resolve()
        for yaml_file in _get_domain_yaml_files(project_path, domain_folder):
            try:
                # Load the YAML file
                content = read_yaml_file(yaml_file)
                if not isinstance(content, dict) or KEY_RESPONSES not in content:
                    continue

                # Get the responses data
                responses_data = content.get(KEY_RESPONSES, {})
                if not isinstance(responses_data, dict):
                    continue

                # Check if the response name matches the lookup
                if lookup not in responses_data:
                    continue

                # Get the variants
                variants = responses_data[lookup]
                if not isinstance(variants, list):
                    variants = []

                file_path = str(yaml_file.relative_to(project_path))
                return GetResponseResponse(
                    success=True,
                    response=ResponseInfo(
                        name=lookup,
                        file_path=file_path,
                        definition=variants,
                    ),
                )
            except Exception as file_error:
                structlogger.warning(
                    "mcp_server.tools.get_project_response.file_parse_error",
                    file=str(yaml_file),
                    error=str(file_error),
                )
                continue

        return GetResponseResponse(
            success=False,
            error=f"Response not found: {response_name!r}",
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.get_project_response.error",
            event_info="MCP tool failed to get response",
            error=str(e),
        )
        return GetResponseResponse(
            success=False,
            error=f"Failed to get response: {e!s}",
        )


# HELPER FUNCTIONS ---------------------------------------------------------------------
# These functions are used to extract information from the project files.


def _glob_yaml_files(directory: Path) -> Iterator[Path]:
    """Glob for YAML files (.yml and .yaml, case-insensitive) in a directory.

    Args:
        directory: The directory to search in

    Returns:
        Iterator of Path objects for YAML files
    """
    # Use character classes for case-insensitive matching across all platforms
    return chain(
        directory.glob("**/*.[yY][mM][lL]"),
        directory.glob("**/*.[yY][aA][mM][lL]"),
    )


def _get_domain_yaml_files(
    project_path: Path, domain_folder: Optional[str] = "domain"
) -> Iterator[Path]:
    """Get all domain YAML files from the project.

    If domain_folder is None, assumes domain.yml in the project root.
    If domain_folder is provided, checks for that directory first,
    then falls back to domain.yml if the directory doesn't exist.

    Args:
        project_path: Resolved path to the project folder.
        domain_folder: Name of the domain folder (None to use domain.yml directly)

    Yields:
        Path objects for domain YAML files
    """
    single_domain = project_path / "domain.yml"

    if domain_folder is None:
        # No domain folder specified, use domain.yml directly
        if single_domain.exists():
            yield single_domain
        return

    # Check for domain folder first
    domain_dir = validate_subfolder_path(project_path, domain_folder)
    if domain_dir.exists():
        yield from _glob_yaml_files(domain_dir)
    elif single_domain.exists():
        # Fall back to single domain.yml file
        yield single_domain


def _get_flow_yaml_files(
    project_path: Path, data_folder: Optional[str] = "data"
) -> Iterator[Path]:
    """Get all potential flow YAML files from the project.

    If data_folder is None, searches the entire project.
    If data_folder is provided, checks for that directory and it's subdirectories first,
    then falls back to searching the entire project if it doesn't exist.

    Args:
        project_path: Resolved path to the project folder.
        data_folder: Name of the data folder (None to search entire project)

    Yields:
        Path objects for potential flow YAML files
    """
    if data_folder is None:
        # No data folder specified, search entire project
        yield from _glob_yaml_files(project_path)
        return

    # Check for data folder first
    data_dir = validate_subfolder_path(project_path, data_folder)
    if data_dir.exists():
        yield from _glob_yaml_files(data_dir)
    else:
        # Fall back to searching entire project
        yield from _glob_yaml_files(project_path)
