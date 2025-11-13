"""Validation functions for agent configurations with reduced redundancies."""

import os
import urllib.parse
from collections import Counter
from typing import Any, Dict, List, NoReturn, Set

import jinja2.exceptions
import structlog
from pydantic import ValidationError as PydanticValidationError

from rasa.agents.exceptions import (
    AgentNameFlowConflictException,
    DuplicatedAgentNameException,
)
from rasa.core.available_agents import (
    DEFAULT_AGENTS_CONFIG_FOLDER,
    AgentConfig,
    AgentConfiguration,
    AgentConnections,
    AgentInfo,
    AvailableAgents,
    ProtocolConfig,
)
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.exceptions import ValidationError
from rasa.shared.agents.auth.utils import validate_secrets_in_params
from rasa.shared.utils.llm import get_prompt_template, validate_jinja2_template
from rasa.shared.utils.yaml import read_config_file

# Initialize logger
structlogger = structlog.get_logger()

# Centralized allowed keys configuration to eliminate duplication
ALLOWED_KEYS = {
    "agent": {"name", "protocol", "description"},
    "configuration": {
        "llm",
        "prompt_template",
        "module",
        "timeout",
        "max_retries",
        "agent_card",
        "auth",
        "include_date_time",
        "timezone",
    },
    "connections": {"mcp_servers"},
}


def validate_agent_names_unique(agents: List[AgentConfig]) -> None:
    """Validate that agent names are unique across all loaded agents.

    Args:
        agents: List of agent configurations.

    Raises:
        DuplicatedAgentNameException: If agent names are not unique.
    """
    agent_names = [agent_config.agent.name for agent_config in agents]
    name_counts = Counter(agent_names)

    duplicated_names = [name for name, count in name_counts.items() if count > 1]

    if duplicated_names:
        raise DuplicatedAgentNameException(duplicated_names)


def validate_agent_names_not_conflicting_with_flows(
    agents: Dict[str, AgentConfig], flow_names: Set[str]
) -> None:
    """Validate that agent names do not conflict with flow names.

    Args:
        agents: Dictionary of agent configurations.
        flow_names: Set of flow names to check against.

    Raises:
        AgentNameFlowConflictException: If agent names conflict with flow names.
    """
    conflicting_names = [
        agent_name for agent_name in agents.keys() if agent_name in flow_names
    ]

    if conflicting_names:
        raise AgentNameFlowConflictException(conflicting_names)


def _validate_mcp_config(agent_config: AgentConfig) -> None:
    """Validate MCP-specific configuration requirements."""
    agent_name = agent_config.agent.name

    # Check connections.mcp_servers exists
    if agent_config.connections is None or agent_config.connections.mcp_servers is None:
        raise ValidationError(
            code="agent.validation.mcp.missing_connections",
            event_info=f"For protocol 'RASA', agent '{agent_name}' must have "
            "'connections.mcp_servers' configured.",
        )

    # Check mcp_servers list is not empty
    if not agent_config.connections.mcp_servers:
        raise ValidationError(
            code="agent.validation.mcp.empty_servers_list",
            event_info=f"For protocol 'RASA', agent '{agent_name}' must have "
            "at least one MCP server configured in 'connections.mcp_servers'.",
        )

    # Validate each MCP server configuration
    for i, server in enumerate(agent_config.connections.mcp_servers):
        if not server.name:
            raise ValidationError(
                code="agent.validation.mcp.server_missing_name",
                event_info=f"For protocol 'RASA', agent '{agent_name}' MCP server "
                f"at index {i} must have a 'name' field.",
            )


def _validate_a2a_config(agent_config: AgentConfig) -> None:
    """Validate A2A-specific configuration requirements."""
    agent_name = agent_config.agent.name

    # Check configuration.agent_card exists
    if (
        agent_config.configuration is None
        or agent_config.configuration.agent_card is None
    ):
        raise ValidationError(
            code="agent.validation.a2a.missing_agent_card",
            event_info=f"For protocol 'A2A', agent '{agent_name}' must have "
            "'configuration.agent_card' specified.",
        )

    # Validate agent_card path or URL
    agent_card = agent_config.configuration.agent_card
    if not agent_card:
        raise ValidationError(
            code="agent.validation.a2a.empty_agent_card",
            event_info=f"Agent '{agent_name}' has empty 'agent_card' value",
        )

    # Check if it's a URL
    parsed_url = urllib.parse.urlparse(agent_card)
    if parsed_url.scheme and parsed_url.netloc:
        # It's a URL, validate format
        if parsed_url.scheme not in ["http", "https"]:
            raise ValidationError(
                code="agent.validation.a2a.invalid_agent_card_url",
                event_info=f"Agent '{agent_name}' has invalid URL scheme "
                f"'{parsed_url.scheme}' for 'agent_card'",
            )
    else:
        # It's a file path, check if it exists
        if not os.path.exists(agent_card):
            raise ValidationError(
                code="agent.validation.a2a.agent_card_file_not_found",
                event_info=f"Agent '{agent_name}' has 'agent_card' file that doesn't "
                f"exist: {agent_card}",
            )

    # Validate right use of secrets in auth configuration
    if agent_config.configuration.auth:
        validate_secrets_in_params(
            agent_config.configuration.auth, f"agent '{agent_name}'"
        )


def _validate_prompt_template_syntax(prompt_path: str, agent_name: str) -> None:
    """Validate Jinja2 syntax of a prompt template file."""
    try:
        # Use a simple default template, as we're assuming
        # that the default templates are valid
        default_template = "{{ content }}"
        template_content = get_prompt_template(
            prompt_path,
            default_template,
            log_source_component=f"agent.validation.{agent_name}",
            log_source_method="init",
        )
        validate_jinja2_template(template_content)

    except jinja2.exceptions.TemplateSyntaxError as e:
        raise ValidationError(
            code="agent.validation.prompt_template_syntax_error",
            event_info=(
                f"Agent '{agent_name}' has invalid Jinja2 template syntax at line "
                f"{e.lineno}: {e.message}"
            ),
        ) from e
    except Exception as e:
        raise ValidationError(
            code="agent.validation.optional.prompt_template_error",
            event_info=(f"Agent '{agent_name}' has error reading prompt template: {e}"),
        ) from e


def _validate_optional_keys(agent_config: AgentConfig) -> None:
    """Validate optional keys in agent configuration."""
    agent_name = agent_config.agent.name

    # Validate prompt_template if present
    if agent_config.configuration and agent_config.configuration.prompt_template:
        prompt_path = agent_config.configuration.prompt_template
        if not os.path.exists(prompt_path):
            # If reading the custom prompt fails,
            # allow fallback to default prompt template
            structlogger.warning(
                "agent.validation.optional.prompt_template_file_not_found",
                agent_name=agent_name,
                prompt_path=prompt_path,
                event_info=(
                    f"Prompt template file not found: {prompt_path}. "
                    f"Agent will use default template."
                ),
            )
            # Don't raise ValidationError, allow fallback to default template
            return

        # Validate Jinja2 syntax
        _validate_prompt_template_syntax(prompt_path, agent_name)

    # Validate module if present
    if agent_config.configuration and agent_config.configuration.module:
        import importlib

        module_name = agent_config.configuration.module
        try:
            module_path, class_name = module_name.rsplit(".", 1)
            getattr(importlib.import_module(module_path), class_name)
        except (ImportError, AttributeError) as e:
            raise ValidationError(
                code="agent.validation.optional.module_not_found",
                event_info=f"Agent '{agent_name}' has module '{module_name}' "
                f"that could not be imported: {e}",
            )


def _validate_llm_references(
    llm_config: Dict[str, Any],
    endpoints: "AvailableEndpoints",
    agent_name: str,
) -> None:
    """Validate LLM configuration references against endpoints."""
    if "model_group" in llm_config:
        from rasa.engine.validation import (
            _validate_component_model_client_config_has_references_to_endpoints,
        )

        component_config = {"llm": llm_config}
        _validate_component_model_client_config_has_references_to_endpoints(
            component_config, "llm", component_name=agent_name
        )


def _validate_mcp_server_references(
    mcp_servers: list, endpoints: "AvailableEndpoints", agent_name: str
) -> None:
    """Validate MCP server references against endpoints."""
    if not endpoints.mcp_servers:
        raise ValidationError(
            code="agent.validation.endpoints.no_mcp_servers",
            event_info=(
                f"Agent '{agent_name}' references MCP servers but no MCP "
                "servers are defined in endpoints.yml."
            ),
        )

    available_mcp_server_names = [server.name for server in endpoints.mcp_servers]

    for i, mcp_server in enumerate(mcp_servers):
        server_name = mcp_server.name
        if server_name not in available_mcp_server_names:
            raise ValidationError(
                code="agent.validation.endpoints.invalid_mcp_server",
                event_info=(
                    f"MCP server '{server_name}' at index {i} for Agent "
                    f"'{agent_name}' does not exist in endpoints.yml. Available MCP "
                    f"servers: {', '.join(available_mcp_server_names)}"
                ),
            )


def _handle_pydantic_validation_error(
    error: PydanticValidationError, agent_name: str
) -> NoReturn:
    """Handle specific Pydantic validation errors that are actually possible.

    Args:
        error: The Pydantic validation error to handle
        agent_name: Name of the agent for error messages
    """
    missing_fields = []
    invalid_protocol = False
    type_error = None

    for pydantic_error in error.errors():
        error_type = pydantic_error["type"]
        field_path = ".".join(str(loc) for loc in pydantic_error["loc"])

        if error_type == "missing":
            for field in ["name", "protocol", "description"]:
                if field in field_path:
                    missing_fields.append(field)
        elif error_type == "enum" and "protocol" in field_path:
            invalid_protocol = True
        elif error_type in ["string_type", "int_parsing"]:
            type_error = (field_path, pydantic_error["msg"])

    # Handle missing required fields
    if missing_fields:
        raise ValidationError(
            code="agent.validation.mandatory.fields_missing",
            event_info=(
                f"Agent '{agent_name}' is missing required fields "
                f"in agent section: {', '.join(missing_fields)}"
            ),
        )

    # Handle invalid protocol
    elif invalid_protocol:
        raise ValidationError(
            code="agent.validation.pydantic.invalid_protocol",
            event_info=(
                f"Agent '{agent_name}' has invalid protocol value. "
                "Supported protocols: MCP, A2A"
            ),
        )

    # Handle type errors
    elif type_error:
        field, msg = type_error
        raise ValidationError(
            code="agent.validation.pydantic.type_error",
            event_info=(
                f"Agent '{agent_name}' has invalid type for field " f"'{field}': {msg}"
            ),
        )

    # Handle other Pydantic validation errors
    else:
        raise ValidationError(
            code="agent.validation.pydantic.failed",
            event_info=f"Agent '{agent_name}' validation failed: {error}",
        )


def _validate_endpoint_references(agent_config: AgentConfig) -> None:
    """Validate that LLM and MCP server references in agent config are valid."""
    agent_name = agent_config.agent.name
    endpoints = Configuration.get_instance().endpoints
    if not endpoints.config_file_path:
        # If no endpoints were loaded (e.g., `data validate` without --endpoints), skip
        # endpoint reference checks
        return

    # Validate LLM configuration references
    if agent_config.configuration and agent_config.configuration.llm:
        _validate_llm_references(agent_config.configuration.llm, endpoints, agent_name)

    # Validate MCP server references
    if agent_config.connections and agent_config.connections.mcp_servers:
        _validate_mcp_server_references(
            agent_config.connections.mcp_servers, endpoints, agent_name
        )


def _validate_section_keys(
    data: Dict[str, Any], section: str, allowed_keys: set
) -> None:
    """Generic function to validate keys in a specific section."""
    if section not in data:
        return

    section_data = data[section]
    if not isinstance(section_data, dict):
        return

    additional_keys = set(section_data.keys()) - allowed_keys
    if additional_keys:
        agent_name = data.get("agent", {}).get("name", "unknown")
        raise ValidationError(
            code=f"agent.validation.structure.additional_{section}_keys",
            event_info=(
                f"Agent '{agent_name}' contains additional keys in "
                f"'{section}' section: {', '.join(sorted(additional_keys))}"
            ),
        )


def _validate_mandatory_fields(data: Dict[str, Any], agent_name: str) -> None:
    """Validate that all mandatory fields are present in the agent section."""
    if "agent" not in data:
        raise ValidationError(
            code="agent.validation.mandatory.agent_section_missing",
            event_info=f"Agent '{agent_name}' is missing 'agent' section",
        )

    agent_data = data["agent"]
    if not isinstance(agent_data, dict):
        raise ValidationError(
            code="agent.validation.mandatory.agent_section_invalid",
            event_info=(
                f"Agent '{agent_name}' has invalid 'agent' section - "
                "must be a dictionary"
            ),
        )

    # Check for required fields (protocol is optional due to default in model)
    missing_fields = []
    for field in ["name", "description"]:
        if field not in agent_data or not agent_data[field]:
            missing_fields.append(field)

    if missing_fields:
        raise ValidationError(
            code="agent.validation.mandatory.fields_missing",
            event_info=(
                f"Agent '{agent_name}' is missing required fields in agent section: "
                f"{', '.join(missing_fields)}"
            ),
        )


def _validate_no_additional_keys_raw_data(data: Dict[str, Any]) -> None:
    """Validate that no additional, unexpected keys are present in the raw data."""
    # Use the generic validation function for each section
    _validate_section_keys(data, "agent", ALLOWED_KEYS["agent"])
    _validate_section_keys(data, "configuration", ALLOWED_KEYS["configuration"])
    _validate_section_keys(data, "connections", ALLOWED_KEYS["connections"])


def _validate_folder_structure(agent_folder: str) -> None:
    """Validate agent folder structure."""
    if not os.path.exists(agent_folder):
        raise ValidationError(
            code="agent.validation.folder.not_found",
            event_info=f"Agent folder does not exist: {agent_folder}",
        )

    if not os.path.isdir(agent_folder):
        raise ValidationError(
            code="agent.validation.folder.not_directory",
            event_info=f"Agent folder is not a directory: {agent_folder}",
        )


def _validate_config_file_exists(config_path: str, agent_name: str) -> None:
    """Validate config.yml file exists."""
    if not os.path.isfile(config_path):
        raise ValidationError(
            code="agent.validation.folder.missing_config",
            event_info=f"Agent '{agent_name}' is missing 'config.yml' file",
        )


def validate_agent_config(agent_config: AgentConfig) -> None:
    """Validate an agent configuration using all applicable validators."""
    protocol = agent_config.agent.protocol

    # Run protocol-specific validation
    if protocol == ProtocolConfig.RASA:
        _validate_mcp_config(agent_config)
    elif protocol == ProtocolConfig.A2A:
        _validate_a2a_config(agent_config)

    # Run optional keys validation
    _validate_optional_keys(agent_config)

    # Run endpoint references validation
    _validate_endpoint_references(agent_config)


def validate_agent_folder(agent_folder: str = DEFAULT_AGENTS_CONFIG_FOLDER) -> None:
    """Validate all agent configurations in a folder."""
    # Validate folder structure
    _validate_folder_structure(agent_folder)

    # Scan for agent folders
    for agent_folder_name in os.listdir(agent_folder):
        agent_path = os.path.join(agent_folder, agent_folder_name)

        if not os.path.isdir(agent_path):
            continue

        config_path = os.path.join(agent_path, "config.yml")

        # Validate config file exists
        _validate_config_file_exists(config_path, agent_folder_name)

        # Read and validate the config content
        try:
            # First read the raw YAML data to validate structure
            data = read_config_file(config_path)

            # Validate no additional keys
            _validate_no_additional_keys_raw_data(data)

            # Validate mandatory fields before creating Pydantic models
            _validate_mandatory_fields(data, agent_folder_name)

            agent_config = AvailableAgents.from_dict(data)

            # Validate the agent config (protocol-specific and endpoint references)
            validate_agent_config(agent_config)
        except PydanticValidationError as e:
            _handle_pydantic_validation_error(e, agent_folder_name)
        except Exception as e:
            # Handle non-Pydantic exceptions
            raise ValidationError(
                code="agent.validation.folder.config_validation_failed",
                event_info=f"Agent '{agent_folder_name}' validation failed: {e}",
            )


def validate_agent_config_data(data: Dict[str, Any]) -> "AgentConfig":
    """Validate agent configuration data."""
    try:
        # Validate no additional keys before creating Pydantic models
        _validate_no_additional_keys_raw_data(data)

        # Create the agent config (this will trigger Pydantic validation)
        agent_config = AgentConfig(
            agent=AgentInfo(**data.get("agent", {})),
            configuration=AgentConfiguration(**data.get("configuration", {}))
            if data.get("configuration")
            else None,
            connections=AgentConnections(**data.get("connections", {}))
            if data.get("connections")
            else None,
        )

        return agent_config

    except PydanticValidationError as e:
        _handle_pydantic_validation_error(e, "Agent configuration")
    except Exception:
        # Re-raise other exceptions
        raise
