import os
import tempfile
from textwrap import dedent
from typing import Any, Callable, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from rasa.agents.validation import validate_agent_folder
from rasa.cli.validation.bot_config import _validate_sub_agents
from rasa.core.config.configuration import Configuration
from rasa.exceptions import ValidationError


def create_agent_config(temp_dir: str, agent_name: str, config_content: str) -> str:
    """Helper function to create an agent configuration."""
    agent_dir = os.path.join(temp_dir, agent_name)
    os.makedirs(agent_dir)
    with open(os.path.join(agent_dir, "config.yml"), "w") as f:
        f.write(config_content)
    return agent_dir


def create_mock_endpoints(
    temp_dir: str,
    mcp_servers: Optional[List[str]] = None,
    model_groups: Optional[List[str]] = None,
) -> None:
    """Helper function to create mock endpoints configuration."""
    endpoints_path = os.path.join(temp_dir, "endpoints.yml")
    mock_endpoints = dedent("""
        action_endpoint:
          actions_module: "actions"
    """)

    if mcp_servers:
        mock_endpoints += "\nmcp_servers:\n"
        for server in mcp_servers:
            mock_endpoints += f"  - name: {server}\n"

    if model_groups:
        mock_endpoints += "\nmodel_groups:\n"
        for group in model_groups:
            mock_endpoints += f"  - id: {group}\n"

    with open(endpoints_path, "w") as f:
        f.write(mock_endpoints)


@pytest.mark.parametrize(
    "test_name,config_content",
    [
        (
            "valid_mcp_agent",
            dedent("""
                agent:
                  name: "valid_mcp_agent"
                  protocol: "RASA"
                  description: "A valid MCP agent for testing"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
            """),
        ),
        (
            "valid_mcp_agent_no_protocol",
            dedent("""
                agent:
                  name: "valid_mcp_agent_no_protocol"
                  description: "A valid MCP agent without explicit protocol"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
            """),
        ),
        (
            "valid_a2a_agent",
            dedent("""
                agent:
                  name: "valid_a2a_agent"
                  protocol: "A2A"
                  description: "A valid A2A agent for testing"
                configuration:
                  agent_card: "https://example.com/agent-card.json"
            """),
        ),
        (
            "valid_mcp_agent_lowercase_protocol",
            dedent("""
                agent:
                  name: "valid_mcp_agent_lowercase"
                  protocol: "rasa"
                  description: "A valid MCP agent with lowercase protocol"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
            """),
        ),
    ],
)
def test_validate_sub_agents_valid_configs(test_name: str, config_content: str) -> None:
    """Test validation with valid agent configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_agent_config(temp_dir, test_name, config_content)
        _validate_sub_agents(temp_dir)


@pytest.mark.parametrize(
    "test_name,config_content,expected_error_patterns",
    [
        (
            "invalid_mcp_missing_connections",
            dedent("""
                agent:
                  name: "invalid_mcp_agent"
                  protocol: "RASA"
                  description: "An invalid MCP agent missing connections"
            """),
            ["connections.mcp_servers"],
        ),
        (
            "invalid_mcp_empty_servers_list",
            dedent("""
                agent:
                  name: "invalid_mcp_agent"
                  protocol: "RASA"
                  description: "An invalid MCP agent with empty servers list"
                connections:
                  mcp_servers: []
            """),
            ["at least one MCP server configured"],
        ),
        (
            "invalid_a2a_missing_agent_card",
            dedent("""
                agent:
                  name: "invalid_a2a_agent"
                  protocol: "A2A"
                  description: "An invalid A2A agent missing agent_card"
            """),
            ["configuration.agent_card"],
        ),
        (
            "missing_mandatory_fields",
            dedent("""
                agent:
                  description: "An agent missing mandatory fields"
            """),
            ["Missing mandatory fields"],
        ),
        (
            "valid_mcp_agent_lowercase_protocol",
            dedent("""
                agent:
                  name: "valid_mcp_agent_lowercase"
                  protocol: "rasa"
                  description: "A valid MCP agent with lowercase protocol"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
            """),
            ["Invalid protocol"],
        ),
    ],
)
def test_validate_sub_agents_invalid_configs(
    test_name: str, config_content: str, expected_error_patterns: List[str]
) -> None:
    """Test validation fails for various invalid agent configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_agent_config(temp_dir, test_name, config_content)
        result = _validate_sub_agents(temp_dir)
        assert result is False


@pytest.mark.parametrize(
    "test_name,test_input,expected_result,setup_func",
    [
        ("nonexistent_directory", "nonexistent_directory", False, None),
        ("empty_directory", None, True, None),
        (
            "missing_config_yml",
            None,
            False,
            lambda temp_dir: (
                os.makedirs(os.path.join(temp_dir, "agent_without_config")),
                open(
                    os.path.join(temp_dir, "agent_without_config", "other_file.txt"),
                    "w",
                ).write("some content"),
            ),
        ),
    ],
)
def test_validate_sub_agents_directory_scenarios(
    test_name: str,
    test_input: Optional[str],
    expected_result: bool,
    setup_func: Optional[Callable[[str], Any]],
) -> None:
    """Test validation behavior with various directory scenarios."""
    if test_input is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            if setup_func:
                setup_func(temp_dir)
            result = _validate_sub_agents(temp_dir)
            assert result is expected_result
    else:
        result = _validate_sub_agents(test_input)
        assert result is expected_result


@pytest.mark.parametrize(
    "protocol_input",
    ["rasa", "RASA", "a2a", "A2A"],
)
def test_validate_sub_agents_protocol_normalization(protocol_input: str) -> None:
    """Test that protocol names are properly normalized."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_content = dedent(f"""
            agent:
              name: "agent_{protocol_input}"
              protocol: "{protocol_input}"
              description: "Test agent with {protocol_input} protocol"
            connections:
              mcp_servers:
                - name: "test_server"
        """)
        create_agent_config(temp_dir, f"agent_{protocol_input}", config_content)
        _validate_sub_agents(temp_dir)


@pytest.mark.parametrize(
    "test_name,config_content,expected_error_patterns",
    [
        (
            "invalid_prompt_template",
            dedent("""
                agent:
                  name: "invalid_prompt_mcp_agent"
                  protocol: "RASA"
                  description: "An MCP agent with non-existent prompt template"
                configuration:
                  prompt_template: "non_existent_prompt.jinja2"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
            """),
            ["prompt template", "does not exist"],
        ),
        (
            "invalid_module_path",
            dedent("""
                agent:
                  name: "invalid_module_agent"
                  protocol: "RASA"
                  description: "An agent with non-existent module path"
                configuration:
                  module: "non_existent_module.py"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
            """),
            ["module", "could not be imported"],
        ),
    ],
)
def test_validate_sub_agents_optional_keys_validation_failures(
    test_name: str, config_content: str, expected_error_patterns: List[str]
) -> None:
    """Test validation fails for non-existent optional key paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_agent_config(temp_dir, test_name, config_content)
        with pytest.raises(ValidationError) as exc_info:
            validate_agent_folder(temp_dir)
        error_message = str(exc_info.value)
        for pattern in expected_error_patterns:
            assert pattern in error_message


def test_validate_prompt_template_success() -> None:
    """Test validation succeeds for valid prompt template file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid prompt template file
        prompt_file = os.path.join(temp_dir, "valid_template.jinja2")
        with open(prompt_file, "w") as f:
            f.write("This is a valid prompt template")

        config_content = dedent("""
            agent:
              name: "valid_prompt_agent"
              protocol: "RASA"
              description: "An MCP agent with valid prompt template"
            configuration:
              prompt_template: "{file_path}"
            connections:
              mcp_servers:
                - name: "test_mcp_server"
        """).format(file_path=prompt_file)

        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            type("MCPServerConfig", (), {"name": "test_mcp_server"})()
        ]
        mock_instance.endpoints.model_groups = []

        with patch.object(Configuration, "get_instance", return_value=mock_instance):
            create_agent_config(temp_dir, "valid_prompt", config_content)
            validate_agent_folder(temp_dir)


def test_validate_module_success() -> None:
    """Test validation succeeds for valid Python module path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_content = dedent("""
            agent:
              name: "valid_module_agent"
              protocol: "RASA"
              description: "An agent with valid module path"
            configuration:
              module: "os.path"
            connections:
              mcp_servers:
                - name: "test_mcp_server"
        """)

        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            type("MCPServerConfig", (), {"name": "test_mcp_server"})()
        ]
        mock_instance.endpoints.model_groups = []

        with patch.object(Configuration, "get_instance", return_value=mock_instance):
            create_agent_config(temp_dir, "valid_module", config_content)
            validate_agent_folder(temp_dir)


@pytest.mark.parametrize(
    "test_name,config_content,expected_error_patterns",
    [
        (
            "additional_agent_keys",
            dedent("""
                agent:
                  name: "additional_agent_keys_agent"
                  protocol: "RASA"
                  description: "An agent with additional keys in agent section"
                  version: "1.0.0"
                  author: "Test Author"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
            """),
            ["additional keys in 'agent' section", "version", "author"],
        ),
        (
            "additional_configuration_keys",
            dedent("""
                agent:
                  name: "additional_config_keys_agent"
                  protocol: "RASA"
                  description: "An agent with additional keys in configuration section"
                configuration:
                  llm:
                    type: "openai"
                    model: "gpt-4"
                  custom_setting: "value"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
            """),
            ["additional keys in 'configuration' section", "custom_setting"],
        ),
    ],
)
def test_validate_sub_agents_no_additional_keys_failures(
    test_name: str, config_content: str, expected_error_patterns: List[str]
) -> None:
    """Test validation fails for additional keys in various sections."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_agent_config(temp_dir, test_name, config_content)
        with pytest.raises(ValidationError) as exc_info:
            validate_agent_folder(temp_dir)
        error_message = str(exc_info.value)
        for pattern in expected_error_patterns:
            assert pattern in error_message


@pytest.mark.parametrize(
    "test_name,config_content,expected_error_patterns",
    [
        (
            "invalid_model_group_reference",
            dedent("""
                agent:
                  name: "invalid_model_group_agent"
                  protocol: "RASA"
                  description: "An agent with invalid model group reference"
                configuration:
                  llm:
                    model_group: "non_existent_model_group"
                connections:
                  mcp_servers:
                    - name: "test_mcp_server"
                        """),
            ["model group", "DOES NOT EXIST in the endpoints.yml file"],
        ),
        (
            "invalid_mcp_server_reference",
            dedent("""
                agent:
                  name: "invalid_mcp_server_agent"
                  protocol: "RASA"
                  description: "An agent with invalid MCP server reference"
                connections:
                  mcp_servers:
                    - name: "non_existent_mcp_server"
                        """),
            ["MCP server", "does not exist in endpoints.yml"],
        ),
    ],
)
def test_validate_sub_agents_endpoint_references_failures(
    test_name: str, config_content: str, expected_error_patterns: List[str]
) -> None:
    """Test validation fails for invalid endpoint references."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_mock_endpoints(temp_dir, ["test_mcp_server"], ["valid_model_group"])

        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            type("MCPServerConfig", (), {"name": "test_mcp_server"})(),
        ]
        mock_instance.endpoints.model_groups = [{"id": "valid_model_group"}]

        with patch.object(Configuration, "get_instance", return_value=mock_instance):
            create_agent_config(temp_dir, test_name, config_content)
            with pytest.raises(ValidationError) as exc_info:
                validate_agent_folder(temp_dir)
            error_message = str(exc_info.value)
            for pattern in expected_error_patterns:
                assert pattern in error_message


@pytest.mark.parametrize(
    "test_name,config_content",
    [
        (
            "valid_mcp_agent",
            dedent("""
                agent:
                  name: "valid_mcp_agent"
                  protocol: "RASA"
                  description: "A valid MCP agent with valid endpoint references"
                connections:
                  mcp_servers:
                    - name: "valid_mcp_server"
            """),
        ),
        (
            "valid_llm_agent",
            dedent("""
                agent:
                  name: "valid_llm_agent"
                  protocol: "RASA"
                  description: "A valid agent with valid model group reference"
                configuration:
                  llm:
                    model_group: "valid_model_group"
                connections:
                  mcp_servers:
                    - name: "valid_mcp_server"
            """),
        ),
    ],
)
def test_validate_sub_agents_endpoint_references_success(
    test_name: str, config_content: str
) -> None:
    """Test validation succeeds for valid endpoint references."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_mock_endpoints(temp_dir, ["valid_mcp_server"], ["valid_model_group"])

        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            type("MCPServerConfig", (), {"name": "valid_mcp_server"})(),
        ]
        mock_instance.endpoints.model_groups = [{"id": "valid_model_group"}]

        with patch.object(Configuration, "get_instance", return_value=mock_instance):
            create_agent_config(temp_dir, test_name, config_content)
            validate_agent_folder(temp_dir)


def test_auth_key_in_allowed_keys() -> None:
    """Test that 'auth' key is included in ALLOWED_KEYS for configuration section."""
    from rasa.agents.validation import ALLOWED_KEYS

    assert "auth" in ALLOWED_KEYS["configuration"]


def test_validate_agent_with_auth_configuration() -> None:
    """Test validation succeeds for agent configuration with auth key."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_content = dedent("""
            agent:
              name: "agent_with_auth"
              protocol: "A2A"
              description: "An agent with auth configuration"
            configuration:
              agent_card: "https://example.com/agent-card.json"
              auth:
                token: "test_token"
            connections:
              mcp_servers:
                - name: "test_mcp_server"
        """)

        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            type("MCPServerConfig", (), {"name": "test_mcp_server"})()
        ]
        mock_instance.endpoints.model_groups = []

        with patch.object(Configuration, "get_instance", return_value=mock_instance):
            create_agent_config(temp_dir, "agent_with_auth", config_content)
            # This should not raise any validation errors
            validate_agent_folder(temp_dir)
