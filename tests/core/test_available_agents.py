from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError

from rasa.agents.exceptions import (
    AgentNameFlowConflictException,
    DuplicatedAgentNameException,
)
from rasa.agents.validation import (
    validate_agent_config,
    validate_agent_names_not_conflicting_with_flows,
    validate_agent_names_unique,
)
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentConnections,
    AgentInfo,
    AgentMCPServerConfig,
    AvailableAgents,
    ProtocolConfig,
)
from rasa.exceptions import ValidationError


@pytest.fixture
def deserialized_agent_config_mcp() -> AgentConfig:
    """Fixture to provide a deserialized MCP AgentConfig object for testing."""
    return AgentConfig(
        agent=AgentInfo(
            name="mcp_test_agent",
            # default protocol is RASA
            description="An MCP test agent for unit testing",
        ),
        configuration=AgentConfiguration(
            llm={
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 150,
            },
            prompt_template="test_prompt_template",
            module="test_module",
            timeout=30,
            max_retries=3,
        ),
        connections=AgentConnections(
            mcp_servers=[
                AgentMCPServerConfig(
                    name="test_mcp_server",
                    include_tools=["tool1", "tool2"],
                    exclude_tools=["tool3"],
                )
            ]
        ),
    )


@pytest.fixture
def deserialized_agent_config_a2a() -> AgentConfig:
    """Fixture to provide a deserialized A2A AgentConfig object for testing."""
    return AgentConfig(
        agent=AgentInfo(
            name="a2a_test_agent",
            protocol=ProtocolConfig.A2A,
            description="An A2A test agent for unit testing",
        ),
        configuration=AgentConfiguration(agent_card="test_agent_card"),
        connections=None,
    )


@pytest.fixture
def mock_read_agent_config(
    deserialized_agent_config_mcp: AgentConfig, monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Mock the `_read_agent_config` method to avoid reading from files."""
    mock = MagicMock(return_value=deserialized_agent_config_mcp)
    monkeypatch.setattr(AvailableAgents, "_read_agent_config", mock)
    return mock


@pytest.fixture
def patch_listdir(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch the os.listdir method to return a predefined list of agents."""
    monkeypatch.setattr("os.path.isdir", lambda path: True)

    def _patch(agent_list: List[str]) -> None:
        monkeypatch.setattr("os.listdir", lambda _: agent_list)

    return _patch


@pytest.mark.parametrize(
    "agent_name,expected_protocol",
    [
        ("mcp_test_agent", "RASA"),
        ("a2a_test_agent", "A2A"),
    ],
)
def test_read_agent_folder(
    monkeypatch: pytest.MonkeyPatch,
    patch_listdir: Any,
    request: pytest.FixtureRequest,
    agent_name: str,
    expected_protocol: str,
) -> None:
    """Test reading agent folders and protocol detection."""
    folder_names = ["mcp_test_agent", "a2a_test_agent"]
    patch_listdir(folder_names)
    config_map = {
        "sub_agents/mcp_test_agent/config.yml": request.getfixturevalue(
            "deserialized_agent_config_mcp"
        ),
        "sub_agents/a2a_test_agent/config.yml": request.getfixturevalue(
            "deserialized_agent_config_a2a"
        ),
    }
    monkeypatch.setattr(
        AvailableAgents,
        "_read_agent_config_file",
        MagicMock(side_effect=lambda path: config_map[path]),
    )
    monkeypatch.setattr("os.path.isfile", lambda path: True)

    agents = AvailableAgents.read_from_folder("sub_agents")
    assert isinstance(agents, AvailableAgents)
    assert agent_name in agents.agents
    assert agents.agents[agent_name].agent.protocol == expected_protocol


@pytest.mark.parametrize(
    "agent_name,agent_config_fixture,expected_protocol",
    [
        ("agent_a", "deserialized_agent_config_mcp", "RASA"),
        ("agent_b", "deserialized_agent_config_a2a", "A2A"),
    ],
)
def test_read_agent_config(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    agent_name: str,
    agent_config_fixture: str,
    expected_protocol: str,
) -> None:
    """Test reading individual agent configurations."""
    agent_config = request.getfixturevalue(agent_config_fixture)
    monkeypatch.setattr(
        AvailableAgents,
        "_read_agent_config_file",
        MagicMock(return_value=agent_config),
    )
    config = AvailableAgents._read_agent_config_file(
        f"sub_agents/{agent_name}/config.yml"
    )
    assert isinstance(config, AgentConfig)
    assert config.agent.protocol == expected_protocol


def test_read_agent_config_file_not_found() -> None:
    """Test that _read_agent_config_file raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        AvailableAgents._read_agent_config_file("nonexistent_file.yml")


def test_read_agent_config_error(
    monkeypatch: pytest.MonkeyPatch, patch_listdir: Any
) -> None:
    """Test that an error in reading agent config raises ValidationError."""
    patch_listdir(["agent_a"])
    monkeypatch.setattr(
        AvailableAgents,
        "_read_agent_config_file",
        MagicMock(side_effect=Exception("Read error")),
    )
    monkeypatch.setattr("os.path.isfile", lambda path: True)
    with pytest.raises(
        ValidationError, match="Failed to load agent 'agent_a': Read error"
    ):
        AvailableAgents.read_from_folder("sub_agents/")


def test_custom_agents_config_folder_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an error is raised when a custom agents config folder doesn't exist."""
    monkeypatch.setattr("os.path.isdir", lambda path: False)
    with pytest.raises(ValidationError, match="does not exist or is not a directory"):
        AvailableAgents.read_from_folder("non_existent_folder")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"name": "test_agent"},
        {"protocol": "RASA"},
        {"description": "example description"},
        {"name": "test_agent", "protocol": "RASA"},
        {"protocol": "RASA", "description": "example description"},
    ],
)
def test_agent_info_fields_are_mandatory(kwargs: Dict[str, Any]) -> None:
    """Test that all mandatory fields are required for AgentInfo."""
    with pytest.raises(PydanticValidationError):
        AgentInfo(**kwargs)


def test_agent_info_fields_are_mandatory_with_default_protocol() -> None:
    """Test that all mandatory fields are required with default protocol."""
    agent_info = AgentInfo(
        **{"name": "test_agent", "description": "example description"}
    )
    assert agent_info.protocol == ProtocolConfig.RASA


@pytest.mark.parametrize(
    "agent_name,protocol,should_raise,expected_error",
    [
        (
            "agent_a",
            "RASA",
            True,
            (
                "For protocol 'RASA', agent 'test_agent' must have "
                "'connections.mcp_servers' configured"
            ),
        ),
        (
            "agent_b",
            "A2A",
            True,
            (
                "For protocol 'A2A', agent 'test_agent' must have "
                "'configuration.agent_card' specified"
            ),
        ),
        ("agent_a", "RASA", False, None),
        ("agent_b", "A2A", False, None),
    ],
)
def test_agent_config_protocol_required_fields(
    patch_listdir: Any,
    agent_name: str,
    protocol: str,
    should_raise: bool,
    expected_error: Optional[str],
) -> None:
    """Test protocol-specific required field validation."""
    patch_listdir([agent_name])
    protocol_enum: ProtocolConfig = ProtocolConfig[protocol]
    if should_raise:
        agent_config = AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                protocol=protocol_enum,
                description=f"A {protocol} test agent missing config",
            ),
            configuration=None,
            connections=None,
        )
        with pytest.raises(ValidationError):
            validate_agent_config(agent_config)
    else:
        config = (
            AgentConfiguration(agent_card="test_card")
            if protocol == "A2A"
            else AgentConfiguration()
        )
        connections = None if protocol == "A2A" else AgentConnections(mcp_servers=[])
        agent_config = AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                protocol=protocol_enum,
                description=f"A {protocol} test agent",
            ),
            configuration=config,
            connections=connections,
        )
        assert isinstance(agent_config, AgentConfig)


@pytest.mark.parametrize(
    "protocol_input,expected_protocol",
    [
        # Standard case variations
        ("rasa", ProtocolConfig.RASA),
        ("RASA", ProtocolConfig.RASA),
        ("a2a", ProtocolConfig.A2A),
        ("A2A", ProtocolConfig.A2A),
        # Mixed case variations
        ("rAsa", ProtocolConfig.RASA),
        ("Rasa", ProtocolConfig.RASA),
        ("RaSa", ProtocolConfig.RASA),
        ("A2a", ProtocolConfig.A2A),
        ("a2A", ProtocolConfig.A2A),
    ],
)
def test_agent_info_protocol_case_insensitive(
    protocol_input: str, expected_protocol: ProtocolConfig
) -> None:
    """Test that AgentInfo accepts protocol names in different cases."""
    agent_info = AgentInfo(
        name="test_agent", protocol=protocol_input, description="Test agent description"
    )
    assert agent_info.protocol == expected_protocol


def test_agent_info_protocol_preserves_original_enum() -> None:
    """Test that the protocol field preserves the original enum type."""
    agent_info = AgentInfo(
        name="test_agent", protocol="rasa", description="Test agent description"
    )
    assert isinstance(agent_info.protocol, ProtocolConfig)
    assert agent_info.protocol == ProtocolConfig.RASA
    assert agent_info.protocol.value == "RASA"


@pytest.fixture
def sample_agent_configs() -> Dict[str, AgentConfig]:
    """Create sample agent configurations for testing."""
    return {
        "agent_1": AgentConfig(
            agent=AgentInfo(
                name="agent_1", protocol=ProtocolConfig.RASA, description="Test agent 1"
            )
        ),
        "agent_2": AgentConfig(
            agent=AgentInfo(
                name="agent_2", protocol=ProtocolConfig.A2A, description="Test agent 2"
            )
        ),
        "Agent_1": AgentConfig(
            agent=AgentInfo(
                name="Agent_1",
                protocol=ProtocolConfig.RASA,
                description="Test Agent 1 (case-sensitive)",
            )
        ),
        "conflicting_a": AgentConfig(
            agent=AgentInfo(
                name="conflicting_a",
                protocol=ProtocolConfig.RASA,
                description="Test conflicting agent A",
            )
        ),
        "conflicting_b": AgentConfig(
            agent=AgentInfo(
                name="conflicting_b",
                protocol=ProtocolConfig.RASA,
                description="Test conflicting agent B",
            )
        ),
    }


@pytest.mark.parametrize(
    "agents,expected_duplicates",
    [
        # Single duplicate
        (
            ["duplicate_agent", "duplicate_agent"],
            ["duplicate_agent"],
        ),
        # Multiple duplicates
        (
            ["duplicate_1", "duplicate_1", "duplicate_2", "duplicate_2"],
            ["duplicate_1", "duplicate_2"],
        ),
        # No duplicates
        (
            ["agent_1", "agent_2", "agent_3"],
            [],
        ),
    ],
)
def test_agent_names_unique_validation(
    agents: List[str],
    expected_duplicates: List[str],
    sample_agent_configs: Dict[str, AgentConfig],
) -> None:
    """Test that agent names are validated to be unique."""
    # Create agent configs with the specified names
    agent_configs = []
    for name in agents:
        agent_configs.append(
            AgentConfig(
                agent=AgentInfo(
                    name=name, protocol=ProtocolConfig.RASA, description=f"Test {name}"
                )
            )
        )

    if expected_duplicates:
        # Should raise DuplicatedAgentNameException
        with pytest.raises(DuplicatedAgentNameException) as exc_info:
            validate_agent_names_unique(agent_configs)

        assert exc_info.value.code == "agent.duplicated_name"
        assert set(exc_info.value.ctx["duplicated_names"]) == set(expected_duplicates)

        # Check that all expected duplicates are mentioned in the error message
        for duplicate in expected_duplicates:
            assert duplicate in str(exc_info.value)
    else:
        # Should not raise any exception
        validate_agent_names_unique(agent_configs)


@pytest.mark.parametrize(
    "agent_names,flow_names,expected_conflicts",
    [
        # Single conflict
        (["agent_1"], ["agent_1"], ["agent_1"]),
        # Multiple conflicts
        (
            ["conflicting_a", "conflicting_b", "agent_1"],
            ["conflicting_a", "conflicting_b", "other_flow"],
            ["conflicting_a", "conflicting_b"],
        ),
        # No conflicts
        (["agent_1", "agent_2"], ["flow_1", "flow_2"], []),
        # Empty flow names
        (["agent_1"], [], []),
        # No agents
        (
            [],
            ["customer_service"],
            [],
        ),
        # Case-sensitive conflicts
        (
            ["agent_1", "Agent_1"],
            ["Agent_1"],
            ["Agent_1"],
        ),
    ],
)
def test_agent_names_flow_conflict_validation(
    agent_names: List[str],
    flow_names: List[str],
    expected_conflicts: List[str],
    sample_agent_configs: Dict[str, AgentConfig],
) -> None:
    """Test that agent names are validated to not conflict with flow names."""
    # Create agents dictionary with the specified names
    agents = {}
    for name in agent_names:
        agents[name] = sample_agent_configs[name]

    if expected_conflicts:
        # Should raise AgentNameFlowConflictException
        with pytest.raises(AgentNameFlowConflictException) as exc_info:
            validate_agent_names_not_conflicting_with_flows(agents, set(flow_names))

        assert exc_info.value.code == "agent.flow_name_conflict"
        assert set(exc_info.value.ctx["conflicting_names"]) == set(expected_conflicts)

        # Check that all expected conflicts are mentioned in the error message
        for conflict in expected_conflicts:
            assert conflict in str(exc_info.value)
    else:
        # Should not raise any exception
        validate_agent_names_not_conflicting_with_flows(agents, set(flow_names))


class TestAgentMCPServerConfigAdditionalParams:
    """Test cases for AgentMCPServerConfig additional_params functionality."""

    def test_mcp_server_config_without_additional_params(self):
        """Test AgentMCPServerConfig creation without additional_params."""
        config = AgentMCPServerConfig(
            name="test_server",
            url="http://localhost:8000",
            type="http",
            include_tools=["tool1", "tool2"],
            exclude_tools=["tool3"],
        )

        assert config.name == "test_server"
        assert config.url == "http://localhost:8000"
        assert config.type == "http"
        assert config.include_tools == ["tool1", "tool2"]
        assert config.exclude_tools == ["tool3"]
        assert config.additional_params is None

    def test_mcp_server_config_with_simple_additional_params(self):
        """Test AgentMCPServerConfig creation with simple additional_params."""
        additional_params = {
            "timeout": 30,
            "retries": 3,
            "debug": True,
        }

        config = AgentMCPServerConfig(
            name="test_server",
            additional_params=additional_params,
        )

        assert config.name == "test_server"
        assert config.additional_params == additional_params
        assert config.additional_params["timeout"] == 30
        assert config.additional_params["retries"] == 3
        assert config.additional_params["debug"] is True

    def test_mcp_server_config_with_complex_additional_params(self):
        """Test AgentMCPServerConfig creation with complex additional_params."""
        additional_params = {
            "type": "bearer",
            "token": "secret_token",
        }

        config = AgentMCPServerConfig(
            name="test_server",
            additional_params=additional_params,
        )

        assert config.name == "test_server"
        assert config.additional_params == additional_params
        assert config.additional_params["type"] == "bearer"
        assert config.additional_params["token"] == "secret_token"


class TestAgentConfigurationAuthField:
    """Test cases for AgentConfiguration auth field functionality."""

    def test_auth_field_optional_and_defaults_to_none(self):
        """Test that auth field is optional and defaults to None."""
        config = AgentConfiguration()
        assert config.auth is None

    def test_auth_field_with_simple_dict(self):
        """Test AgentConfiguration creation with simple auth configuration."""
        auth_config = {"token": "secret_token_123"}

        config = AgentConfiguration(auth=auth_config)
        assert config.auth == auth_config
        assert config.auth["token"] == "secret_token_123"

    def test_auth_field_with_complex_dict(self):
        """Test AgentConfiguration creation with complex auth configuration."""
        auth_config = {
            "oauth": {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "scopes": ["read", "write"],
                "endpoints": {
                    "authorization": "https://auth.example.com/authorize",
                    "token": "https://auth.example.com/token",
                },
            }
        }

        config = AgentConfiguration(auth=auth_config)
        assert config.auth == auth_config
        assert config.auth["oauth"]["client_id"] == "test_client_id"
        assert config.auth["oauth"]["scopes"] == ["read", "write"]
        assert (
            config.auth["oauth"]["endpoints"]["authorization"]
            == "https://auth.example.com/authorize"
        )

    def test_auth_field_with_empty_dict(self):
        """Test AgentConfiguration creation with empty auth dictionary."""
        auth_config = {}

        config = AgentConfiguration(auth=auth_config)
        assert config.auth == auth_config
        assert isinstance(config.auth, dict)
        assert len(config.auth) == 0

    def test_auth_field_with_other_configuration_fields(self):
        """Test auth field works alongside other AgentConfiguration fields."""
        auth_config = {
            "username": "test_user",
            "password": "${PASS}",
        }

        config = AgentConfiguration(
            llm={"model": "gpt-4", "temperature": 0.7},
            prompt_template="test_prompt",
            module="test_module",
            timeout=60,
            max_retries=5,
            agent_card="test_card",
            auth=auth_config,
        )

        assert config.llm == {"model": "gpt-4", "temperature": 0.7}
        assert config.prompt_template == "test_prompt"
        assert config.module == "test_module"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.agent_card == "test_card"
        assert config.auth == auth_config

    def test_auth_field_validation_with_invalid_types(self):
        """Test that auth field validation works with invalid input types."""
        # Test with string instead of dict
        with pytest.raises(PydanticValidationError):
            AgentConfiguration(auth="invalid_string")

        # Test with list instead of dict
        with pytest.raises(PydanticValidationError):
            AgentConfiguration(auth=["item1", "item2"])

        # Test with integer instead of dict
        with pytest.raises(PydanticValidationError):
            AgentConfiguration(auth=123)
