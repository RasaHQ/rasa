from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from rasa.core.config.available_endpoints import MCPServerConfig
from rasa.core.config.configuration import Configuration
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.shared.core.flows.yaml_flows_io import FLOWS_SCHEMA_FILE
from rasa.shared.utils.yaml import (
    YamlValidationException,
    validate_yaml_with_jsonschema,
)


@pytest.fixture
def basic_call_step() -> CallFlowStep:
    """Create a basic CallFlowStep for testing."""
    return CallFlowStep(
        custom_id="test_call_step",
        idx=0,
        description="Test call step",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call="test_flow_to_call",
    )


@pytest.fixture
def mcp_call_step() -> CallFlowStep:
    """Create a CallFlowStep configured for MCP tool calling."""
    return CallFlowStep(
        custom_id="mcp_call_step",
        idx=1,
        description="MCP tool call step",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call="test_tool",
        mcp_server="test_server",
        mapping={"input": "value", "output": "result"},
    )


@pytest.fixture
def agent_call_step() -> CallFlowStep:
    """Create a CallFlowStep configured for agent calling."""
    return CallFlowStep(
        custom_id="agent_call_step",
        idx=2,
        description="Agent call step",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call="test_agent",
    )


@pytest.mark.parametrize(
    "step_fixture,expected_flow,expected_mcp,expected_agent,agents_dict",
    [
        ("basic_call_step", True, False, False, {"other_agent": {}}),
        ("mcp_call_step", False, True, False, {"test_agent": {}}),
        ("agent_call_step", False, False, True, {"test_agent": {}}),
    ],
)
def test_is_calling_methods_mutually_exclusive(
    step_fixture: str,
    expected_flow: bool,
    expected_mcp: bool,
    expected_agent: bool,
    agents_dict: Dict[str, Any],
    request,
):
    """Test that the three is_calling methods are mutually exclusive."""
    step = request.getfixturevalue(step_fixture)

    with patch.object(Configuration, "get_instance") as mock_get_instance:
        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            MCPServerConfig(name="test_server", url="http://test_server", type="http")
        ]
        mock_get_instance.return_value = mock_instance

        mock_available_agents = MagicMock()
        mock_available_agents.agents = agents_dict
        mock_instance.available_agents = mock_available_agents

        assert step.is_calling_flow() == expected_flow
        assert step.is_calling_mcp_tool() == expected_mcp
        assert step.is_calling_agent() == expected_agent


@pytest.mark.parametrize(
    "call,mcp_server,mapping,expected_flow,expected_mcp,expected_agent",
    [
        ("test_flow", None, None, True, False, False),
        ("test_tool", "test_server", {"input": "value"}, False, True, False),
        ("test_flow", "test_server", None, True, False, False),
        ("test_flow", None, {"input": "value"}, True, False, False),
        ("test_tool", "test_server", {}, False, True, False),
        ("test_agent", None, None, False, False, True),
    ],
)
def test_is_calling_flow_and_mcp_tool_conditions(
    call: str,
    mcp_server: str,
    mapping: Dict[str, Any],
    expected_flow: bool,
    expected_mcp: bool,
    expected_agent: bool,
):
    """Test is_calling_flow and is_calling_mcp_tool with various configurations."""
    step = CallFlowStep(
        custom_id="test_step",
        idx=0,
        description="Test step",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call=call,
        mcp_server=mcp_server,
        mapping=mapping,
    )

    # Mock the AvailableEndpoints singleton
    with patch.object(Configuration, "get_instance") as mock_get_instance:
        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            MCPServerConfig(name="test_server", url="http://test_server", type="http")
        ]
        mock_get_instance.return_value = mock_instance

        # Mock available agents via Configuration singleton instance
        mock_available_agents = MagicMock()
        mock_available_agents.agents = {"test_agent": {}}
        mock_instance.available_agents = mock_available_agents

        assert step.is_calling_flow() == expected_flow
        assert step.is_calling_mcp_tool() == expected_mcp
        assert step.is_calling_agent() == expected_agent


@pytest.mark.parametrize(
    "call,agents_dict,expected_agent",
    [
        ("test_agent", {"test_agent": {}}, True),
        ("test_agent", {"other_agent": {}}, False),
        ("test_agent", {}, False),
        ("", {"": {}}, True),
        ("   ", {"   ": {}}, True),
        ("test_agent", {"TEST_AGENT": {}}, False),  # Case sensitive
    ],
)
def test_is_calling_agent_conditions(
    call: str, agents_dict: Dict[str, Any], expected_agent: bool
):
    """Test is_calling_agent with various agent configurations."""
    step = CallFlowStep(
        custom_id="test_step",
        idx=0,
        description="Test step",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call=call,
    )

    with patch.object(Configuration, "get_instance") as mock_get_instance:
        cfg = MagicMock()
        cfg.available_agents = MagicMock()
        cfg.available_agents.agents = agents_dict
        mock_get_instance.return_value = cfg

        assert step.is_calling_agent() == expected_agent


@pytest.mark.parametrize(
    "data,expected_call,expected_mcp_server,expected_mapping,expected_id,"
    "expected_description",
    [
        (
            {
                "call": "test_tool",
                "mcp_server": "test_server",
                "mapping": {"input": "value", "output": "result"},
                "id": "test_step",
                "description": "Test step",
            },
            "test_tool",
            "test_server",
            {"input": "value", "output": "result"},
            "test_step",
            "Test step",
        ),
        (
            {
                "call": "test_flow",
                "id": "test_step",
            },
            "test_flow",
            None,
            None,
            "test_step",
            None,
        ),
        (
            {
                "call": "test_flow",
            },
            "test_flow",
            None,
            None,
            None,
            None,
        ),
    ],
)
def test_from_json_variations(
    data: Dict[str, Any],
    expected_call: str,
    expected_mcp_server: str,
    expected_mapping: Dict[str, Any],
    expected_id: str,
    expected_description: str,
):
    """Test creating CallFlowStep from JSON with various configurations."""
    step = CallFlowStep.from_json("test_flow", data)

    assert step.call == expected_call
    assert step.mcp_server == expected_mcp_server
    assert step.mapping == expected_mapping
    assert step.custom_id == expected_id
    assert step.description == expected_description


@pytest.mark.parametrize(
    "step_fixture,expected_call,expected_mcp_server,expected_mapping",
    [
        (
            "mcp_call_step",
            "test_tool",
            "test_server",
            {"input": "value", "output": "result"},
        ),
        ("basic_call_step", "test_flow_to_call", None, None),
    ],
)
def test_as_json_variations(
    step_fixture: str,
    expected_call: str,
    expected_mcp_server: str,
    expected_mapping: Dict[str, Any],
    request,
):
    """Test serializing CallFlowStep to JSON with various configurations."""
    with patch.object(Configuration, "get_instance") as mock_get_instance:
        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            MCPServerConfig(name="test_server", url="http://test_server", type="http")
        ]
        mock_get_instance.return_value = mock_instance

        step = request.getfixturevalue(step_fixture)
        json_data = step.as_json()

        assert json_data["call"] == expected_call
        if expected_mcp_server is not None:
            assert json_data["mcp_server"] == expected_mcp_server
        if expected_mapping is not None:
            assert json_data["mapping"] == expected_mapping


def test_round_trip_serialization():
    """Test that serialization and deserialization preserves all data."""
    original_step = CallFlowStep(
        custom_id="round_trip_test",
        idx=12,
        description="Round trip test",
        metadata={"test": "data"},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call="test_flow",
        mcp_server="test_server",
        mapping={"input": "value"},
    )

    with patch.object(Configuration, "get_instance") as mock_get_instance:
        mock_instance = MagicMock()
        mock_instance.endpoints.mcp_servers = [
            MCPServerConfig(name="test_server", url="http://test_server", type="http")
        ]
        mock_get_instance.return_value = mock_instance

        json_data = original_step.as_json()
        reconstructed_step = CallFlowStep.from_json("test_flow", json_data)

        # Set the idx to match since it's not part of serialization
        reconstructed_step.idx = original_step.idx

        assert reconstructed_step.call == original_step.call
        assert reconstructed_step.mcp_server == original_step.mcp_server
        assert reconstructed_step.mapping == original_step.mapping
        assert reconstructed_step.custom_id == original_step.custom_id
        assert reconstructed_step.description == original_step.description


@pytest.mark.parametrize(
    "invalid_call_data",
    [
        # missing 'call'
        """
        flows:
          flow_invalid:
            description: Invalid flow example
            name: invalid_flow
            steps:
              - mcp_server: server
        """,
        # mapping present but not a dict
        """
        flows:
          flow_invalid:
            description: Invalid flow example
            name: invalid_flow
            steps:
              - call: my_tool
                mcp_server: server
                mapping: not_a_dict
        """,
        # mcp_server present but mapping missing
        """
        flows:
          flow_invalid:
            description: Invalid flow example
            name: invalid_flow
            steps:
              - call: my_tool
                mcp_server: server
        """,
        # mapping present but mcp_server missing
        """
        flows:
          flow_invalid:
            description: Invalid flow example
            name: invalid_flow
            steps:
              - call: my_tool
                mapping:
                  input:
                    - slot: foo
                      param: bar
                  output:
                    - slot: bazz
                      value: result.bazz
        """,
        # invalid output mapping structure
        """
        flows:
          valid_flow:
            description: Valid flow example
            name: valid_flow
            steps:
              - call: mcp_tool
                mcp_server: server
                mapping:
                  input:
                    - slot: foo
                      param: bar
                  output: output_slot
        """,
    ],
)
def test_call_step_schema_invalid_cases(invalid_call_data: str):
    with pytest.raises(YamlValidationException):
        validate_yaml_with_jsonschema(invalid_call_data, FLOWS_SCHEMA_FILE)


@pytest.mark.parametrize(
    "valid_call_data",
    [
        """
        flows:
          valid_flow:
            description: Valid flow example
            name: valid_flow
            steps:
              - call: another_flow
        """,
        """
        flows:
          valid_flow:
            description: Valid flow example
            name: valid_flow
            steps:
              - call: mcp_tool
                mcp_server: server
                mapping:
                  input:
                    - slot: foo
                      param: bar
                  output:
                    - slot: bazz
                      value: result.bazz
        """,
    ],
)
def test_call_step_schema_valid_cases(valid_call_data: str):
    validate_yaml_with_jsonschema(valid_call_data, FLOWS_SCHEMA_FILE)
