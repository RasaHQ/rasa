"""Tests for attribute extraction functions in tracing."""

import json
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.tracing.instrumentation.attribute_extractors import (
    extract_attrs_for_datetime_configuration,
    extract_attrs_for_enterprise_search_invoke_llm,
    extract_attrs_for_llm_based_command_generator,
    extract_attrs_for_mcp_agent_llm_call,
    extract_call_flow_step_attributes,
)


@pytest.fixture
def base_call_flow_step() -> CallFlowStep:
    """Base CallFlowStep fixture with common attributes."""
    return CallFlowStep(
        idx=1,
        custom_id="test_step",
        description="Test call step",
        call="test_target",
        flow_id="test_flow",
        metadata={},
        next=FlowStepLinks(links=[]),
    )


@pytest.mark.parametrize(
    ("step_config", "expected_attrs"),
    [
        # MCPTaskAgent case - tests exit_if extraction
        (
            {
                "call": "mcp_task_agent",
                "exit_if": ["slots.selected_appointment_slot is not null"],
            },
            {
                "step_type": "CallFlowStep",
                "call_target": "mcp_task_agent",
                "exit_if_conditions": json.dumps(
                    ["slots.selected_appointment_slot is not null"], sort_keys=True
                ),
                "absent_attrs": ["mcp_server", "mapping_config"],
            },
        ),
        # MCPOpenAgent/A2AAgent case - tests basic agent call
        (
            {
                "call": "mcp_open_or_a2a_agent",
            },
            {
                "step_type": "CallFlowStep",
                "call_target": "mcp_open_or_a2a_agent",
                "absent_attrs": ["mcp_server", "mapping_config", "exit_if_conditions"],
            },
        ),
        # MCP Tool case - tests mcp_server and mapping extraction
        (
            {
                "call": "mcp_tool",
                "mcp_server": "appointment-booking",
                "mapping": {
                    "input": [
                        {
                            "param": "appointment_slot",
                            "slot": "selected_appointment_slot",
                        }
                    ],
                    "output": [
                        {
                            "slot": "appointment_confirmed",
                            "result_key": (
                                "result.structuredContent.appointment_confirmed"
                            ),
                        }
                    ],
                },
            },
            {
                "step_type": "CallFlowStep",
                "call_target": "mcp_tool",
                "mcp_server": "appointment-booking",
                "mapping_config": json.dumps(
                    {
                        "input": [
                            {
                                "param": "appointment_slot",
                                "slot": "selected_appointment_slot",
                            }
                        ],
                        "output": [
                            {
                                "slot": "appointment_confirmed",
                                "result_key": (
                                    "result.structuredContent.appointment_confirmed"
                                ),
                            }
                        ],
                    },
                    sort_keys=True,
                ),
                "absent_attrs": ["exit_if_conditions"],
            },
        ),
    ],
)
def test_call_flow_step_attributes(
    base_call_flow_step: CallFlowStep,
    step_config: Dict[str, Any],
    expected_attrs: Dict[str, Any],
) -> None:
    """Test CallFlowStep attribute extraction for different agent types."""
    # Update the base step with the specific configuration
    for key, value in step_config.items():
        setattr(base_call_flow_step, key, value)

    attrs: Dict[str, Any] = extract_call_flow_step_attributes(base_call_flow_step)

    # Check expected attributes are present
    for attr_name, expected_value in expected_attrs.items():
        if attr_name == "absent_attrs":
            continue
        assert attrs[attr_name] == expected_value

    # Check expected absent attributes are not present
    absent_attrs: List[str] = expected_attrs.get("absent_attrs", [])
    for attr_name in absent_attrs:
        assert attr_name not in attrs


def test_mapping_config_serialization() -> None:
    """Test that mapping configuration is properly serialized to JSON."""
    step: CallFlowStep = CallFlowStep(
        idx=1,
        custom_id="call_mcp_tool",
        description="Call MCP tool",
        call="mcp_tool",
        mcp_server="appointment-booking",
        mapping={
            "input": [
                {"param": "appointment_slot", "slot": "selected_appointment_slot"}
            ],
            "output": [
                {
                    "slot": "appointment_confirmed",
                    "result_key": "result.structuredContent.appointment_confirmed",
                }
            ],
        },
        flow_id="test_flow",
        metadata={},
        next=FlowStepLinks(links=[]),
    )

    attrs: Dict[str, Any] = extract_call_flow_step_attributes(step)

    # Verify mapping config is properly serialized and contains expected structure
    mapping_config: Dict[str, Any] = json.loads(attrs["mapping_config"])
    assert mapping_config["input"][0]["param"] == "appointment_slot"
    assert mapping_config["input"][0]["slot"] == "selected_appointment_slot"
    assert mapping_config["output"][0]["slot"] == "appointment_confirmed"
    assert (
        mapping_config["output"][0]["result_key"]
        == "result.structuredContent.appointment_confirmed"
    )


@pytest.mark.parametrize(
    ("component_attrs", "expected_attrs"),
    [
        # Component with public attributes
        # LLM command generators, EnterpriseSearchPolicy
        (
            {"include_date_time": True, "timezone": "America/New_York"},
            {"include_date_time": "True", "timezone": "America/New_York"},
        ),
        # Component with public attributes, different values
        (
            {"include_date_time": False, "timezone": "Europe/London"},
            {"include_date_time": "False", "timezone": "Europe/London"},
        ),
        # Component with private attributes (MCPBaseAgent)
        (
            {"_include_date_time": True, "_timezone": "Asia/Tokyo"},
            {"include_date_time": "True", "timezone": "Asia/Tokyo"},
        ),
        # Component with only include_date_time
        (
            {"include_date_time": True},
            {"include_date_time": "True"},
        ),
        # Component with only timezone
        (
            {"timezone": "UTC"},
            {"timezone": "UTC"},
        ),
        # Component with no datetime attributes
        (
            {},
            {},
        ),
    ],
)
def test_extract_attrs_for_datetime_configuration(
    component_attrs: Dict[str, Any],
    expected_attrs: Dict[str, Any],
) -> None:
    """Test datetime configuration extraction from different component types."""
    component = type("Component", (), {})()
    for attr_name, attr_value in component_attrs.items():
        setattr(component, attr_name, attr_value)

    # When
    result = extract_attrs_for_datetime_configuration(component)

    # Then
    assert result == expected_attrs


def test_extract_attrs_for_datetime_configuration_prioritizes_public_attrs() -> None:
    """Test that public attributes are preferred over private attributes."""
    # Given
    component = Mock()
    component.include_date_time = True
    component.timezone = "UTC"
    component._include_date_time = False
    component._timezone = "America/New_York"

    # When
    result = extract_attrs_for_datetime_configuration(component)

    # Then - should use public attributes
    assert result == {"include_date_time": "True", "timezone": "UTC"}


def test_extract_attrs_for_llm_based_command_generator_includes_datetime_config() -> (
    None
):
    """Test that LLMCommandGenerator extractor includes datetime configuration."""
    from rasa.dialogue_understanding.generator import LLMCommandGenerator

    # Given
    component = Mock(spec=LLMCommandGenerator)
    component.include_date_time = True
    component.timezone = "Europe/Berlin"
    component.trace_prompt_tokens = False
    component.get_default_llm_config.return_value = {"model": "test-model"}

    with patch(
        "rasa.tracing.instrumentation.attribute_extractors.extract_llm_config"
    ) as mock_extract_llm:
        mock_extract_llm.return_value = {
            "llm_model": "test-model",
            "llm_type": "openai",
        }

        # When
        result = extract_attrs_for_llm_based_command_generator(component, "test prompt")

        # Then
        assert "include_date_time" in result
        assert "timezone" in result
        assert result["include_date_time"] == "True"
        assert result["timezone"] == "Europe/Berlin"


def test_extract_attrs_for_enterprise_search_invoke_llm_includes_datetime_config() -> (
    None
):
    """Test that EnterpriseSearchPolicy extractor includes datetime configuration."""
    from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy

    # Given
    component = Mock(spec=EnterpriseSearchPolicy)
    component.include_date_time = False
    component.timezone = "America/Los_Angeles"
    component.trace_prompt_tokens = False

    with patch(
        "rasa.tracing.instrumentation.attribute_extractors.extract_llm_config"
    ) as mock_extract_llm:
        mock_extract_llm.return_value = {
            "llm_model": "test-model",
            "llm_type": "openai",
        }

        # When
        result = extract_attrs_for_enterprise_search_invoke_llm(
            component, "test prompt"
        )

        # Then
        assert "include_date_time" in result
        assert "timezone" in result
        assert result["include_date_time"] == "False"
        assert result["timezone"] == "America/Los_Angeles"


def test_extract_attrs_for_mcp_agent_llm_call_includes_datetime_config() -> None:
    """Test that MCP agent extractor includes datetime configuration."""
    from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
    from rasa.agents.schemas import AgentInput

    # Given
    component = Mock(spec=MCPBaseAgent)
    component._include_date_time = True
    component._timezone = "Asia/Singapore"
    # Set up llm_client as a mock with config attribute
    component.llm_client = Mock()
    component.llm_client.config = {"model": "test-model"}
    component.build_messages_for_llm_request.return_value = [
        {"role": "user", "content": "test"}
    ]

    agent_input = Mock(spec=AgentInput)

    with (
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extract_llm_config"
        ) as mock_extract_llm,
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extend_attributes_with_prompt_tokens_length_for_mcp_agent"
        ) as mock_extend_tokens,
    ):
        mock_extract_llm.return_value = {
            "llm_model": "test-model",
            "llm_type": "openai",
        }

        # The extend function should preserve the datetime attributes that were added
        def side_effect(self, attributes, messages):
            return attributes

        mock_extend_tokens.side_effect = side_effect

        # When
        result = extract_attrs_for_mcp_agent_llm_call(component, agent_input)

        # Then
        assert "include_date_time" in result
        assert "timezone" in result
        assert result["include_date_time"] == "True"
        assert result["timezone"] == "Asia/Singapore"
        assert result["prompt_messages_count"] == 1
