"""Tests for attribute extraction functions in tracing."""

import json
from typing import Any, Dict, List

import pytest

from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.tracing.instrumentation.attribute_extractors import (
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
