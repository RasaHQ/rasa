"""Unit tests for AgentToolSchema."""

import pytest

from rasa.agents.constants import (
    TOOL_ADDITIONAL_PROPERTIES_KEY,
)
from rasa.agents.schemas.agent_tool_schema import AgentToolSchema


def test_basic_schema_creation():
    """Test basic schema creation with minimal data."""
    schema = AgentToolSchema(name="test_tool", parameters={}, strict=False)

    assert schema.name == "test_tool"
    assert schema.parameters == {}
    assert schema.strict is False
    assert schema.type == "function"
    assert schema.description is None


def test_schema_with_description():
    """Test schema creation with description."""
    schema = AgentToolSchema(
        name="test_tool", parameters={}, strict=False, description="A test tool"
    )

    assert schema.description == "A test tool"


def test_from_mcp_tool_basic(mock_mcp_tool):
    """Test creating schema from MCP Tool with basic data."""
    schema = AgentToolSchema.from_mcp_tool(mock_mcp_tool)

    assert schema.name == "test_mcp_tool"
    assert schema.description == "Test MCP tool description"
    assert schema.strict is False
    assert schema.type == "function"
    assert schema.parameters["type"] == "object"
    assert schema.parameters["additionalProperties"] is False
    assert schema.parameters["properties"]["param1"]["type"] == "string"


def test_from_litellm_json_format_basic(mock_openai_tool):
    """Test creating schema from LiteLLM format that is based on OpenAI JSON format."""
    schema = AgentToolSchema.from_litellm_json_format(mock_openai_tool)

    assert schema.name == "test_openai_tool"
    assert schema.description == "Test OpenAI tool description"
    assert schema.strict is False
    assert schema.type == "function"
    assert schema.parameters["properties"]["param1"]["type"] == "string"


def test_preserves_existing_required_field():
    """Test that existing required fields are preserved during validation."""
    # Create parameters with an existing required field
    parameters = {
        "type": "object",
        "properties": {
            "param1": {"description": "First parameter"},
            "param2": {"description": "Second parameter"},
        },
        "required": ["param1"],  # Only param1 is required
    }

    # Run validation
    AgentToolSchema._validate_and_fix_parameters(parameters)

    # Required field should remain as provided, not overwritten
    assert parameters["required"] == ["param1"]
    # Additional properties should be added
    assert parameters["additionalProperties"] is False
    # Property types should be added
    assert parameters["properties"]["param1"]["type"] == "string"
    assert parameters["properties"]["param2"]["type"] == "string"


def test_property_types_with_non_dict_properties():
    """Test handling non-dict properties gracefully."""
    parameters = {
        "type": "object",
        "properties": {
            "param1": "not_a_dict",
            "param2": {"description": "Valid parameter"},
        },
    }

    # Should not raise an error
    AgentToolSchema._ensure_property_types(parameters)

    # Only valid properties should be processed
    assert parameters["properties"]["param2"]["type"] == "string"
    assert "type" not in parameters["properties"]["param1"]


def test_validation_methods_modify_input_in_place():
    """Test that validation methods modify input parameters in place
    (expected behavior).
    """
    original_parameters = {
        "type": "object",
        "properties": {"test_param": {"description": "Test parameter"}},
    }

    # Run validation on the parameters
    AgentToolSchema._ensure_property_types(original_parameters)

    # Original should be modified (this is the expected behavior)
    assert "type" in original_parameters["properties"]["test_param"]
    assert original_parameters["properties"]["test_param"]["type"] == "string"


@pytest.mark.parametrize(
    "property_type,expected_additional_properties",
    [
        ("object", False),
        ("string", None),  # Should not have additionalProperties added
        ("number", None),
        ("array", None),
        ("boolean", None),
    ],
)
def test_property_types_additional_properties_handling(
    property_type, expected_additional_properties
):
    """Test that additionalProperties is only added to object types."""
    parameters = {
        "type": "object",
        "properties": {
            "test_param": {
                "type": property_type,
                "description": f"Test {property_type} parameter",
            }
        },
    }

    AgentToolSchema._ensure_property_types(parameters)

    test_param = parameters["properties"]["test_param"]
    if expected_additional_properties is None:
        assert TOOL_ADDITIONAL_PROPERTIES_KEY not in test_param
    else:
        assert (
            test_param[TOOL_ADDITIONAL_PROPERTIES_KEY] == expected_additional_properties
        )


@pytest.mark.parametrize(
    "description,expected_type",
    [("A simple parameter", "string"), ("", "string"), (None, "string")],
)
def test_missing_type_defaults_to_string(description, expected_type):
    """Test that properties without types default to string type."""
    parameters = {
        "type": "object",
        "properties": {"test_param": {"description": description}},
    }

    AgentToolSchema._ensure_property_types(parameters)

    assert parameters["properties"]["test_param"]["type"] == expected_type


def test_empty_properties_section():
    """Test handling of empty properties section."""
    parameters = {"type": "object", "properties": {}}

    # Should not raise an error
    AgentToolSchema._ensure_property_types(parameters)

    # Should remain unchanged
    assert parameters["properties"] == {}


def test_none_properties_section():
    """Test handling of None properties section."""
    parameters = {"type": "object", "properties": None}

    # Should not raise an error
    AgentToolSchema._ensure_property_types(parameters)

    # Should remain unchanged
    assert parameters["properties"] is None


def test_current_validation_behavior():
    """Test the current simplified validation behavior."""
    # Create parameters similar to the symbol_search function that was failing
    parameters = {
        "type": "object",
        "properties": {
            "exchange": {"description": "Exchange name"},
            "symbol": {"description": "Symbol to search"},
            "query": {"description": "Search query"},
        },
    }

    # Run validation
    AgentToolSchema._validate_and_fix_parameters(parameters)

    # Top level should have additionalProperties
    assert parameters["additionalProperties"] is False

    # All properties should have types
    assert parameters["properties"]["exchange"]["type"] == "string"
    assert parameters["properties"]["symbol"]["type"] == "string"
    assert parameters["properties"]["query"]["type"] == "string"


def test_from_litellm_json_format_wrong_type_value():
    """Test that tools with wrong 'type' value are accepted (current behavior)."""
    invalid_tool = {
        "type": "not_function",
        "function": {
            "name": "test_tool",
            "description": "Test tool description",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    schema = AgentToolSchema.from_litellm_json_format(invalid_tool)
    assert schema.name == "test_tool"
    assert schema.type == "not_function"


def test_from_litellm_json_format_valid_structure():
    """Test that valid LiteLLM/OpenAI format is accepted."""
    valid_tool = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "Test tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "number", "description": "Second parameter"},
                },
                "required": ["param1"],
            },
        },
    }

    schema = AgentToolSchema.from_litellm_json_format(valid_tool)

    assert schema.name == "test_tool"
    assert schema.description == "Test tool description"
    assert schema.type == "function"
    assert schema.strict is False
    assert schema.parameters["type"] == "object"
    assert schema.parameters["properties"]["param1"]["type"] == "string"
    assert schema.parameters["properties"]["param2"]["type"] == "number"
    assert schema.parameters["required"] == ["param1"]


@pytest.mark.parametrize(
    "invalid_tool,expected_error_pattern",
    [
        # Missing type key
        (
            {"function": {"name": "test", "parameters": {}}},
            "Expected a dictionary with 'type' and 'function' keys",
        ),
        # Missing function key
        (
            {"type": "function", "name": "test", "parameters": {}},
            "Expected a dictionary with 'type' and 'function' keys",
        ),
        # Anthropic format
        (
            {"name": "test", "description": "test", "input_schema": {}},
            "Anthropic Tool format is not supported yet",
        ),
        # Empty tool
        ({}, "Expected a dictionary with 'type' and 'function' keys"),
        # Missing tool name
        (
            {"type": "function", "function": {"description": "test", "parameters": {}}},
            "'function' must contain 'name' and 'description' keys",
        ),
    ],
)
def test_from_litellm_json_format_various_invalid_inputs(
    invalid_tool, expected_error_pattern
):
    """Test various invalid input formats are properly rejected."""
    with pytest.raises(ValueError, match=expected_error_pattern):
        AgentToolSchema.from_litellm_json_format(invalid_tool)


def test_from_litellm_json_format_none_input():
    """Test that None input raises TypeError."""
    with pytest.raises(TypeError, match="argument of type 'NoneType' is not iterable"):
        AgentToolSchema.from_litellm_json_format(None)
