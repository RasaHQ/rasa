from unittest.mock import Mock

import pytest


@pytest.fixture(scope="function")
def mock_mcp_tool():
    """Create a mock MCP Tool for testing."""
    tool = Mock()
    tool.name = "test_mcp_tool"
    tool.description = "Test MCP tool description"
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "param1": {"description": "First parameter"},
            "param2": {"description": "Second parameter"},
        },
    }
    return tool


@pytest.fixture(scope="function")
def mock_openai_tool():
    """Create a mock OpenAI tool format for testing."""
    return {
        "type": "function",
        "function": {
            "name": "test_openai_tool",
            "description": "Test OpenAI tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"description": "First parameter"},
                    "param2": {"description": "Second parameter"},
                },
            },
        },
    }


@pytest.fixture(scope="function")
def sample_parameters():
    """Create sample parameters for testing."""
    return {
        "type": "object",
        "properties": {
            "simple_param": {"description": "Simple parameter"},
            "object_param": {
                "type": "object",
                "properties": {"nested_param": {"description": "Nested parameter"}},
            },
        },
    }


@pytest.fixture(scope="function")
def complex_nested_parameters():
    """Create complex nested parameters for testing."""
    return {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "level3": {
                                "type": "object",
                                "properties": {
                                    "deep_param": {
                                        "description": "Deep nested parameter"
                                    }
                                },
                            }
                        },
                    }
                },
            }
        },
    }
