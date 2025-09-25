"""Tests for MCP server structure validation in endpoints.yml."""

import os
import tempfile
from typing import Dict, List, Optional

import pytest
from pydantic import ValidationError

from rasa.core.config.available_endpoints import AvailableEndpoints, MCPServerConfig


def create_endpoints_file(
    temp_dir: str, mcp_servers: Optional[List[Dict[str, str]]] = None
) -> str:
    """Helper function to create an endpoints file with MCP servers."""
    endpoints_path = os.path.join(temp_dir, "endpoints.yml")

    if mcp_servers:
        mcp_servers_yaml = "mcp_servers:\n"
        for server in mcp_servers:
            mcp_servers_yaml += f"  - name: {server['name']}\n"
            if "url" in server:
                mcp_servers_yaml += f"    url: {server['url']}\n"
            if "type" in server:
                mcp_servers_yaml += f"    type: {server['type']}\n"
    else:
        mcp_servers_yaml = ""

    mock_endpoints = f"""action_endpoint:
  url: "http://localhost:5055/webhook"
{mcp_servers_yaml}"""

    with open(endpoints_path, "w") as f:
        f.write(mock_endpoints)

    return endpoints_path


@pytest.mark.parametrize(
    "mcp_servers,expected_error_patterns",
    [
        # Missing required fields
        ([{"name": "server1"}], ["url", "type", "Field required"]),
        (
            [{"name": "server1", "url": "http://localhost:8080"}],
            ["type", "Field required"],
        ),
        ([{"name": "server1", "type": "http"}], ["url", "Field required"]),
        # Invalid type
        (
            [{"name": "server1", "url": "http://localhost:8080", "type": "invalid"}],
            ["Invalid MCP server type: invalid"],
        ),
        # Empty values (YAML converts empty strings to None)
        (
            [{"name": "", "url": "http://localhost:8080", "type": "http"}],
            ["Input should be a valid string"],
        ),
        (
            [{"name": "server1", "url": "", "type": "http"}],
            ["Input should be a valid string"],
        ),
    ],
)
def test_mcp_server_structure_validation_errors(
    mcp_servers: List[Dict[str, str]], expected_error_patterns: List[str]
) -> None:
    """Test MCP server structure validation errors in endpoints.yml."""
    with tempfile.TemporaryDirectory() as temp_dir:
        endpoints_path = create_endpoints_file(temp_dir, mcp_servers)

        with pytest.raises(ValidationError) as exc_info:
            AvailableEndpoints.read_endpoints(endpoints_path)

        error_message = str(exc_info.value)
        for pattern in expected_error_patterns:
            assert pattern in error_message, (
                f"Expected pattern '{pattern}' not found in error: " f"{error_message}"
            )


def test_mcp_server_config_direct_validation() -> None:
    """Test MCPServerConfig validation directly.

    Covers edge cases not in endpoints parsing.
    """
    # Test invalid type (this is the main validation not covered elsewhere)
    with pytest.raises(ValidationError) as exc_info:
        MCPServerConfig(name="test_server", url="http://localhost:8080", type="invalid")
    assert "Invalid MCP server type: invalid" in str(exc_info.value)

    # Test empty values (this validation is also not covered elsewhere)
    with pytest.raises(ValidationError) as exc_info:
        MCPServerConfig(name="", url="http://localhost:8080", type="http")
    assert "Name and URL cannot be empty" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        MCPServerConfig(name="test_server", url="", type="http")
    assert "Name and URL cannot be empty" in str(exc_info.value)
