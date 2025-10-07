from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from rasa.core.config.available_endpoints import (
    InteractionHandlingConfig,
    MCPServerConfig,
)
from rasa.core.config.configuration import Configuration, EndpointsConfigPath
from rasa.exceptions import ValidationError
from rasa.shared.core.constants import GLOBAL_SILENCE_TIMEOUT_KEY
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def deserialized_endpoint_config() -> Dict[str, Any]:
    return {
        "nlg": EndpointConfig(url="some/nlg/url"),
        "nlu": EndpointConfig(url="some/nlu/url"),
        "action_endpoint": EndpointConfig(url="some/action/endpoint"),
        "models": EndpointConfig(url="some/models/url"),
        "tracker_store": EndpointConfig(url="some/tracker/url"),
        "lock_store": EndpointConfig(url="some/lock/url"),
        "event_broker": EndpointConfig(url="some/event/broker/url"),
        "vector_store": EndpointConfig(url="some/vector/store/url"),
        "mcp_servers": [
            {"name": "server_1", "url": "some/mcp/server_1/url", "type": "http"},
            {"name": "server_2", "url": "some/mcp/server_2/url", "type": "http"},
            {
                "name": "server_3",
                "url": "some/mcp/server_3/url",
                "type": "https",
                "api_key": "${SECRET_API_KEY}",
            },
        ],
        "model_groups": [
            {
                "id": "default",
                "models": [{"provider": "openai"}],
                "router": {"routing_strategy": "simple-shuffle"},
            }
        ],
        "privacy": {
            "enabled": True,
            "anonymization": {
                "enabled": True,
                "anonymization_fields": ["user_id", "session_id"],
            },
        },
        "interaction_handling": {
            GLOBAL_SILENCE_TIMEOUT_KEY: 5.0,  # Default silence timeout
        },
    }


@pytest.fixture
def mock_read_endpoint_config(
    deserialized_endpoint_config,
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """Mock the `read_endpoint_config` function to avoid reading from files."""

    def _read_endpoint_config(endpoint_file: str, endpoint_type: str) -> Dict[str, Any]:
        """Mocked function to return a predefined endpoint configuration."""
        return deserialized_endpoint_config[endpoint_type]

    _mock_read_endpoint_config = MagicMock(side_effect=_read_endpoint_config)
    monkeypatch.setattr(
        "rasa.core.config.available_endpoints.read_endpoint_config",
        _mock_read_endpoint_config,
    )

    return _mock_read_endpoint_config


@pytest.fixture
def mock_read_property_config_from_endpoints_file(
    deserialized_endpoint_config,
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """Mock the `read_property_config_from_endpoints_file` function to avoid reading from files."""  # noqa: E501

    def _read_property_config_from_endpoints_file(
        endpoint_file: str, property_name: str
    ) -> Dict[str, Any]:
        """Mocked function to return a predefined endpoint configuration."""
        return deserialized_endpoint_config[property_name]

    _mock_read_property_config_from_endpoints_file = MagicMock(
        side_effect=_read_property_config_from_endpoints_file
    )
    monkeypatch.setattr(
        "rasa.core.config.available_endpoints.read_property_config_from_endpoints_file",
        _mock_read_property_config_from_endpoints_file,
    )

    return _mock_read_property_config_from_endpoints_file


@pytest.mark.usefixtures(
    "mock_read_endpoint_config", "mock_read_property_config_from_endpoints_file"
)
def test_available_endpoints_read_endpoints(
    deserialized_endpoint_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the `AvailableEndpoints` class reads the endpoints correctly."""
    endpoint_file = "some/path/to/endpoints.yml"

    mock_endpoint_config_path = MagicMock(spec=EndpointsConfigPath)
    mock_endpoint_config_path.validate.return_value = Path(endpoint_file)
    monkeypatch.setattr(
        "rasa.core.config.configuration.EndpointsConfigPath", mock_endpoint_config_path
    )

    # Create an instance of AvailableEndpoints
    available_endpoints = Configuration.initialise_endpoints(
        endpoints_path=Path(endpoint_file)
    ).endpoints

    # Assert that the attributes are set correctly
    assert available_endpoints.nlg == deserialized_endpoint_config["nlg"]
    assert available_endpoints.nlu == deserialized_endpoint_config["nlu"]
    assert available_endpoints.action == deserialized_endpoint_config["action_endpoint"]
    assert available_endpoints.model == deserialized_endpoint_config["models"]
    assert (
        available_endpoints.tracker_store
        == deserialized_endpoint_config["tracker_store"]
    )
    assert available_endpoints.lock_store == deserialized_endpoint_config["lock_store"]
    assert (
        available_endpoints.event_broker == deserialized_endpoint_config["event_broker"]
    )
    assert (
        available_endpoints.vector_store == deserialized_endpoint_config["vector_store"]
    )
    expected_mcp_servers = [
        MCPServerConfig(name="server_1", url="some/mcp/server_1/url", type="http"),
        MCPServerConfig(name="server_2", url="some/mcp/server_2/url", type="http"),
        MCPServerConfig(
            name="server_3",
            url="some/mcp/server_3/url",
            type="https",
            api_key="${SECRET_API_KEY}",
        ),
    ]
    assert available_endpoints.mcp_servers == expected_mcp_servers
    assert available_endpoints.mcp_servers[2].additional_params == {
        "api_key": "${SECRET_API_KEY}"
    }
    assert expected_mcp_servers[2].additional_params == {"api_key": "${SECRET_API_KEY}"}
    assert (
        available_endpoints.model_groups == deserialized_endpoint_config["model_groups"]
    )
    assert available_endpoints.privacy == deserialized_endpoint_config["privacy"]
    assert (
        available_endpoints.interaction_handling
        == InteractionHandlingConfig.from_dict(
            deserialized_endpoint_config["interaction_handling"]
        )
    )


def test_interaction_handling_config_from_dict():
    """Test that the InteractionHandlingConfig can be created from a dictionary."""
    interaction_handling_config_dict = {
        GLOBAL_SILENCE_TIMEOUT_KEY: 5.0,  # Default silence timeout
    }
    interaction_handling_config = InteractionHandlingConfig.from_dict(
        interaction_handling_config_dict
    )

    assert interaction_handling_config.global_silence_timeout == 5.0
    assert isinstance(interaction_handling_config, InteractionHandlingConfig)


@pytest.mark.parametrize("bad_value", [None, "not_a_number", -1, 0])
def test_interaction_handling_config_from_dict_wrong_data_type(bad_value: Any):
    """Test that the InteractionHandlingConfig raises exception for invalid data."""
    interaction_handling = {
        GLOBAL_SILENCE_TIMEOUT_KEY: bad_value,  # Default silence timeout
    }
    with pytest.raises(RasaException):
        InteractionHandlingConfig.from_dict(interaction_handling)


def test_validate_mcp_server_with_invalid_api_key():
    """Test validation fails for MCP server configuration with invalid api_key."""
    with pytest.raises(ValidationError) as exc_info:
        MCPServerConfig(
            name="server_3",
            url="https://example.com/mcp",
            type="https",
            api_key="some_key",
        )

    # Verify the specific error message
    error_message = str(exc_info.value)
    assert (
        "You defined the 'api_key' in MCP server - 'server_3' as a string"
        in error_message
    )
    assert "The 'api_key' must be set as an environment variable" in error_message


def test_validate_mcp_server_with_invalid_token():
    """Test validation fails for MCP server configuration with invalid token."""
    with pytest.raises(ValidationError) as exc_info:
        MCPServerConfig(
            name="server_4",
            url="https://example.com/mcp",
            type="https",
            token="some_token",
        )

    # Verify the specific error message
    error_message = str(exc_info.value)
    assert (
        "You defined the 'token' in MCP server - 'server_4' as a string"
        in error_message
    )
    assert "The 'token' must be set as an environment variable" in error_message


def test_validate_mcp_server_with_invalid_oauth_client_secret():
    """Test validation fails for MCP server configuration with invalid oauth."""
    with pytest.raises(ValidationError) as exc_info:
        MCPServerConfig(
            name="server_5",
            url="https://example.com/mcp",
            type="https",
            oauth={
                "token_url": "https://example.com/oauth/token",
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "scope": "test_scope",
                "audience": "test_audience",
            },
        )

    # Verify the specific error message
    error_message = str(exc_info.value)
    assert (
        "You defined the 'client_secret' in MCP server - 'server_5' as a string"
        in error_message
    )
    assert "The 'client_secret' must be set as an environment variable" in error_message


def test_validate_mcp_server_with_valid_api_key():
    """Test validation succeeds for MCP server configuration with valid api_key."""
    server_config = MCPServerConfig(
        name="server_6",
        url="https://example.com/mcp",
        type="https",
        api_key="${SECRET_API_KEY}",
    )

    # Verify the configuration was created successfully
    assert server_config.name == "server_6"
    assert server_config.additional_params["api_key"] == "${SECRET_API_KEY}"


def test_validate_mcp_server_with_valid_token():
    """Test validation succeeds for MCP server configuration with valid token."""
    server_config = MCPServerConfig(
        name="server_7",
        url="https://example.com/mcp",
        type="https",
        token="${SECRET_TOKEN}",
    )

    # Verify the configuration was created successfully
    assert server_config.name == "server_7"
    assert server_config.additional_params["token"] == "${SECRET_TOKEN}"


def test_validate_mcp_server_with_valid_oauth_client_secret():
    """Test validation succeeds for MCP server configuration with valid oauth."""
    # This should not raise any validation errors when using environment variable format
    server_config = MCPServerConfig(
        name="server_8",
        url="https://example.com/mcp",
        type="https",
        oauth={
            "token_url": "https://example.com/oauth/token",
            "client_id": "test_client_id",
            "client_secret": "${SECRET_CLIENT_SECRET}",
            "scope": "test_scope",
            "audience": "test_audience",
        },
    )

    # Verify the configuration was created successfully
    assert server_config.name == "server_8"
    assert (
        server_config.additional_params["oauth"]["client_secret"]
        == "${SECRET_CLIENT_SECRET}"
    )
