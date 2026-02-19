from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from rasa.core.config.available_endpoints import (
    InteractionHandlingConfig,
    MCPFromSlotsEntry,
    MCPMetaMapConfig,
    MCPServerConfig,
)
from rasa.core.config.configuration import Configuration, EndpointsConfigPath
from rasa.exceptions import ValidationError
from rasa.shared.core.constants import GLOBAL_SILENCE_TIMEOUT_KEY
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.mcp.utils import build_mcp_meta
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
        "timer_store": EndpointConfig(url="some/timer/store/url"),
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
    assert (
        available_endpoints.timer_store == deserialized_endpoint_config["timer_store"]
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


@pytest.mark.parametrize(
    "server_kwargs,expected_substrings",
    [
        (
            {
                "name": "server_3",
                "url": "https://example.com/mcp",
                "type": "https",
                "api_key": "some_key",
            },
            ["'api_key'", "server_3", "environment variable"],
        ),
        (
            {
                "name": "server_4",
                "url": "https://example.com/mcp",
                "type": "https",
                "token": "some_token",
            },
            ["'token'", "server_4", "environment variable"],
        ),
        (
            {
                "name": "server_5",
                "url": "https://example.com/mcp",
                "type": "https",
                "oauth": {
                    "token_url": "https://example.com/oauth/token",
                    "client_id": "test_client_id",
                    "client_secret": "test_client_secret",
                    "scope": "test_scope",
                    "audience": "test_audience",
                },
            },
            ["'client_secret'", "server_5", "environment variable"],
        ),
    ],
    ids=["invalid_api_key", "invalid_token", "invalid_oauth_client_secret"],
)
def test_validate_mcp_server_invalid_secrets(
    server_kwargs: Dict[str, Any], expected_substrings: list
) -> None:
    """Test validation fails for MCP server config with invalid secret."""
    with pytest.raises(ValidationError) as exc_info:
        MCPServerConfig(**server_kwargs)
    error_message = str(exc_info.value)
    for substring in expected_substrings:
        assert substring in error_message


@pytest.mark.parametrize(
    "server_kwargs,expected_name,expected_param_path,expected_value",
    [
        (
            {
                "name": "server_6",
                "url": "https://example.com/mcp",
                "type": "https",
                "api_key": "${SECRET_API_KEY}",
            },
            "server_6",
            ["api_key"],
            "${SECRET_API_KEY}",
        ),
        (
            {
                "name": "server_7",
                "url": "https://example.com/mcp",
                "type": "https",
                "token": "${SECRET_TOKEN}",
            },
            "server_7",
            ["token"],
            "${SECRET_TOKEN}",
        ),
        (
            {
                "name": "server_8",
                "url": "https://example.com/mcp",
                "type": "https",
                "oauth": {
                    "token_url": "https://example.com/oauth/token",
                    "client_id": "test_client_id",
                    "client_secret": "${SECRET_CLIENT_SECRET}",
                    "scope": "test_scope",
                    "audience": "test_audience",
                },
            },
            "server_8",
            ["oauth", "client_secret"],
            "${SECRET_CLIENT_SECRET}",
        ),
    ],
    ids=["valid_api_key", "valid_token", "valid_oauth_client_secret"],
)
def test_validate_mcp_server_valid_secrets(
    server_kwargs: Dict[str, Any],
    expected_name: str,
    expected_param_path: list,
    expected_value: str,
) -> None:
    """Test validation succeeds for MCP server config with env-var style secret."""
    server_config = MCPServerConfig(**server_kwargs)
    assert server_config.name == expected_name
    params = server_config.additional_params or {}
    for key in expected_param_path:
        params = params[key]
    assert params == expected_value


def test_mcp_meta_map_config_from_slots():
    """Test MCPMetaMapConfig with from_slots (list of slot/param)."""
    meta_map = MCPMetaMapConfig(
        from_slots=[
            MCPFromSlotsEntry(slot="user_id", param="user_id"),
            MCPFromSlotsEntry(slot="role", param="role"),
        ],
    )
    assert len(meta_map.from_slots) == 2
    assert (
        meta_map.from_slots[0].slot == "user_id"
        and meta_map.from_slots[0].param == "user_id"
    )
    assert (
        meta_map.from_slots[1].slot == "role" and meta_map.from_slots[1].param == "role"
    )


def test_mcp_server_config_with_meta_map():
    """Test MCPServerConfig with meta_map is parsed and not in additional_params."""
    server_config = MCPServerConfig(
        name="internal_api",
        url="http://internal:8000/mcp/",
        type="http",
        meta_map=MCPMetaMapConfig(
            from_slots=[
                MCPFromSlotsEntry(slot="user_id", param="user_id"),
                MCPFromSlotsEntry(slot="role", param="user_role"),
            ],
            static={"api_version": "v2", "source": "rasa_agent"},
        ),
    )
    assert server_config.meta_map is not None
    assert len(server_config.meta_map.from_slots) == 2
    assert server_config.meta_map.from_slots[0].slot == "user_id"
    assert server_config.meta_map.from_slots[0].param == "user_id"
    assert server_config.meta_map.from_slots[1].slot == "role"
    assert server_config.meta_map.from_slots[1].param == "user_role"
    assert server_config.meta_map.static == {
        "api_version": "v2",
        "source": "rasa_agent",
    }
    assert "meta_map" not in (server_config.additional_params or {})
    dumped = server_config.model_dump()
    assert "meta_map" in dumped
    assert dumped["meta_map"]["from_slots"] == [
        {"slot": "user_id", "param": "user_id"},
        {"slot": "role", "param": "user_role"},
    ]
    assert dumped["meta_map"]["static"] == {"api_version": "v2", "source": "rasa_agent"}


@pytest.mark.parametrize(
    "config,slots,expected_meta",
    [
        (
            MCPMetaMapConfig(
                from_slots=[
                    MCPFromSlotsEntry(slot="user_id", param="user_id"),
                    MCPFromSlotsEntry(slot="role", param="user_role"),
                ],
            ),
            {"user_id": "u-123", "role": "admin"},
            {"user_id": "u-123", "user_role": "admin"},
        ),
        (
            MCPMetaMapConfig(
                from_slots=[
                    MCPFromSlotsEntry(slot="user_id", param="user_id"),
                    MCPFromSlotsEntry(slot="role", param="user_role"),
                ],
            ),
            {"user_id": "u-123"},
            {"user_id": "u-123", "user_role": None},
        ),
        (None, {"user_id": "x"}, {}),
        (MCPMetaMapConfig(from_slots=None), {"user_id": "x"}, {}),
        (MCPMetaMapConfig(from_slots=[]), {"user_id": "x"}, {}),
        (
            MCPMetaMapConfig(
                from_slots=None,
                static={"api_version": "v2", "source": "rasa_agent"},
            ),
            {},
            {"api_version": "v2", "source": "rasa_agent"},
        ),
        (
            MCPMetaMapConfig(
                from_slots=[
                    MCPFromSlotsEntry(slot="user_id", param="user_id"),
                ],
                static={"api_version": "v2", "source": "rasa_agent"},
            ),
            {"user_id": "u-456"},
            {
                "api_version": "v2",
                "source": "rasa_agent",
                "user_id": "u-456",
            },
        ),
    ],
    ids=[
        "from_slots_only",
        "omits_none_or_missing_slots",
        "config_none",
        "from_slots_none",
        "from_slots_empty",
        "static_only",
        "static_and_from_slots",
    ],
)
def test_build_mcp_meta(
    config: Any, slots: Dict[str, Any], expected_meta: Dict[str, Any]
) -> None:
    """Test build_mcp_meta returns expected _meta dict for config and slots."""
    assert build_mcp_meta(config, slots) == expected_meta


@pytest.mark.parametrize(
    "config_factory,match_re",
    [
        (
            lambda: MCPMetaMapConfig(
                from_slots=[MCPFromSlotsEntry(slot="", param="user_id")],
            ),
            "slot must be a non-empty string",
        ),
        (
            lambda: MCPMetaMapConfig(
                from_slots=[MCPFromSlotsEntry(slot="user_id", param="")],
            ),
            "param must be a non-empty string",
        ),
        (lambda: MCPMetaMapConfig(static={"": "v"}), "meta_map.static.*keys"),
        (lambda: MCPMetaMapConfig(static={"k": ""}), "meta_map.static.*values"),
    ],
    ids=["empty_slot", "empty_param", "empty_static_key", "empty_static_value"],
)
def test_mcp_meta_map_config_rejects_empty_strings(
    config_factory: Any, match_re: str
) -> None:
    """Test MCPMetaMapConfig validates non-empty slot, param, and static."""
    with pytest.raises(ValueError, match=match_re):
        config_factory()
