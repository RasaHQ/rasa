from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from rasa.core.available_endpoints import AvailableEndpoints, InteractionHandlingConfig
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
        "model_groups": {
            "id": "default",
            "models": [{"provider": "openai"}],
            "router": {"routing_strategy": "round_robin"},
        },
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
        "rasa.core.available_endpoints.read_endpoint_config", _mock_read_endpoint_config
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
        "rasa.core.available_endpoints.read_property_config_from_endpoints_file",
        _mock_read_property_config_from_endpoints_file,
    )

    return _mock_read_property_config_from_endpoints_file


@pytest.mark.usefixtures(
    "mock_read_endpoint_config", "mock_read_property_config_from_endpoints_file"
)
def test_available_endpoints_read_endpoints(
    deserialized_endpoint_config,
) -> None:
    """Test that the `AvailableEndpoints` class reads the endpoints correctly."""

    endpoint_file = "some/path/to/endpoints.yml"
    AvailableEndpoints.reset_instance()  # Reset the instance before the test
    # Create an instance of AvailableEndpoints
    available_endpoints = AvailableEndpoints.read_endpoints(endpoint_file)

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
        available_endpoints.model_groups == deserialized_endpoint_config["model_groups"]
    )
    assert available_endpoints.privacy == deserialized_endpoint_config["privacy"]
    assert (
        available_endpoints.interaction_handling
        == InteractionHandlingConfig.from_dict(
            deserialized_endpoint_config["interaction_handling"]
        )
    )

    # Reset the instance after the test,
    # because AvailableEndpoints is using a singleton pattern
    AvailableEndpoints.reset_instance()


def test_available_endpoints_get_instance(
    mock_read_endpoint_config: MagicMock,
    mock_read_property_config_from_endpoints_file: MagicMock,
) -> None:
    """Test that `AvailableEndpoints` instance is not re-created on multiple calls."""
    endpoint_file = "some/path/to/endpoints.yml"
    AvailableEndpoints.reset_instance()  # Reset the instance before the test

    AvailableEndpoints.get_instance(endpoint_file)
    AvailableEndpoints.get_instance(endpoint_file)

    assert mock_read_endpoint_config.call_count == 8
    assert mock_read_property_config_from_endpoints_file.call_count == 3

    # Reset the instance after the test,
    # because AvailableEndpoints is using a singleton pattern
    AvailableEndpoints.reset_instance()


def test_available_endpoints_reset_instance(
    mock_read_endpoint_config: MagicMock,
    mock_read_property_config_from_endpoints_file: MagicMock,
) -> None:
    """Test that `AvailableEndpoints` instance can be reset and re-initialized."""
    endpoint_file = "some/path/to/endpoints.yml"

    AvailableEndpoints.reset_instance()  # Reset the instance before the test

    # Get the instance for the first time
    AvailableEndpoints.get_instance(endpoint_file)

    # Reset the instance
    AvailableEndpoints.reset_instance()

    assert AvailableEndpoints._instance is None

    # Get the instance again
    AvailableEndpoints.get_instance(endpoint_file)

    assert mock_read_endpoint_config.call_count == 16  # Called twice for each endpoint
    assert (
        mock_read_property_config_from_endpoints_file.call_count == 6
    )  # Called twice for each property

    # Reset the instance after the test,
    # because AvailableEndpoints is using a singleton pattern
    AvailableEndpoints.reset_instance()


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
