"""Unit tests for APIKeyAuthStrategy."""

from unittest.mock import Mock

import httpx
import pytest
from pytest import MonkeyPatch

from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.auth_strategy.api_key_auth_strategy import (
    APIKeyAuthStrategy,
)
from rasa.shared.agents.auth.types import AgentAuthType


class TestAPIKeyAuthStrategy:
    """Test cases for APIKeyAuthStrategy."""

    @pytest.fixture
    def api_key_strategy(self):
        """Fixture for API key strategy with default values."""
        api_key = "test_api_key"
        return APIKeyAuthStrategy(api_key)

    def test_init_with_default_values(self, api_key_strategy: APIKeyAuthStrategy):
        """Test initialization with default values."""
        assert api_key_strategy.api_key == "test_api_key"
        assert api_key_strategy.header_name == "Authorization"
        assert api_key_strategy.header_format == "Bearer {key}"

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        api_key = "test_api_key"
        header_name = "X-API-Key"
        header_format = "API-Key {key}"

        strategy = APIKeyAuthStrategy(api_key, header_name, header_format)

        assert strategy.api_key == api_key
        assert strategy.header_name == header_name
        assert strategy.header_format == header_format

    def test_init_with_empty_api_key(self):
        """Test initialization with empty API key."""
        api_key = ""
        strategy = APIKeyAuthStrategy(api_key)

        assert strategy.api_key == ""
        assert strategy.header_name == "Authorization"
        assert strategy.header_format == "Bearer {key}"

    def test_auth_type_property(self, api_key_strategy: APIKeyAuthStrategy):
        """Test auth_type property returns correct type."""
        assert api_key_strategy.auth_type == AgentAuthType.API_KEY

    @pytest.mark.asyncio
    async def test_get_headers_default_format(
        self, api_key_strategy: APIKeyAuthStrategy
    ):
        """Test getting headers with default format."""
        headers = await api_key_strategy.get_headers()

        expected = {"Authorization": "Bearer test_api_key"}
        assert headers == expected

    @pytest.mark.asyncio
    async def test_get_headers_custom_format(self):
        """Test getting headers with custom format."""
        api_key = "test_api_key"
        header_name = "X-API-Key"
        header_format = "API-Key {key}"
        strategy = APIKeyAuthStrategy(api_key, header_name, header_format)

        headers = await strategy.get_headers()

        expected = {"X-API-Key": "API-Key test_api_key"}
        assert headers == expected

    @pytest.mark.asyncio
    async def test_get_headers_custom_format_without_key_placeholder(self):
        """Test getting headers with custom format without key placeholder."""
        api_key = "test_api_key"
        header_name = "X-API-Key"
        header_format = "static_value"
        strategy = APIKeyAuthStrategy(api_key, header_name, header_format)

        headers = await strategy.get_headers()

        expected = {"X-API-Key": "static_value"}
        assert headers == expected

    def test_from_config_with_minimal_config(self):
        """Test creating strategy from minimal config."""
        config = {"api_key": "test_key"}

        strategy = APIKeyAuthStrategy.from_config(config)

        assert isinstance(strategy, APIKeyAuthStrategy)
        assert strategy.api_key == "test_key"
        assert strategy.header_name == "Authorization"
        assert strategy.header_format == "Bearer {key}"

    def test_from_config_with_full_config(self):
        """Test creating strategy from full config."""
        config = {
            "api_key": "test_key",
            "header_name": "X-API-Key",
            "header_format": "API-Key {key}",
        }

        strategy = APIKeyAuthStrategy.from_config(config)

        assert isinstance(strategy, APIKeyAuthStrategy)
        assert strategy.api_key == "test_key"
        assert strategy.header_name == "X-API-Key"
        assert strategy.header_format == "API-Key {key}"

    def test_from_config_with_extra_parameters(self):
        """Test creating strategy from config with extra parameters."""
        config = {
            "api_key": "test_key",
            "header_name": "X-API-Key",
            "header_format": "API-Key {key}",
            "extra_param": "extra_value",
            "another_param": 123,
        }

        strategy = APIKeyAuthStrategy.from_config(config)

        assert isinstance(strategy, APIKeyAuthStrategy)
        assert strategy.api_key == "test_key"
        assert strategy.header_name == "X-API-Key"
        assert strategy.header_format == "API-Key {key}"

    @pytest.mark.parametrize(
        "invalid_config,expected_error",
        [
            ({}, "API key is required for API KEY authentication"),
            (
                {"header_name": "X-API-Key"},
                "API key is required for API KEY authentication",
            ),
            ({"api_key": ""}, "API key is required for API KEY authentication"),
            ({"api_key": None}, "API key is required for API KEY authentication"),
        ],
    )
    def test_from_config_invalid_configs(
        self, invalid_config: dict, expected_error: str
    ):
        """Test creating strategy from invalid configs."""
        with pytest.raises(ValueError, match=expected_error):
            APIKeyAuthStrategy.from_config(invalid_config)

    @pytest.mark.asyncio
    async def test_get_headers_with_complex_header_format(self):
        """Test getting headers with complex header format."""
        api_key = "complex_key_123"
        header_name = "X-Authentication"
        header_format = "Rasa-Auth key={key} version=1.0"
        strategy = APIKeyAuthStrategy(api_key, header_name, header_format)

        headers = await strategy.get_headers()

        expected = {"X-Authentication": "Rasa-Auth key=complex_key_123 version=1.0"}
        assert headers == expected

    @pytest.mark.asyncio
    async def test_async_auth_flow(self, api_key_strategy: APIKeyAuthStrategy):
        """Test asynchronous auth_flow method."""
        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        auth_requests = []
        async for auth_request in api_key_strategy.async_auth_flow(request):
            auth_requests.append(auth_request)

        # Should yield exactly one request
        assert len(auth_requests) == 1
        assert auth_requests[0] is request

        # Check that headers were updated
        expected_headers = {"Authorization": "Bearer test_api_key"}
        assert request.headers == expected_headers

    async def test_auth_flow_overwrites_existing_authorization_header(self):
        """Test that auth_flow overwrites existing Authorization header."""
        key = "new_api_key"
        strategy = APIKeyAuthStrategy(key)

        # Create a mock request with existing Authorization header
        request = Mock(spec=httpx.Request)
        request.headers = {"Authorization": "Bearer old_api_key"}

        # Test the auth_flow generator
        async for auth_request in strategy.async_auth_flow(request):
            pass

        # Check that Authorization header is overwritten
        expected_headers = {"Authorization": "Bearer new_api_key"}
        assert request.headers == expected_headers

    def test_adheres_to_agent_auth_strategy_protocol(
        self, api_key_strategy: APIKeyAuthStrategy
    ):
        assert isinstance(api_key_strategy, AgentAuthStrategy)

    @pytest.mark.asyncio
    async def test_async_auth_flow_with_environment_variables(
        self, monkeypatch: MonkeyPatch
    ):
        """Test async_auth_flow with environment variables in API key."""
        # Given
        monkeypatch.setenv("TEST_API_KEY", "test_api_key_auth_strategy")

        # Create strategy with environment variable in API key
        strategy = APIKeyAuthStrategy("${TEST_API_KEY}")

        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        async for auth_request in strategy.async_auth_flow(request):
            pass

        # Check that environment variable was resolved
        expected_headers = {"Authorization": "Bearer test_api_key_auth_strategy"}
        assert request.headers == expected_headers
