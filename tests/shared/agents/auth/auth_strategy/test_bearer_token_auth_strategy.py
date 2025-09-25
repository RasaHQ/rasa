"""Unit tests for BearerTokenAuthStrategy."""

from unittest.mock import Mock

import httpx
import pytest
from pytest import MonkeyPatch

from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.auth_strategy.bearer_token_auth_strategy import (
    BearerTokenAuthStrategy,
)
from rasa.shared.agents.auth.types import AgentAuthType


class TestBearerTokenAuthStrategy:
    """Test cases for BearerTokenAuthStrategy."""

    @pytest.fixture
    def bearer_token_strategy(self):
        """Fixture for bearer token strategy with default values."""
        token = "test_token"
        return BearerTokenAuthStrategy(token)

    def test_init_with_valid_token(
        self, bearer_token_strategy: BearerTokenAuthStrategy
    ):
        """Test initialization with valid token."""
        assert bearer_token_strategy._token == "test_token"

    def test_auth_type_property(self, bearer_token_strategy: BearerTokenAuthStrategy):
        """Test auth_type property returns correct type."""
        assert bearer_token_strategy.auth_type == AgentAuthType.BEARER_TOKEN

    @pytest.mark.asyncio
    async def test_get_headers_with_valid_token(
        self, bearer_token_strategy: BearerTokenAuthStrategy
    ):
        """Test getting headers with valid token."""
        headers = await bearer_token_strategy.get_headers()

        expected = {"Authorization": "Bearer test_token"}
        assert headers == expected

    @pytest.mark.asyncio
    async def test_get_headers_with_special_characters_in_token(self):
        """Test getting headers with special characters in token."""
        token = "test-token_with.special@chars#123"
        strategy = BearerTokenAuthStrategy(token)

        headers = await strategy.get_headers()

        expected = {"Authorization": f"Bearer {token}"}
        assert headers == expected

    def test_from_config_with_valid_token(self):
        """Test creating strategy from config with valid token."""
        config = {"token": "test_token"}

        strategy = BearerTokenAuthStrategy.from_config(config)

        assert isinstance(strategy, BearerTokenAuthStrategy)
        assert strategy._token == "test_token"

    def test_from_config_with_extra_parameters(self):
        """Test creating strategy from config with extra parameters."""
        config = {
            "token": "test_token",
            "extra_param": "extra_value",
            "another_param": 123,
        }

        strategy = BearerTokenAuthStrategy.from_config(config)

        assert isinstance(strategy, BearerTokenAuthStrategy)
        assert strategy._token == "test_token"

    @pytest.mark.parametrize(
        "invalid_config,expected_error",
        [
            ({}, "Access token is required for Bearer Token authentication"),
            (
                {"other_param": "value"},
                "Access token is required for Bearer Token authentication",
            ),
            ({"token": ""}, "Access token is required for Bearer Token authentication"),
            (
                {"token": None},
                "Access token is required for Bearer Token authentication",
            ),
        ],
    )
    def test_from_config_invalid_configs(
        self, invalid_config: dict, expected_error: str
    ):
        """Test creating strategy from invalid configs."""
        with pytest.raises(ValueError, match=expected_error):
            BearerTokenAuthStrategy.from_config(invalid_config)

    @pytest.mark.asyncio
    async def test_async_auth_flow(
        self, bearer_token_strategy: BearerTokenAuthStrategy
    ):
        """Test asynchronous auth_flow method."""
        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        auth_requests = []
        async for auth_request in bearer_token_strategy.async_auth_flow(request):
            auth_requests.append(auth_request)

        # Should yield exactly one request
        assert len(auth_requests) == 1
        assert auth_requests[0] is request

        # Check that headers were updated
        expected_headers = {"Authorization": "Bearer test_token"}
        assert request.headers == expected_headers

    async def test_auth_flow_overwrites_existing_authorization_header(self):
        """Test that auth_flow overwrites existing Authorization header."""
        token = "new_token"
        strategy = BearerTokenAuthStrategy(token)

        # Create a mock request with existing Authorization header
        request = Mock(spec=httpx.Request)
        request.headers = {"Authorization": "Bearer old_token"}

        # Test the auth_flow generator
        async for auth_request in strategy.async_auth_flow(request):
            pass

        # Check that Authorization header is overwritten
        expected_headers = {"Authorization": "Bearer new_token"}
        assert request.headers == expected_headers

    def test_adheres_to_agent_auth_strategy_protocol(
        self, bearer_token_strategy: BearerTokenAuthStrategy
    ):
        assert isinstance(bearer_token_strategy, AgentAuthStrategy)

    @pytest.mark.asyncio
    async def test_async_auth_flow_with_environment_variables(
        self, monkeypatch: MonkeyPatch
    ):
        """Test async_auth_flow with environment variables in API key."""
        # Given
        monkeypatch.setenv("TEST_BEARER_TOKEN", "test_bearer_token_auth_strategy")

        # Create strategy with environment variable in Bearer token
        strategy = BearerTokenAuthStrategy("${TEST_BEARER_TOKEN}")

        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        async for auth_request in strategy.async_auth_flow(request):
            pass

        # Check that environment variable was resolved
        expected_headers = {"Authorization": "Bearer test_bearer_token_auth_strategy"}
        assert request.headers == expected_headers
