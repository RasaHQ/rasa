"""Unit tests for AgentAuthStrategy base class."""

from unittest.mock import Mock

import httpx
import pytest
from pytest import MonkeyPatch

from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.types import AgentAuthType


class ConcreteAuthStrategy(AgentAuthStrategy):
    """Concrete implementation of AgentAuthStrategy for testing."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @classmethod
    def from_config(cls, config):
        return cls(config.get("api_key", "default_key"))

    @property
    def auth_type(self) -> AgentAuthType:
        return AgentAuthType.API_KEY

    async def get_headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}


class TestAgentAuthStrategy:
    """Test cases for AgentAuthStrategy base class."""

    @pytest.fixture
    def concrete_strategy(self):
        """Fixture for concrete auth strategy."""
        return ConcreteAuthStrategy("test_key")

    def test_inherits_from_httpx_auth(self, concrete_strategy):
        """Test that AgentAuthStrategy inherits from httpx.Auth."""
        assert isinstance(concrete_strategy, httpx.Auth)

    @pytest.mark.asyncio
    async def test_async_auth_flow_basic_functionality(self, concrete_strategy):
        """Test basic async_auth_flow functionality from base class."""
        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        auth_requests = []
        async for auth_request in concrete_strategy.async_auth_flow(request):
            auth_requests.append(auth_request)

        # Should yield exactly one request
        assert len(auth_requests) == 1
        assert auth_requests[0] is request

        # Check that headers were updated
        expected_headers = {"Authorization": "Bearer test_key"}
        assert request.headers == expected_headers

    @pytest.mark.asyncio
    async def test_async_auth_flow_with_environment_variables(
        self, monkeypatch: MonkeyPatch
    ):
        """Test async_auth_flow with environment variables in API key."""
        # Given
        monkeypatch.setenv("TEST_API_KEY", "test_agent_auth_strategy")

        # Create strategy with environment variable in API key
        strategy = ConcreteAuthStrategy("${TEST_API_KEY}")

        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        async for auth_request in strategy.async_auth_flow(request):
            pass

        # Check that environment variable was resolved
        expected_headers = {"Authorization": "Bearer test_agent_auth_strategy"}
        assert request.headers == expected_headers

    @pytest.mark.asyncio
    async def test_async_auth_flow_with_multiple_environment_variables(
        self, monkeypatch: MonkeyPatch
    ):
        """Test async_auth_flow with multiple environment variables."""
        # Given
        monkeypatch.setenv("API_KEY", "secret_key")
        monkeypatch.setenv("API_VERSION", "v2")

        # Create strategy that returns headers with multiple environment variables
        class MultiEnvVarStrategy(ConcreteAuthStrategy):
            async def get_headers(self):
                return {
                    "Authorization": "Bearer ${API_KEY}",
                    "X-API-Version": "${API_VERSION}",
                    "X-Client": "test-client",
                }

        strategy = MultiEnvVarStrategy("test_key")

        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        async for auth_request in strategy.async_auth_flow(request):
            pass

        # Check that environment variables were resolved
        expected_headers = {
            "Authorization": "Bearer secret_key",
            "X-API-Version": "v2",
            "X-Client": "test-client",
        }
        assert request.headers == expected_headers

    @pytest.mark.asyncio
    async def test_async_auth_flow_with_undefined_environment_variables(self):
        """Test async_auth_flow with undefined environment variables."""

        # Create strategy that returns headers with undefined environment variables
        class UndefinedEnvVarStrategy(ConcreteAuthStrategy):
            async def get_headers(self):
                return {"Authorization": "Bearer ${UNDEFINED_VAR}"}

        strategy = UndefinedEnvVarStrategy("test_key")

        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        async for auth_request in strategy.async_auth_flow(request):
            pass

        # Check that undefined environment variable is left as-is
        expected_headers = {"Authorization": "Bearer ${UNDEFINED_VAR}"}
        assert request.headers == expected_headers
