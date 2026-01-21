"""Unit tests for OAuth2AuthStrategy."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pytest import MonkeyPatch

from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.auth_strategy.oauth2_auth_strategy import (
    OAuth2AuthStrategy,
)
from rasa.shared.agents.auth.types import AgentAuthType


class TestOAuth2AuthStrategy:
    """Test cases for OAuth2AuthStrategy."""

    @pytest.fixture
    def oauth2_config(self):
        """Fixture for OAuth2 configuration."""
        return {
            "oauth": {
                "token_url": "https://auth.example.com/oauth/token",
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "scope": "read write",
                "audience": "test_audience",
            }
        }

    @pytest.fixture
    def oauth2_strategy(self, oauth2_config):
        """Fixture for OAuth2 strategy with default values."""
        return OAuth2AuthStrategy.from_config(oauth2_config)

    def test_init_with_missing_oauth_config(self):
        """Test initialization with missing OAuth2 configuration."""
        with pytest.raises(ValueError, match="OAuth2 configuration is required"):
            OAuth2AuthStrategy.from_config({})

    def test_init_with_required_parameters(self):
        """Test initialization with required parameters."""
        strategy = OAuth2AuthStrategy(
            token_url="https://auth.example.com/oauth/token",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scope="read write",
            audience="test_audience",
        )

        assert strategy.token_url == "https://auth.example.com/oauth/token"
        assert strategy.client_id == "test_client_id"
        assert strategy.client_secret == "test_client_secret"
        assert strategy.scope == "read write"
        assert strategy.audience == "test_audience"
        assert strategy.timeout == 5
        assert strategy._grant_type == "client_credentials"
        assert strategy._access_token is None
        assert strategy._expires_at is None

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        strategy = OAuth2AuthStrategy(
            token_url="https://auth.example.com/oauth/token",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scope="read write",
            audience="test_audience",
            timeout=10,
        )

        assert strategy.token_url == "https://auth.example.com/oauth/token"
        assert strategy.client_id == "test_client_id"
        assert strategy.client_secret == "test_client_secret"
        assert strategy.scope == "read write"
        assert strategy.audience == "test_audience"
        assert strategy.timeout == 10
        assert strategy._grant_type == "client_credentials"
        assert strategy._access_token is None
        assert strategy._expires_at is None

    def test_auth_type_property(self, oauth2_strategy: OAuth2AuthStrategy):
        """Test auth_type property returns correct type."""
        assert oauth2_strategy.auth_type == AgentAuthType.OAUTH2

    @pytest.mark.parametrize(
        "missing_field,expected_error",
        [
            ("token_url", "Token URL is required for OAuth2 authentication"),
            ("client_id", "Client ID is required for OAuth2 authentication"),
            ("client_secret", "Client secret is required for OAuth2 authentication"),
        ],
    )
    def test_from_config_missing_required_fields(self, missing_field, expected_error):
        """Test creating strategy with missing required fields."""
        config = {
            "token_url": "https://auth.example.com/oauth/token",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "scope": "read write",
        }
        del config[missing_field]

        with pytest.raises(ValueError, match=expected_error):
            OAuth2AuthStrategy.from_config(config)

    @pytest.mark.parametrize(
        "missing_field,expected_error",
        [
            ("token_url", "Token URL is required for OAuth2 authentication"),
            ("client_id", "Client ID is required for OAuth2 authentication"),
            ("client_secret", "Client secret is required for OAuth2 authentication"),
        ],
    )
    def test_from_config_with_empty_token_url(self, missing_field, expected_error):
        """Test creating strategy with empty token URL."""
        config = {
            "token_url": "https://auth.example.com/oauth/token",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "scope": "read write",
        }
        config[missing_field] = ""

        with pytest.raises(ValueError, match=expected_error):
            OAuth2AuthStrategy.from_config(config)

    def test_from_config_with_environment_variables(self, monkeypatch: MonkeyPatch):
        """Test creating strategy from config with environment variables."""
        # Set environment variables
        monkeypatch.setenv("OAUTH_TOKEN_URL", "https://auth.example.com/oauth/token")
        monkeypatch.setenv("OAUTH_CLIENT_ID", "env_client_id")
        monkeypatch.setenv("OAUTH_CLIENT_SECRET", "env_client_secret")
        monkeypatch.setenv("OAUTH_SCOPE", "env_scope")
        monkeypatch.setenv("OAUTH_AUDIENCE", "env_audience")

        config = {
            "token_url": "${OAUTH_TOKEN_URL}",
            "client_id": "${OAUTH_CLIENT_ID}",
            "client_secret": "${OAUTH_CLIENT_SECRET}",
            "scope": "${OAUTH_SCOPE}",
            "audience": "${OAUTH_AUDIENCE}",
        }

        strategy = OAuth2AuthStrategy.from_config(config)

        # Verify that the strategy is created with environment variable placeholders
        # (resolution happens during token refresh, not during initialization)
        assert strategy.token_url == "${OAUTH_TOKEN_URL}"
        assert strategy.client_id == "${OAUTH_CLIENT_ID}"
        assert strategy.client_secret == "${OAUTH_CLIENT_SECRET}"
        assert strategy.scope == "${OAUTH_SCOPE}"
        assert strategy.audience == "${OAUTH_AUDIENCE}"

    def test_token_expiration_logic(self, oauth2_strategy: OAuth2AuthStrategy):
        """Test token expiration logic."""
        # Test with no expiration time
        oauth2_strategy._expires_at = None
        assert oauth2_strategy._is_expired() is True

        # Test with future expiration time (beyond buffer)
        oauth2_strategy._expires_at = time.time() + 3600
        assert oauth2_strategy._is_expired() is False

        # Test with past expiration time
        oauth2_strategy._expires_at = time.time() - 3600
        assert oauth2_strategy._is_expired() is True

        # Test with expiration time within buffer
        # Less than DEFAULT_BUFFER_TIME_SECONDS (10)
        oauth2_strategy._expires_at = time.time() + 5
        assert oauth2_strategy._is_expired() is True

    @pytest.mark.asyncio
    async def test_get_headers_with_valid_token(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test getting headers with valid token."""
        # Set a valid token
        oauth2_strategy._access_token = "test_access_token"
        oauth2_strategy._expires_at = time.time() + 3600

        # Mock the _refresh_access_token method
        mock_refresh_failure = AsyncMock()
        oauth2_strategy._refresh_access_token = mock_refresh_failure

        # Test the get_headers method
        headers = await oauth2_strategy.get_headers()

        # Verify the headers
        expected = {"Authorization": "Bearer test_access_token"}
        assert headers == expected
        assert mock_refresh_failure.call_count == 0

    @pytest.mark.asyncio
    async def test_get_headers_fetches_token_when_missing(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test getting headers fetches token when missing."""

        # Mock the _refresh_access_token method
        async def mock_refresh_failure():
            pass  # Don't set _access_token, simulating a failure

        oauth2_strategy._refresh_access_token = mock_refresh_failure
        oauth2_strategy._access_token = None

        with pytest.raises(ValueError, match="Failed to obtain access token"):
            await oauth2_strategy.get_headers()

    @pytest.mark.asyncio
    async def test_get_headers_successful_token_refresh(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test getting headers with successful token refresh."""
        # Set an expired token
        oauth2_strategy._access_token = "old_token"
        oauth2_strategy._expires_at = time.time() - 3600

        # Mock the _refresh_access_token method to set a new token
        async def mock_refresh_success():
            oauth2_strategy._access_token = "new_token"
            oauth2_strategy._expires_at = time.time() + 3600

        oauth2_strategy._refresh_access_token = mock_refresh_success

        headers = await oauth2_strategy.get_headers()
        expected = {"Authorization": "Bearer new_token"}
        assert headers == expected

    @pytest.mark.asyncio
    async def test_async_auth_flow_updates_headers(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test that async_auth_flow updates request headers."""
        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Set a valid token
        oauth2_strategy._access_token = "test_access_token"
        oauth2_strategy._expires_at = time.time() + 3600

        # Test the async auth_flow generator
        auth_requests = []
        async for auth_request in oauth2_strategy.async_auth_flow(request):
            auth_requests.append(auth_request)

        # Verify that headers were updated
        assert len(auth_requests) == 1
        assert auth_requests[0] is request
        expected_headers = {"Authorization": "Bearer test_access_token"}
        assert request.headers == expected_headers

    @pytest.mark.asyncio
    async def test_async_auth_flow_with_environment_variables(
        self, monkeypatch: MonkeyPatch
    ):
        """Test async_auth_flow with environment variables in OAuth2 config."""
        # Set environment variables
        monkeypatch.setenv("TEST_CLIENT_ID", "env_client_id")
        monkeypatch.setenv("TEST_CLIENT_SECRET", "env_client_secret")
        monkeypatch.setenv("TEST_SCOPE", "env_scope")
        monkeypatch.setenv("TEST_AUDIENCE", "env_audience")

        # Create strategy with environment variables in config
        strategy = OAuth2AuthStrategy(
            token_url="https://auth.example.com/oauth/token",
            client_id="${TEST_CLIENT_ID}",
            client_secret="${TEST_CLIENT_SECRET}",
            scope="${TEST_SCOPE}",
            audience="${TEST_AUDIENCE}",
        )

        # Set a valid token (simulating successful token refresh)
        strategy._access_token = "test_access_token"
        strategy._expires_at = time.time() + 3600

        # Create a mock request
        request = Mock(spec=httpx.Request)
        request.headers = {}

        # Test the async auth_flow generator
        async for auth_request in strategy.async_auth_flow(request):
            pass

        # Check that headers were updated
        expected_headers = {"Authorization": "Bearer test_access_token"}
        assert request.headers == expected_headers

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test successful token refresh."""
        # Mock httpx response
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "expires_in": 7200,
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            await oauth2_strategy._refresh_access_token()

            assert oauth2_strategy._access_token == "new_access_token"
            assert oauth2_strategy._expires_at > time.time()

    @pytest.mark.asyncio
    async def test_refresh_access_token_http_error(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test token refresh with HTTP error."""
        # Mock httpx HTTPStatusError
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "401 Unauthorized", request=Mock(), response=mock_response
                )
            )

            with pytest.raises(httpx.HTTPStatusError):
                await oauth2_strategy._refresh_access_token()

    @pytest.mark.asyncio
    async def test_refresh_access_token_missing_access_token(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test token refresh with missing access_token in response."""
        # Mock httpx response without access_token
        mock_response = Mock()
        mock_response.json.return_value = {"token_type": "Bearer", "expires_in": 7200}
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ValueError, match="No `access_token` in OAuth2 response"
            ):
                await oauth2_strategy._refresh_access_token()

    @pytest.mark.asyncio
    async def test_concurrent_get_headers_prevents_race_condition(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test that concurrent get_headers calls don't cause race conditions."""
        # Set an expired token to trigger refresh
        oauth2_strategy._access_token = "old_token"
        oauth2_strategy._expires_at = time.time() - 3600

        # Track how many times _refresh_access_token is called
        refresh_call_count = 0

        async def mock_refresh_access_token():
            nonlocal refresh_call_count
            refresh_call_count += 1
            # Simulate some async work (like HTTP request)
            await asyncio.sleep(0.01)
            oauth2_strategy._access_token = f"new_token_{refresh_call_count}"
            oauth2_strategy._expires_at = time.time() + 3600

        oauth2_strategy._refresh_access_token = mock_refresh_access_token

        # Make multiple concurrent calls to get_headers
        tasks = [oauth2_strategy.get_headers() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify that _refresh_access_token was called only once
        assert refresh_call_count == 1

        # All results should have the same token
        expected_headers = {"Authorization": "Bearer new_token_1"}
        for result in results:
            assert result == expected_headers

    @pytest.mark.asyncio
    async def test_concurrent_get_headers_with_valid_token_no_refresh(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        """Test that concurrent calls with valid token don't trigger refresh."""
        # Set a valid token
        oauth2_strategy._access_token = "valid_token"
        oauth2_strategy._expires_at = time.time() + 3600

        # Track refresh calls
        refresh_call_count = 0

        async def mock_refresh_access_token():
            nonlocal refresh_call_count
            refresh_call_count += 1

        oauth2_strategy._refresh_access_token = mock_refresh_access_token

        # Make multiple concurrent calls to get_headers
        tasks = [oauth2_strategy.get_headers() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify that _refresh_access_token was never called
        assert refresh_call_count == 0

        # All results should have the same valid token
        expected_headers = {"Authorization": "Bearer valid_token"}
        for result in results:
            assert result == expected_headers

    @pytest.mark.asyncio
    async def test_lock_initialization(self):
        """Test that the lock is properly initialized."""
        strategy = OAuth2AuthStrategy(
            token_url="https://auth.example.com/oauth/token",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scope="read write",
            audience="test_audience",
        )

        # Verify lock is initialized
        assert hasattr(strategy, "_refresh_lock")
        assert isinstance(strategy._refresh_lock, asyncio.Lock)

    def test_adheres_to_agent_auth_strategy_protocol(
        self, oauth2_strategy: OAuth2AuthStrategy
    ):
        assert isinstance(oauth2_strategy, AgentAuthStrategy)

    def test_inherits_from_httpx_auth(self, oauth2_strategy: OAuth2AuthStrategy):
        assert isinstance(oauth2_strategy, httpx.Auth)

    @pytest.mark.asyncio
    async def test_refresh_token_uses_basic_auth_header(self, monkeypatch: MonkeyPatch):
        """Test that credentials are sent in Authorization header via Basic Auth."""
        monkeypatch.setenv("TEST_CLIENT_ID", "resolved_client_id")
        monkeypatch.setenv("TEST_CLIENT_SECRET", "resolved_client_secret")

        strategy = OAuth2AuthStrategy(
            token_url="https://auth.example.com/oauth/token",
            client_id="${TEST_CLIENT_ID}",
            client_secret="${TEST_CLIENT_SECRET}",
            scope="read write",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "expires_in": 7200,
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await strategy.get_headers()

            mock_post.assert_called_once()
            call_args = mock_post.call_args

            # Verify Basic Auth is used (RFC 6749 Section 2.3.1)
            auth = call_args[1].get("auth")
            assert auth is not None
            assert isinstance(auth, httpx.BasicAuth)

            # Verify client_id and client_secret are NOT in the body
            resolved_data = call_args[1]["data"]
            assert "client_id" not in resolved_data
            assert "client_secret" not in resolved_data
            assert resolved_data["grant_type"] == "client_credentials"
            assert resolved_data["scope"] == "read write"

    @pytest.mark.asyncio
    async def test_refresh_token_basic_auth_resolves_env_vars(
        self, monkeypatch: MonkeyPatch
    ):
        """Test that Basic Auth credentials are resolved from environment variables."""
        monkeypatch.setenv("TEST_CLIENT_ID", "env_client_id")
        monkeypatch.setenv("TEST_CLIENT_SECRET", "env_client_secret")

        strategy = OAuth2AuthStrategy(
            token_url="https://auth.example.com/oauth/token",
            client_id="${TEST_CLIENT_ID}",
            client_secret="${TEST_CLIENT_SECRET}",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "expires_in": 7200,
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await strategy.get_headers()

            call_args = mock_post.call_args
            auth = call_args[1].get("auth")

            # Verify the auth object has resolved credentials
            assert auth is not None
            assert (
                auth._auth_header
                == httpx.BasicAuth("env_client_id", "env_client_secret")._auth_header
            )
