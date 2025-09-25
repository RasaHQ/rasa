"""Unit tests for AgentAuthFactory."""

from unittest.mock import MagicMock, patch

import pytest

from rasa.shared.agents.auth.agent_auth_factory import AgentAuthFactory
from rasa.shared.agents.auth.auth_strategy import (
    AgentAuthStrategy,
    APIKeyAuthStrategy,
    BearerTokenAuthStrategy,
    OAuth2AuthStrategy,
)
from rasa.shared.agents.auth.constants import CONFIG_MODULE_KEY
from rasa.shared.agents.auth.types import AgentAuthType


class TestAgentAuthFactory:
    """Test cases for AgentAuthFactory."""

    @pytest.fixture
    def mock_auth_config(self) -> dict:
        """Fixture for creating a mock auth configuration."""
        return {
            "api_key": "test_api_key",
            "header_name": "X-API-Key",
            "header_format": "{api_key}",
        }

    @pytest.fixture
    def mock_oauth_config(self) -> dict:
        """Fixture for creating a mock OAuth2 configuration."""
        return {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "token_url": "https://example.com/oauth/token",
            "scope": "read write",
            "grant_type": "client_credentials",
        }

    @pytest.fixture
    def mock_bearer_config(self) -> dict:
        """Fixture for creating a mock Bearer token configuration."""
        return {
            "token": "test_bearer_token",
        }

    @pytest.fixture
    def mock_custom_auth_config(self) -> dict:
        """Fixture for creating a mock custom auth configuration."""
        return {
            CONFIG_MODULE_KEY: "test.custom_auth.CustomAuthStrategy",
            "custom_param": "custom_value",
        }

    def test_auth_strategies_mapping(self) -> None:
        """Test that the auth strategies mapping is correct."""
        expected_mapping = {
            AgentAuthType.API_KEY: APIKeyAuthStrategy,
            AgentAuthType.OAUTH2: OAuth2AuthStrategy,
            AgentAuthType.BEARER_TOKEN: BearerTokenAuthStrategy,
        }

        assert AgentAuthFactory._auth_strategies == expected_mapping

    def test_get_supported_auth_strategy_types(self) -> None:
        """Test getting supported auth strategy types."""
        result = AgentAuthFactory.get_supported_auth_strategy_types()

        expected = [
            AgentAuthType.API_KEY,
            AgentAuthType.OAUTH2,
            AgentAuthType.BEARER_TOKEN,
        ]
        assert set(result) == set(expected)
        assert len(result) == 3

    def test_is_auth_strategy_supported_true(self) -> None:
        """Test checking if supported auth strategy is supported."""
        assert (
            AgentAuthFactory.is_auth_strategy_supported(AgentAuthType.API_KEY) is True
        )
        assert AgentAuthFactory.is_auth_strategy_supported(AgentAuthType.OAUTH2) is True
        assert (
            AgentAuthFactory.is_auth_strategy_supported(AgentAuthType.BEARER_TOKEN)
            is True
        )

    def test_is_auth_strategy_supported_false(self) -> None:
        """Test checking if unsupported auth strategy is not supported."""
        # Create a mock auth type that's not in the mapping
        mock_auth_type = MagicMock()
        mock_auth_type.value = "unsupported_auth_type"
        assert AgentAuthFactory.is_auth_strategy_supported(mock_auth_type) is False

    def test_get_auth_strategy_class_supported(self) -> None:
        """Test getting auth strategy class for supported auth types."""
        api_key_class = AgentAuthFactory._get_auth_strategy_class(AgentAuthType.API_KEY)
        oauth2_class = AgentAuthFactory._get_auth_strategy_class(AgentAuthType.OAUTH2)
        bearer_token_class = AgentAuthFactory._get_auth_strategy_class(
            AgentAuthType.BEARER_TOKEN
        )

        assert api_key_class is APIKeyAuthStrategy
        assert oauth2_class is OAuth2AuthStrategy
        assert bearer_token_class is BearerTokenAuthStrategy

    def test_get_auth_strategy_class_unsupported(self) -> None:
        """Test getting auth strategy class for unsupported auth types."""
        mock_auth_type = MagicMock()
        mock_auth_type.value = "unsupported_auth_type"

        with pytest.raises(
            ValueError, match="Unsupported authentication strategy type"
        ):
            AgentAuthFactory._get_auth_strategy_class(mock_auth_type)

    def test_is_valid_custom_auth_strategy_valid(self) -> None:
        """Test checking if a valid custom auth strategy class is valid."""

        # Create a mock custom auth strategy class that subclasses AgentAuthStrategy
        class CustomAuthStrategy(AgentAuthStrategy):
            def from_config(cls, config):
                return cls()

            @property
            def auth_type(self):
                return AgentAuthType.CUSTOM

            async def get_headers(self):
                return {}

        result = AgentAuthFactory._is_valid_custom_auth_strategy(CustomAuthStrategy)
        assert result is True

    def test_is_valid_custom_auth_strategy_invalid(self) -> None:
        """Test checking if an invalid custom auth strategy class is invalid."""

        # Create a mock class that doesn't subclass AgentAuthStrategy
        class InvalidAuthStrategy:
            pass

        result = AgentAuthFactory._is_valid_custom_auth_strategy(InvalidAuthStrategy)
        assert result is False

    def test_get_agent_auth_strategy_base_class(self) -> None:
        """Test getting the agent auth strategy base class."""
        result = AgentAuthFactory.get_agent_auth_strategy_base_class()
        assert result is AgentAuthStrategy

    @patch("rasa.shared.agents.auth.agent_auth_factory.class_from_module_path")
    def test_create_client_custom_auth_strategy_valid(
        self,
        mock_class_from_module_path: MagicMock,
        mock_custom_auth_config: dict,
    ) -> None:
        """Test creating a client with a valid custom auth strategy."""

        # Create a mock custom auth strategy class
        class CustomAuthStrategy(APIKeyAuthStrategy):
            async def get_headers(self):
                return {}

        mock_class_from_module_path.return_value = CustomAuthStrategy

        # Mock the from_config method
        with patch.object(CustomAuthStrategy, "from_config") as mock_from_config:
            mock_auth_instance = MagicMock(spec=AgentAuthStrategy)
            mock_from_config.return_value = mock_auth_instance

            result = AgentAuthFactory.create_client(
                AgentAuthType.CUSTOM, mock_custom_auth_config
            )

            # Verify the custom auth strategy was created
            mock_class_from_module_path.assert_called_once_with(
                "test.custom_auth.CustomAuthStrategy"
            )
            mock_from_config.assert_called_once_with(mock_custom_auth_config)
            assert result is mock_auth_instance

    @patch("rasa.shared.agents.auth.agent_auth_factory.class_from_module_path")
    def test_create_client_custom_auth_strategy_invalid(
        self,
        mock_class_from_module_path: MagicMock,
        mock_custom_auth_config: dict,
    ) -> None:
        """Test creating a client with an invalid custom auth strategy."""

        # Create a mock class that doesn't subclass AgentAuthStrategy
        class InvalidAuthStrategy:
            pass

        mock_class_from_module_path.return_value = InvalidAuthStrategy

        with pytest.raises(
            ValueError,
            match="Authentication strategy class .*InvalidAuthStrategy.* must subclass",
        ):
            AgentAuthFactory.create_client(
                AgentAuthType.CUSTOM, mock_custom_auth_config
            )

    @patch("rasa.shared.agents.auth.agent_auth_factory.class_from_module_path")
    def test_create_client_custom_auth_strategy_module_not_found(
        self,
        mock_class_from_module_path: MagicMock,
        mock_custom_auth_config: dict,
    ) -> None:
        """Test creating a client when custom auth strategy module is not found."""
        mock_class_from_module_path.side_effect = ImportError("Module not found")

        with pytest.raises(ImportError, match="Module not found"):
            AgentAuthFactory.create_client(
                AgentAuthType.CUSTOM, mock_custom_auth_config
            )

    @pytest.mark.parametrize(
        "auth_type,config",
        [
            (AgentAuthType.API_KEY, {"api_key": "test_key"}),
            (
                AgentAuthType.OAUTH2,
                {
                    "client_id": "test_id",
                    "client_secret": "test_secret",
                    "token_url": "https://example.com/token",
                    "scope": "read write",
                    "grant_type": "client_credentials",
                },
            ),
            (AgentAuthType.BEARER_TOKEN, {"token": "test_token"}),
        ],
    )
    def test_create_client_all_supported_types(
        self, auth_type: AgentAuthType, config: dict
    ):
        """Test creating clients for all supported auth types."""
        result = AgentAuthFactory.create_client(auth_type, config)

        assert isinstance(result, AgentAuthStrategy)
        assert result.auth_type == auth_type

    @pytest.mark.parametrize(
        "auth_type,config,expected_error",
        [
            # Unsupported auth type
            (
                MagicMock(value="unsupported_auth_type"),
                {"api_key": "test_key"},
                "Unsupported authentication strategy type",
            ),
            # API key auth with None config
            (
                AgentAuthType.API_KEY,
                None,
                "API key is required for API KEY authentication",
            ),
            # API key auth with empty config
            (
                AgentAuthType.API_KEY,
                {},
                "API key is required for API KEY authentication",
            ),
            # Bearer token auth with None config
            (
                AgentAuthType.BEARER_TOKEN,
                None,
                "Access token is required for Bearer Token authentication",
            ),
            # Bearer token auth with empty config
            (
                AgentAuthType.BEARER_TOKEN,
                {},
                "Access token is required for Bearer Token authentication",
            ),
        ],
    )
    def test_create_client_error_cases(
        self, auth_type: AgentAuthType, config: dict, expected_error: str
    ) -> None:
        """Test creating clients with various error conditions."""
        with pytest.raises(ValueError, match=expected_error):
            AgentAuthFactory.create_client(auth_type, config)
