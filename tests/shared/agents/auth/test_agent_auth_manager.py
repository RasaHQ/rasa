"""Unit tests for AgentAuthManager."""

from unittest.mock import MagicMock, patch

import pytest
import structlog

from rasa.shared.agents.auth.agent_auth_manager import AgentAuthManager
from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.types import AgentAuthType
from rasa.shared.exceptions import AgentAuthInitializationException
from tests.utilities import filter_logs


class TestAgentAuthManager:
    """Test cases for AgentAuthManager."""

    @pytest.fixture
    def mock_auth_strategy(self):
        """Fixture for mock auth strategy."""
        return MagicMock(spec=AgentAuthStrategy)

    @pytest.fixture
    def auth_manager_with_strategy(self, mock_auth_strategy):
        """Fixture for auth manager with strategy."""
        return AgentAuthManager(auth_strategy=mock_auth_strategy)

    @pytest.fixture
    def auth_manager_without_strategy(self):
        """Fixture for auth manager without strategy."""
        return AgentAuthManager()

    def test_init_with_auth_strategy(self):
        """Test initialization with an auth strategy."""
        mock_strategy = MagicMock(spec=AgentAuthStrategy)
        manager = AgentAuthManager(auth_strategy=mock_strategy)

        assert manager._auth_strategy is mock_strategy

    def test_init_without_auth_strategy(self):
        """Test initialization without an auth strategy."""
        manager = AgentAuthManager()

        assert manager._auth_strategy is None

    def test_get_auth_with_strategy(self):
        """Test getting auth when strategy is available."""
        mock_strategy = MagicMock(spec=AgentAuthStrategy)
        manager = AgentAuthManager(auth_strategy=mock_strategy)

        result = manager.get_auth()

        assert result is mock_strategy

    def test_get_auth_without_strategy(self):
        """Test getting auth when no strategy is available."""
        manager = AgentAuthManager()

        with pytest.raises(ValueError, match="No authentication instance available"):
            manager.get_auth()

    @patch("rasa.shared.agents.auth.agent_auth_manager.AgentAuthFactory")
    def test_load_auth_success(self, mock_factory):
        """Test successful loading of authentication."""
        # Setup
        config = {"api_key": "test_key"}
        mock_strategy = MagicMock(spec=AgentAuthStrategy)
        mock_factory.create_client.return_value = mock_strategy

        # Execute
        manager = AgentAuthManager.load_auth(config)

        # Verify
        assert isinstance(manager, AgentAuthManager)
        assert manager._auth_strategy is mock_strategy
        mock_factory.create_client.assert_called_once_with(
            AgentAuthType.API_KEY, config
        )

    @patch("rasa.shared.agents.auth.agent_auth_manager.AgentAuthFactory")
    def test_load_auth_with_none_config(self, mock_factory):
        """Test loading auth with None config."""
        # Setup
        config = None

        # Execute & Verify
        manager = AgentAuthManager.load_auth(config)
        assert manager is None

        mock_factory.create_client.assert_not_called()

    @patch("rasa.shared.agents.auth.agent_auth_manager.AgentAuthFactory")
    def test_load_auth_invalid_type(self, mock_factory):
        """Test loading auth with invalid type in config."""
        # Setup
        config = {"invalid_key": "value"}

        # Execute & Verify
        with pytest.raises(AgentAuthInitializationException):
            AgentAuthManager.load_auth(config)

        mock_factory.create_client.assert_not_called()

    @patch("rasa.shared.agents.auth.agent_auth_manager.AgentAuthFactory")
    def test_load_auth_factory_exception(self, mock_factory):
        """Test loading auth when factory raises exception."""
        # Setup
        config = {"api_key": "test_key"}
        mock_factory.create_client.side_effect = ValueError("Factory error")

        # Execute & Verify
        with pytest.raises(AgentAuthInitializationException) as exc_info:
            AgentAuthManager.load_auth(config)

        # The exception should be raised and contain information
        # about the original error
        assert "Factory error" in str(exc_info.value)

    def test_detect_auth_type_api_key(self):
        """Test detecting API key authentication type."""
        config = {"api_key": "test_key"}

        result = AgentAuthManager.detect_auth_type(config)

        assert result == AgentAuthType.API_KEY

    def test_detect_auth_type_bearer_token(self):
        """Test detecting bearer token authentication type."""
        config = {"token": "test_token"}

        result = AgentAuthManager.detect_auth_type(config)

        assert result == AgentAuthType.BEARER_TOKEN

    def test_detect_auth_type_oauth2(self):
        """Test detecting OAuth2 authentication type."""
        config = {"oauth": {"client_id": "test_id"}}

        result = AgentAuthManager.detect_auth_type(config)

        assert result == AgentAuthType.OAUTH2

    def test_detect_auth_type_invalid_config(self):
        """Test detecting authentication type with invalid config."""
        config = {"invalid_key": "value"}

        with pytest.raises(ValueError, match="Invalid authentication type"):
            AgentAuthManager.detect_auth_type(config)

    def test_detect_auth_type_empty_config(self):
        """Test detecting authentication type with empty config."""
        config = {}

        with pytest.raises(ValueError, match="Invalid authentication type"):
            AgentAuthManager.detect_auth_type(config)

    @patch("rasa.shared.agents.auth.agent_auth_manager.AgentAuthFactory")
    def test_load_auth_logging_success(self, mock_factory):
        """Test that successful auth loading is logged."""
        # Setup
        config = {"api_key": "test_key"}
        mock_strategy = MagicMock(spec=AgentAuthStrategy)
        mock_factory.create_client.return_value = mock_strategy

        expected_event = "agent_auth_manager.load_auth.success"
        expected_log_level = "debug"
        expected_log_message_parts = [
            "Loaded authentication client successfully for `api_key`"
        ]

        # Execute
        with structlog.testing.capture_logs() as caplog:
            AgentAuthManager.load_auth(config)

        # Verify logging
        logs = filter_logs(
            caplog, expected_event, expected_log_level, expected_log_message_parts
        )
        assert len(logs) == 1
        assert logs[0]["auth_type"] == "api_key"

    @patch("rasa.shared.agents.auth.agent_auth_manager.AgentAuthFactory")
    def test_load_auth_logging_failure_missing_type(self, mock_factory):
        """Test that failed auth loading is logged when type is missing."""
        # Setup
        config = {"invalid_key": "test_key"}

        expected_event = "agent_auth_manager.load_auth.failed_to_load"
        expected_log_level = "error"

        # Execute
        with structlog.testing.capture_logs() as caplog:
            with pytest.raises(AgentAuthInitializationException):
                AgentAuthManager.load_auth(config)

        # Verify logging
        logs = filter_logs(caplog, expected_event, expected_log_level)
        assert len(logs) >= 1
        mock_factory.create_client.assert_not_called()

    @patch("rasa.shared.agents.auth.agent_auth_manager.AgentAuthFactory")
    def test_load_auth_bearer_token_success(self, mock_factory):
        """Test successful loading of bearer token authentication."""
        # Setup
        config = {"token": "test_token"}
        mock_strategy = MagicMock(spec=AgentAuthStrategy)
        mock_factory.create_client.return_value = mock_strategy

        # Execute
        manager = AgentAuthManager.load_auth(config)

        # Verify
        assert isinstance(manager, AgentAuthManager)
        assert manager._auth_strategy is mock_strategy
        mock_factory.create_client.assert_called_once_with(
            AgentAuthType.BEARER_TOKEN, config
        )

    @patch("rasa.shared.agents.auth.agent_auth_manager.AgentAuthFactory")
    def test_load_auth_oauth2_success(self, mock_factory):
        """Test successful loading of OAuth2 authentication."""
        # Setup
        config = {"oauth": {"client_id": "test_id", "client_secret": "test_secret"}}
        mock_strategy = MagicMock(spec=AgentAuthStrategy)
        mock_factory.create_client.return_value = mock_strategy

        # Execute
        manager = AgentAuthManager.load_auth(config)

        # Verify
        assert isinstance(manager, AgentAuthManager)
        assert manager._auth_strategy is mock_strategy
        mock_factory.create_client.assert_called_once_with(AgentAuthType.OAUTH2, config)
