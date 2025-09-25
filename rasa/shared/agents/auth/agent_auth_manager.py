"""High-level authentication management with strategy abstraction."""

from typing import Any, Dict, Optional

import structlog

from rasa.shared.agents.auth.agent_auth_factory import AgentAuthFactory
from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.constants import (
    CONFIG_API_KEY_KEY,
    CONFIG_MODULE_KEY,
    CONFIG_OAUTH_KEY,
    CONFIG_TOKEN_KEY,
)
from rasa.shared.agents.auth.types import AgentAuthType
from rasa.shared.exceptions import AgentAuthInitializationException

structlogger = structlog.get_logger()


class AgentAuthManager:
    """High-level authentication management with strategy abstraction."""

    def __init__(self, auth_strategy: Optional[AgentAuthStrategy] = None):
        """Initialize a new authentication manager instance."""
        self._auth_strategy: Optional[AgentAuthStrategy] = auth_strategy

    def get_auth(self) -> AgentAuthStrategy:
        """Retrieve the authentication instance.

        Returns:
            The authentication instance.

        Raises:
            ValueError: If no authentication is connected.
        """
        if self._auth_strategy is None:
            raise ValueError("No authentication instance available")
        return self._auth_strategy

    @staticmethod
    def detect_auth_type(config: Dict[str, Any]) -> AgentAuthType:
        if CONFIG_MODULE_KEY in config:
            return AgentAuthType.CUSTOM
        if CONFIG_API_KEY_KEY in config:
            return AgentAuthType.API_KEY
        if CONFIG_TOKEN_KEY in config:
            return AgentAuthType.BEARER_TOKEN
        if CONFIG_OAUTH_KEY in config:
            return AgentAuthType.OAUTH2
        raise ValueError(
            "Invalid authentication type. Supported types: "
            + ", ".join([auth_type.value for auth_type in AgentAuthType])
        )

    @classmethod
    def load_auth(
        cls, config: Optional[Dict[str, Any]]
    ) -> Optional["AgentAuthManager"]:
        """Connect to authentication using specified strategy type and persist
        the auth instance to the manager in a ready-to-use state.

        Args:
            config: The configuration dictionary for the authentication.

        Raises:
            AgentAuthInitializationException: If the authentication connection fails.
        """
        if not config:
            return None
        try:
            auth_type = AgentAuthManager.detect_auth_type(config)

            # Create the auth client
            client = AgentAuthFactory.create_client(auth_type, config)

            structlogger.debug(
                "agent_auth_manager.load_auth.success",
                auth_type=auth_type.value,
                event_info=(
                    f"Loaded authentication client successfully for `{auth_type.value}`"
                ),
            )
            return cls(client)
        except Exception as e:
            event_info = "Failed to load authentication client"
            structlogger.error(
                "agent_auth_manager.load_auth.failed_to_load",
                event_info=event_info,
                config=config,
            )
            raise AgentAuthInitializationException(e) from e
