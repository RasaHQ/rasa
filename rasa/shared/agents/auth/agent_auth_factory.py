"""Factory for creating authentication strategy instances based on the strategy type."""

from typing import Any, ClassVar, Dict, Optional, Type

import structlog

from rasa.shared.agents.auth.auth_strategy import (
    AgentAuthStrategy,
    APIKeyAuthStrategy,
    BearerTokenAuthStrategy,
    OAuth2AuthStrategy,
)
from rasa.shared.agents.auth.constants import CONFIG_MODULE_KEY
from rasa.shared.agents.auth.types import AgentAuthType
from rasa.shared.utils.common import class_from_module_path

structlogger = structlog.get_logger()


class AgentAuthFactory:
    """Factory for creating authentication strategy instances based on the
    authentication strategy type.
    """

    _auth_strategies: ClassVar[Dict[AgentAuthType, Type[AgentAuthStrategy]]] = {
        AgentAuthType.API_KEY: APIKeyAuthStrategy,
        AgentAuthType.OAUTH2: OAuth2AuthStrategy,
        AgentAuthType.BEARER_TOKEN: BearerTokenAuthStrategy,
    }

    @classmethod
    def create_client(
        cls, auth_type: AgentAuthType, config: Optional[Dict[str, Any]] = None
    ) -> AgentAuthStrategy:
        """Create an authentication strategy instance based on the strategy type.

        Args:
            auth_type: The type of the authentication strategy.
            config: The configuration dictionary for the authentication.

        Returns:
            An instance of the authentication strategy.
        """
        config = config or {}

        # If the auth type is custom, we need to create it from the module
        if auth_type == AgentAuthType.CUSTOM:
            auth_strategy_class: Type[AgentAuthStrategy] = class_from_module_path(
                config[CONFIG_MODULE_KEY]
            )
            if not cls._is_valid_custom_auth_strategy(auth_strategy_class):
                raise ValueError(
                    f"Authentication strategy class `{auth_strategy_class}` must "
                    f"subclass the "
                    f"`{cls.get_agent_auth_strategy_base_class().__name__}` class."
                )
            structlogger.debug(
                "agent_auth_factory.create_client.custom_auth_strategy",
                event_info=f"Initializing `{auth_strategy_class.__name__}`",
                auth_type=auth_type.value,
                auth_strategy_class=auth_strategy_class.__name__,
            )
            return auth_strategy_class.from_config(config)

        # Get the strategy class for the specified type
        auth_strategy_class = cls._get_auth_strategy_class(auth_type)
        if auth_strategy_class is None:
            raise ValueError(
                f"Unsupported strategy type: {auth_type}. "
                f"Supported types: {cls.get_supported_auth_strategy_types()}"
            )
        # Create instance based on strategy type
        return auth_strategy_class.from_config(config)

    @classmethod
    def get_supported_auth_strategy_types(cls) -> list[AgentAuthType]:
        """Get all supported authentication strategy types."""
        return list(cls._auth_strategies.keys())

    @classmethod
    def _get_auth_strategy_class(
        cls, auth_type: AgentAuthType
    ) -> Type[AgentAuthStrategy]:
        """Get the class that implements the authentication strategy."""
        if not cls.is_auth_strategy_supported(auth_type):
            raise ValueError(
                f"Unsupported authentication strategy type: {auth_type}. "
                f"Supported types: {cls.get_supported_auth_strategy_types()}"
            )
        return cls._auth_strategies[auth_type]

    @classmethod
    def is_auth_strategy_supported(cls, auth_type: AgentAuthType) -> bool:
        """Check if the authentication strategy type is supported."""
        return auth_type in cls._auth_strategies

    @classmethod
    def _is_valid_custom_auth_strategy(cls, strategy_class: Any) -> bool:
        """Check if the authentication strategy class is valid."""
        return issubclass(strategy_class, cls.get_agent_auth_strategy_base_class())

    @staticmethod
    def get_agent_auth_strategy_base_class() -> Type[AgentAuthStrategy]:
        """Get the agent authentication strategy base class."""
        return AgentAuthStrategy
