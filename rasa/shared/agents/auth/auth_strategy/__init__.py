"""Authentication strategy implementations."""

from rasa.shared.agents.auth.auth_strategy.agent_auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.auth_strategy.api_key_auth_strategy import (
    APIKeyAuthStrategy,
)
from rasa.shared.agents.auth.auth_strategy.bearer_token_auth_strategy import (
    BearerTokenAuthStrategy,
)
from rasa.shared.agents.auth.auth_strategy.oauth2_auth_strategy import (
    OAuth2AuthStrategy,
)

__all__ = [
    "AgentAuthStrategy",
    "APIKeyAuthStrategy",
    "OAuth2AuthStrategy",
    "BearerTokenAuthStrategy",
]
