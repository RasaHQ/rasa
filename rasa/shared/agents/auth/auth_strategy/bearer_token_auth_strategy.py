"""Bearer Token authentication strategy implementation."""

from typing import Any, Dict

from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.constants import CONFIG_TOKEN_KEY
from rasa.shared.agents.auth.types import AgentAuthType


class BearerTokenAuthStrategy(AgentAuthStrategy):
    """Bearer Token authentication strategy implementation."""

    def __init__(self, token: str):
        self._token = token

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BearerTokenAuthStrategy":
        token = config.get(CONFIG_TOKEN_KEY)
        if not token:
            raise ValueError("Access token is required for Bearer Token authentication")
        return BearerTokenAuthStrategy(token)

    @property
    def auth_type(self) -> AgentAuthType:
        return AgentAuthType.BEARER_TOKEN

    async def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}
