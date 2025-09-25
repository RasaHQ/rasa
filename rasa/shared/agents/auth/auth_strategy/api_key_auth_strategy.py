"""API Key authentication strategy implementation."""

from typing import Any, Dict

from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.constants import (
    CONFIG_API_KEY_KEY,
    CONFIG_HEADER_FORMAT_KEY,
    CONFIG_HEADER_NAME_KEY,
)
from rasa.shared.agents.auth.types import AgentAuthType


class APIKeyAuthStrategy(AgentAuthStrategy):
    """API Key authentication strategy implementation."""

    def __init__(
        self,
        api_key: str,
        header_name: str = "Authorization",
        header_format: str = "Bearer {key}",
    ):
        self.api_key = api_key
        self.header_name = header_name
        self.header_format = header_format

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "APIKeyAuthStrategy":
        api_key = config.get(CONFIG_API_KEY_KEY)
        header_name = config.get(CONFIG_HEADER_NAME_KEY, "Authorization")
        header_format = config.get(CONFIG_HEADER_FORMAT_KEY, "Bearer {key}")
        if not api_key:
            raise ValueError("API key is required for API KEY authentication")
        return APIKeyAuthStrategy(api_key, header_name, header_format)

    @property
    def auth_type(self) -> AgentAuthType:
        return AgentAuthType.API_KEY

    async def get_headers(self) -> Dict[str, str]:
        """Return API key authentication headers."""
        return {self.header_name: self.header_format.format(key=self.api_key)}
