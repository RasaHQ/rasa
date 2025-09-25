from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict

import httpx

from rasa.shared.agents.auth.types import AgentAuthType
from rasa.shared.utils.io import resolve_environment_variables


class AgentAuthStrategy(ABC, httpx.Auth):
    """Base class for authentication strategies."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "AgentAuthStrategy":
        """Create an authentication strategy instance from a config dictionary."""
        ...

    @property
    @abstractmethod
    def auth_type(self) -> AgentAuthType:
        """Return the type of the authentication strategy."""
        ...

    @abstractmethod
    async def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests.

        Returns:
            Dictionary containing authentication headers.
        """
        ...

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Get authentication headers for requests.

        Returns:
            Async generator of requests with authentication headers.
        """
        # Get authentication headers.
        auth_headers = await self.get_headers()

        # Resolve environment variables in authentication headers.
        resolved_auth_headers = resolve_environment_variables(auth_headers)

        # Update request headers with resolved authentication headers.
        request.headers.update(resolved_auth_headers)

        # Yield request with resolved authentication headers.
        yield request
