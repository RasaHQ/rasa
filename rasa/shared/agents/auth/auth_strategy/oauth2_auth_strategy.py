"""OAuth2 authentication strategy client credentials flow implementation."""

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from rasa.shared.agents.auth.auth_strategy import AgentAuthStrategy
from rasa.shared.agents.auth.constants import (
    CONFIG_AUDIENCE_KEY,
    CONFIG_CLIENT_ID_KEY,
    CONFIG_CLIENT_SECRET_KEY,
    CONFIG_OAUTH_KEY,
    CONFIG_SCOPE_KEY,
    CONFIG_TIMEOUT_KEY,
    CONFIG_TOKEN_URL_KEY,
)
from rasa.shared.agents.auth.types import AgentAuthType
from rasa.shared.utils.io import resolve_environment_variables

KEY_ACCESS_TOKEN = "access_token"
KEY_CLIENT_CREDENTIALS = "client_credentials"
KEY_EXPIRES_IN = "expires_in"


class OAuth2AuthStrategy(AgentAuthStrategy):
    """Client Credentials Flow authentication strategy."""

    DEFAULT_ACCESS_TOKEN_TIMEOUT = 5
    DEFAULT_ACCESS_TOKEN_EXPIRES_IN_SECONDS = 3600
    DEFAULT_BUFFER_TIME_SECONDS = 10

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        audience: Optional[str] = None,
        scope: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.audience = audience
        self.scope = scope
        self.timeout = timeout or self.DEFAULT_ACCESS_TOKEN_TIMEOUT

        # Only client credentials flow is supported for server to server communication.
        self._grant_type = KEY_CLIENT_CREDENTIALS

        # Initialize defaults.
        self._access_token: Optional[str] = None
        self._expires_at: Optional[float] = None

        # Initialize lock for concurrent access to the refresh the access token.
        self._refresh_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OAuth2AuthStrategy":
        """Create OAuth2AuthStrategy from configuration dictionary."""
        # Extract OAuth2 config from nested "oauth" key if present
        oauth_config = config.get(CONFIG_OAUTH_KEY, config)
        if not oauth_config:
            raise ValueError("OAuth2 configuration is required")

        token_url = oauth_config.get(CONFIG_TOKEN_URL_KEY)
        client_id = oauth_config.get(CONFIG_CLIENT_ID_KEY)
        client_secret = oauth_config.get(CONFIG_CLIENT_SECRET_KEY)
        audience = oauth_config.get(CONFIG_AUDIENCE_KEY)
        scope = oauth_config.get(CONFIG_SCOPE_KEY)
        timeout = (
            oauth_config.get(CONFIG_TIMEOUT_KEY) or cls.DEFAULT_ACCESS_TOKEN_TIMEOUT
        )

        if not token_url:
            raise ValueError("Token URL is required for OAuth2 authentication")
        if not client_id:
            raise ValueError("Client ID is required for OAuth2 authentication")
        if not client_secret:
            raise ValueError("Client secret is required for OAuth2 authentication")

        return cls(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            audience=audience,
            scope=scope,
            timeout=timeout,
        )

    @property
    def auth_type(self) -> AgentAuthType:
        return AgentAuthType.OAUTH2

    async def get_headers(self) -> Dict[str, str]:
        """Return OAuth2 authentication headers."""
        # Acquire lock to prevent concurrent access to the refresh the access token.
        async with self._refresh_lock:
            # Refresh if missing or expired
            if not self._access_token or self._is_expired():
                await self._refresh_access_token()
            if not self._access_token:
                raise ValueError("Failed to obtain access token")
        return {"Authorization": f"Bearer {self._access_token}"}

    def _is_expired(self) -> bool:
        """Check if access token is expired."""
        # Adding a buffer time to the expiration time to avoid race conditions.
        return (
            not self._expires_at
            or self._expires_at <= time.time() + self.DEFAULT_BUFFER_TIME_SECONDS
        )

    async def _refresh_access_token(self) -> None:
        """Fetch a new access token using client credentials flow."""
        # Prepare data and headers
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": self._grant_type,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self.scope:
            data["scope"] = self.scope
        if self.audience:
            data["audience"] = self.audience

        # Resolve environment variables in data.
        resolved_data = resolve_environment_variables(data)

        # Fetch access token
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.token_url, data=resolved_data, headers=headers
                )
                resp.raise_for_status()
                token_data = resp.json()
        except httpx.HTTPStatusError as e:
            raise e
        except httpx.RequestError as e:
            raise ValueError(f"OAuth2 token request failed - {e}") from e
        except Exception as e:
            raise ValueError(
                f"Unexpected error during OAuth2 token request - {e}"
            ) from e

        # Validate token data
        if KEY_ACCESS_TOKEN not in token_data:
            raise ValueError(f"No `{KEY_ACCESS_TOKEN}` in OAuth2 response")

        # Set access token and expires at
        self._access_token = token_data[KEY_ACCESS_TOKEN]
        self._expires_at = time.time() + token_data.get(
            KEY_EXPIRES_IN, self.DEFAULT_ACCESS_TOKEN_EXPIRES_IN_SECONDS
        )

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Inject Authorization header into outgoing requests.

        Returns:
            Async generator of requests with Authorization header.
        """
        auth_headers = await self.get_headers()
        request.headers.update(auth_headers)
        yield request
