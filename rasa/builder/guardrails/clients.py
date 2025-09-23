"""Guardrails client implementations."""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
import structlog

from rasa.builder import config
from rasa.builder.guardrails.constants import (
    LAKERA_API_KEY_ENV_VAR,
    LAKERA_GUARD_ENDPOINT,
    LAKERA_GUARD_RESULTS_ENDPOINT,
)
from rasa.builder.guardrails.exceptions import GuardrailsError
from rasa.builder.guardrails.models import (
    GuardrailRequest,
    GuardrailResponse,
    LakeraGuardrailRequest,
    LakeraGuardrailResponse,
)

structlogger = structlog.get_logger()


class GuardrailsClient(ABC):
    """Abstract base class for guardrails clients."""

    @property
    @abstractmethod
    def guard_endpoint(self) -> str:
        """Get the guard endpoint for the guardrails API."""
        pass

    @abstractmethod
    async def send_request(self, request: GuardrailRequest) -> GuardrailResponse:
        """Send a request to the guardrails provider.

        Args:
            request: The guardrail request to send to the provider.

        Returns:
            GuardrailResponse with the results of the check.

        Raises:
            GuardrailsError: If the request fails for any reason.
        """
        pass

    @lru_cache(maxsize=512)
    def schedule_check(
        self,
        request: GuardrailRequest,
    ) -> "asyncio.Task[GuardrailResponse]":
        """Return a cached asyncio.Task that resolves to guardrail response.

        Args:
            request: The guardrail request to send to the provider.

        Returns:
            An asyncio Task that resolves to a GuardrailResponse.
        """
        structlogger.debug(
            "guardrails.schedule_check.cache_miss",
            request=request.model_dump(),
        )
        loop = asyncio.get_running_loop()
        return loop.create_task(self.send_request(request))


class LakeraAIGuardrails(GuardrailsClient):
    """Guardrails provider using Lakera AI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize Lakera guardrails provider.

        Args:
            api_key: Lakera AI API key.
            base_url: Optional base URL for the API. If not provided, the default
                Lakera API URL (https://api.lakera.ai/v2) will be used.
        """
        self.base_url: str = base_url or config.LAKERA_BASE_URL
        self._api_key: Optional[str] = api_key or os.getenv(LAKERA_API_KEY_ENV_VAR)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def guard_endpoint(self) -> str:
        """Get the guard endpoint for the Lakera API."""
        return f"{self.base_url}/{LAKERA_GUARD_ENDPOINT}"

    @property
    def guard_results_endpoint(self) -> str:
        """Get the guard results endpoint for the Lakera API."""
        return f"{self.base_url}/{LAKERA_GUARD_RESULTS_ENDPOINT}"

    @asynccontextmanager
    async def _get_session(self) -> AsyncGenerator[aiohttp.ClientSession, None]:
        """Create a fresh ClientSession, yield it, and always close it."""
        session = aiohttp.ClientSession(headers=self._get_headers())
        structlogger.debug("lakera_guardrails._get_session", base_url=self.base_url)

        try:
            yield session
        except Exception as e:
            structlogger.error("lakera_guardrails.session_error", error=str(e))
            raise
        finally:
            try:
                await session.close()
            except Exception as exc:
                structlogger.warning(
                    "lakera_guardrails.session_close_error",
                    event_info="Failed to close aiohttp client session cleanly.",
                    error=str(exc),
                )

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for the Lakera API request.

        Returns:
            A dictionary containing the Authorization header with the API key.
        """
        using_proxy = bool(config.HELLO_LLM_PROXY_BASE_URL)

        if using_proxy:
            if not config.RASA_PRO_LICENSE:
                raise GuardrailsError(
                    "HELLO_LLM_PROXY_BASE_URL is set but RASA_PRO_LICENSE is missing. "
                    "Proxy requires a Rasa Pro license token for authentication."
                )
            return {"Authorization": f"Bearer {config.RASA_PRO_LICENSE}"}

        if not self._api_key:
            raise GuardrailsError(
                "LAKERA_API_KEY is missing. Provide it via env LAKERA_API_KEY or "
                "pass api_key= to LakeraAIGuardrails."
            )
        return {"Authorization": f"Bearer {self._api_key}"}

    async def send_request(self, request: GuardrailRequest) -> GuardrailResponse:
        """Send a request to the Lakera API.

        Args:
            request: The guardrail request to send to the Lakera API.

        Returns:
            GuardrailResponse with the results of the check.

        Raises:
            GuardrailsError: If the request times out or returns a non-200 status code.
            Exception: If the request fails for any other reason.
        """
        if not isinstance(request, LakeraGuardrailRequest):
            raise GuardrailsError(
                "LakeraAIGuardrails only supports LakeraGuardrailRequest"
            )
        start_time = time.time()
        try:
            async with self._get_session() as session:
                raw_response = await self._send_http_request(session, request)
                response = LakeraGuardrailResponse.from_raw_response(
                    raw_response,
                    hello_rasa_user_id=request.hello_rasa_user_id,
                    hello_rasa_project_id=request.hello_rasa_project_id,
                )
                processing_time_ms = (time.time() - start_time) * 1000
                response.processing_time_ms = processing_time_ms
                return response

        # Propagate the GuardrailsError if it occurs.
        except GuardrailsError as e:
            raise e

        except asyncio.TimeoutError:
            message = "Lakera API request timed out."
            structlogger.error(
                "lakera_guardrails.send_request.timeout_error",
                event_info=message,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            raise GuardrailsError(message)

        # Propagate the unexpected exceptions.
        except Exception as e:
            message = "Lakera API request failed."
            structlogger.error(
                "lakera_guardrails.send_request.unexpected_error",
                event_info="Lakera API request failed.",
                request=request,
                error=e,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            raise e

    async def _send_http_request(
        self, session: aiohttp.ClientSession, request: LakeraGuardrailRequest
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Lakera API.

        Args:
            session: The aiohttp session to use for the request.
            request: The guardrail request to send.

        Returns:
            The raw JSON response from the API.

        Raises:
            GuardrailsError: If the request fails or response parsing fails.
        """
        # Log the request details for debugging
        json_payload = request.to_json_payload()
        structlogger.debug(
            "lakera_guardrails.send_request.request",
            url=self.guard_endpoint,
            method="POST",
            request_body=json_payload,
        )

        async with session.post(
            self.guard_endpoint,
            json=json_payload,
        ) as client_response:
            # Check if the response is successful. If not, raise an error.
            if client_response.status >= 400:
                error_text = await client_response.text()
                message = (
                    f"Lakera API request failed with status "
                    f"`{client_response.status}`. Error: "
                    f"`{error_text}`."
                )
                structlogger.error(
                    "lakera_guardrails.send_request.http_error",
                    event_info=message,
                    url=self.guard_endpoint,
                    status=client_response.status,
                    error=error_text,
                    request_body=json_payload,
                )
                raise GuardrailsError(message)

            # Parse the response as a dictionary.
            raw_response = await client_response.json()
            structlogger.debug(
                "lakera_guardrails.send_request.response",
                response_body=raw_response,
            )
            return raw_response
