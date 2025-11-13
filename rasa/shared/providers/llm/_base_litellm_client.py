from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict, List, NoReturn, Union, cast

import structlog
from litellm import acompletion, completion, validate_environment

from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.shared.constants import (
    _VALIDATE_ENVIRONMENT_MISSING_KEYS_KEY,
    API_BASE_CONFIG_KEY,
    API_KEY,
    ROLE_USER,
)
from rasa.shared.exceptions import (
    ProviderClientAPIException,
    ProviderClientValidationError,
)
from rasa.shared.providers._ssl_verification_utils import (
    ensure_ssl_certificates_for_litellm_non_openai_based_clients,
    ensure_ssl_certificates_for_litellm_openai_based_clients,
)
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall, LLMUsage
from rasa.shared.utils.io import resolve_environment_variables, suppress_logs

structlogger = structlog.get_logger()

# Suppress LiteLLM info and debug logs - Global level.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class _BaseLiteLLMClient:
    """An abstract base class for LiteLLM clients.

    This class defines the interface and common functionality for all clients
    based on LiteLLM.

    The class is made private to prevent it from being part of the
    public-facing interface, as it serves as an internal base class
    for specific implementations of clients that are currently based on
    LiteLLM.

    By keeping it private, we ensure that only the derived, concrete
    implementations are exposed to users, maintaining a cleaner and
    more controlled API surface.
    """

    def __init__(self):  # type: ignore
        self._ensure_certificates()

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> _BaseLiteLLMClient:
        pass

    @property
    @abstractmethod
    def config(self) -> dict:
        """Returns the configuration for that the llm client in dictionary form."""
        pass

    @property
    @abstractmethod
    def _litellm_model_name(self) -> str:
        """Returns the value of LiteLLM's model parameter.

        To be used in completion/acompletion in LiteLLM format:
        <provider>/<model or deployment name>
        """
        pass

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        """Returns a dictionary of extra parameters.

        Includes model parameters as well as LiteLLM specific input parameters.
        By default, this returns an empty dictionary (no extra parameters).
        """
        return {}

    @property
    def _completion_fn_args(self) -> dict:
        return {
            # Since all providers covered by LiteLLM use the OpenAI format, but
            # not all support every OpenAI parameter, raise an exception if
            # provider/model uses unsupported parameter
            "drop_params": False,
            # All other parameters set through config, can override drop_params
            **self._litellm_extra_parameters,
            # Model name is constructed in the LiteLLM format from the provided config
            # Non-overridable to ensure consistency
            "model": self._litellm_model_name,
        }

    def validate_client_setup(self) -> None:
        """Perform client validation.

        By default only environment variables are validated.

        Raises:
            ProviderClientValidationError if validation fails.
        """
        self._validate_environment_variables()

    def _validate_environment_variables(self) -> None:
        """Validate that the required environment variables are set."""
        validation_info = validate_environment(
            self._litellm_model_name,
            api_key=self._litellm_extra_parameters.get(API_KEY),
            api_base=self._litellm_extra_parameters.get(API_BASE_CONFIG_KEY),
        )
        if missing_environment_variables := validation_info.get(
            _VALIDATE_ENVIRONMENT_MISSING_KEYS_KEY
        ):
            event_info = (
                f"Environment variables: {missing_environment_variables} "
                f"not set. Required for API calls."
            )
            structlogger.error(
                "base_litellm_client.validate_environment_variables",
                event_info=event_info,
                missing_environment_variables=missing_environment_variables,
            )
            raise ProviderClientValidationError(event_info)

    @suppress_logs(log_level=logging.WARNING)
    def completion(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> LLMResponse:
        """Synchronously generate completions for given list of messages.

        Args:
            messages: The message can be,
                - a list of preformatted messages. Each message should be a dictionary
                    with the following keys:
                    - content: The message content.
                    - role: The role of the message (e.g. user or system).
                - a list of messages. Each message is a string and will be formatted
                    as a user message.
                - a single message as a string which will be formatted as user message.
            **kwargs: Additional parameters to pass to the completion call.

        Returns:
            List of message completions.

        Raises:
            ProviderClientAPIException: If the API request fails.
        """
        try:
            formatted_messages = self._get_formatted_messages(messages)
            arguments = cast(
                Dict[str, Any], resolve_environment_variables(self._completion_fn_args)
            )
            response = completion(
                messages=formatted_messages, **{**arguments, **kwargs}
            )
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(e) from e

    @suppress_logs(log_level=logging.WARNING)
    async def acompletion(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> LLMResponse:
        """Asynchronously generate completions for given list of messages.

        Args:
            messages: The message can be,
                - a list of preformatted messages. Each message should be a dictionary
                    with the following keys:
                    - content: The message content.
                    - role: The role of the message (e.g. user or system).
                - a list of messages. Each message is a string and will be formatted
                    as a user message.
                - a single message as a string which will be formatted as user message.
            **kwargs: Additional parameters to pass to the completion call.

        Returns:
            List of message completions.

        Raises:
            ProviderClientAPIException: If the API request fails.
        """
        try:
            formatted_messages = self._get_formatted_messages(messages)
            arguments = cast(
                Dict[str, Any], resolve_environment_variables(self._completion_fn_args)
            )

            timeout = self._litellm_extra_parameters.get(
                "timeout", DEFAULT_REQUEST_TIMEOUT
            )
            response = await asyncio.wait_for(
                acompletion(messages=formatted_messages, **{**arguments, **kwargs}),
                timeout=timeout,
            )
            return self._format_response(response)
        except asyncio.TimeoutError:
            self._handle_timeout_error()
        except Exception as e:
            message = ""
            from rasa.shared.providers.llm.self_hosted_llm_client import (
                SelfHostedLLMClient,
            )

            if isinstance(self, SelfHostedLLMClient):
                message = (
                    "If you are using 'provider=self-hosted' to call a hosted vllm "
                    "server make sure your config is correctly setup. You should have "
                    "the following mandatory keys in your config: "
                    "provider=self-hosted; "
                    "model='<your-vllm-model-name>'; "
                    "api_base='your-hosted-vllm-serv'."
                    "In case you are getting OpenAI connection errors, such as missing "
                    "API key, your configuration is incorrect."
                )
            raise ProviderClientAPIException(e, message) from e

    @suppress_logs(log_level=logging.WARNING)
    async def acompletion_stream(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously generate streaming completions for given list of messages."""
        try:
            formatted_messages = self._get_formatted_messages(messages)
            arguments = cast(
                Dict[str, Any], resolve_environment_variables(self._completion_fn_args)
            )
            timeout = self._litellm_extra_parameters.get(
                "timeout", DEFAULT_REQUEST_TIMEOUT
            )
            response = await asyncio.wait_for(
                acompletion(
                    messages=formatted_messages, stream=True, **{**arguments, **kwargs}
                ),
                timeout=timeout,
            )
            async for chunk in response:
                yield self._format_response_stream(chunk)
        except asyncio.TimeoutError:
            self._handle_timeout_error()
        except Exception as e:
            message = ""
            from rasa.shared.providers.llm.self_hosted_llm_client import (
                SelfHostedLLMClient,
            )

            if isinstance(self, SelfHostedLLMClient):
                message = (
                    "If you are using 'provider=self-hosted' to call a hosted vllm "
                    "server make sure your config is correctly setup. You should have "
                    "the following mandatory keys in your config: "
                    "provider=self-hosted; "
                    "model='<your-vllm-model-name>'; "
                    "api_base='your-hosted-vllm-serv'."
                    "In case you are getting OpenAI connection errors, such as missing "
                    "API key, your configuration is incorrect."
                )
            raise ProviderClientAPIException(e, message) from e

    def _handle_timeout_error(self) -> NoReturn:
        """Handle asyncio.TimeoutError and raise ProviderClientAPIException.

        Raises:
            ProviderClientAPIException: Always raised with formatted timeout error.
        """
        timeout = self._litellm_extra_parameters.get("timeout", DEFAULT_REQUEST_TIMEOUT)
        error_message = (
            f"APITimeoutError - Request timed out. Error_str: "
            f"Request timed out. - timeout value={timeout:.6f}, "
            f"time taken={timeout:.6f} seconds"
        )
        # nosemgrep: semgrep.rules.pii-positional-arguments-in-logging
        # Error message contains only numeric timeout values, not PII
        structlogger.error(
            f"{self.__class__.__name__.lower()}.llm.timeout", error=error_message
        )
        raise ProviderClientAPIException(asyncio.TimeoutError(error_message)) from None

    def _get_formatted_messages(
        self, messages: Union[List[dict], List[str], str]
    ) -> List[Dict[str, str]]:
        """Returns a list of formatted messages."""
        if (
            isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], dict)
        ):
            # Check if the messages are already formatted. If so, return them as is.
            return cast(List[Dict[str, str]], messages)
        return self._format_messages(messages)

    def _format_messages(self, messages: Union[List[str], str]) -> List[Dict[str, str]]:
        """Formats messages (or a single message) to OpenAI format."""
        if isinstance(messages, str):
            messages = [messages]
        return [{"content": message, "role": ROLE_USER} for message in messages]

    def _format_response(self, response: Any) -> LLMResponse:
        """Parses the LiteLLM response to Rasa format."""
        formatted_response = LLMResponse(
            id=response.id,
            created=response.created,
            choices=[choice.message.content for choice in response.choices],
            model=response.model,
        )
        if (
            response.model_extra
            and (usage := response.model_extra.get("usage")) is not None
        ):
            # We use `.get()` for accessing litellm.utils.Usage attributes.
            # litellm.utils.Usage does not set the attributes if
            # `prompt_tokens` or `completion_tokens` are absent (None).
            prompt_tokens = (
                num_tokens
                if isinstance(num_tokens := usage.get("prompt_tokens", 0), (int, float))
                else 0
            )
            completion_tokens = (
                num_tokens
                if isinstance(
                    num_tokens := usage.get("completion_tokens", 0), (int, float)
                )
                else 0
            )
            formatted_response.usage = LLMUsage(prompt_tokens, completion_tokens)

        # Extract tool calls from all choices
        formatted_response.tool_calls = self._extract_tool_calls(response)

        structlogger.debug(
            "base_litellm_client.formatted_response",
            formatted_response=formatted_response.to_dict(),
        )
        return formatted_response

    def _format_response_stream(self, response: Any) -> LLMResponse:
        """Parses the LiteLLM streaming response chunk to Rasa format."""
        return LLMResponse(
            id=response.id,
            created=response.created,
            choices=[choice.delta.content or "" for choice in response.choices],
            model=response.model,
        )

    def _extract_tool_calls(self, response: Any) -> List[LLMToolCall]:
        """Extract tool calls from response choices.

        Args:
            response: List of response choices from LiteLLM

        Returns:
            List of LLMToolCall objects, empty if no tool calls found
        """
        return [
            LLMToolCall.from_litellm(tool_call)
            for choice in response.choices
            if choice.message.tool_calls
            for tool_call in choice.message.tool_calls
        ]

    def _format_text_completion_response(self, response: Any) -> LLMResponse:
        """Parses the LiteLLM text completion response to Rasa format."""
        formatted_response = LLMResponse(
            id=response.id,
            created=response.created,
            choices=[choice.text for choice in response.choices],
            model=response.model,
        )
        if (usage := response.usage) is not None:
            prompt_tokens = (
                num_tokens
                if isinstance(num_tokens := usage.prompt_tokens, (int, float))
                else 0
            )
            completion_tokens = (
                num_tokens
                if isinstance(num_tokens := usage.completion_tokens, (int, float))
                else 0
            )
            formatted_response.usage = LLMUsage(prompt_tokens, completion_tokens)
        structlogger.debug(
            "base_litellm_client.formatted_response",
            formatted_response=formatted_response.to_dict(),
        )
        return formatted_response

    @staticmethod
    def _ensure_certificates() -> None:
        """Configures SSL certificates for LiteLLM.

        This method is invoked during client initialization.

        LiteLLM may utilize `openai` clients or other providers that require
        SSL verification settings through the `SSL_VERIFY` / `SSL_CERTIFICATE`
        environment variables or the `litellm.ssl_verify` /
        `litellm.ssl_certificate` global settings.

        This method ensures proper SSL configuration for both cases.
        """
        ensure_ssl_certificates_for_litellm_non_openai_based_clients()
        ensure_ssl_certificates_for_litellm_openai_based_clients()
