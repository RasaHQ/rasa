from abc import abstractmethod
from typing import Dict, List, Any, Union

import logging
import structlog
from litellm import (
    completion,
    acompletion,
    validate_environment,
)

from rasa.shared.exceptions import (
    ProviderClientAPIException,
    ProviderClientValidationError,
)
from rasa.shared.providers._ssl_verification_utils import (
    ensure_ssl_certificates_for_litellm_non_openai_based_clients,
    ensure_ssl_certificates_for_litellm_openai_based_clients,
)
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMUsage
from rasa.shared.utils.io import suppress_logs

structlogger = structlog.get_logger()

_VALIDATE_ENVIRONMENT_MISSING_KEYS_KEY = "missing_keys"

# Suppress LiteLLM info and debug logs - Global level.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class _BaseLiteLLMClient:
    """
    An abstract base class for LiteLLM clients.

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
    def from_config(cls, config: Dict[str, Any]) -> "_BaseLiteLLMClient":
        pass

    @property
    @abstractmethod
    def config(self) -> dict:
        """Returns the configuration for that the llm client
        in dictionary form.
        """
        pass

    @property
    @abstractmethod
    def _litellm_model_name(self) -> str:
        """Returns the value of LiteLLM's model parameter to be used in
        completion/acompletion in LiteLLM format:

        <provider>/<model or deployment name>
        """
        pass

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        """Returns a dictionary of extra parameters which include model
        parameters as well as LiteLLM specific input parameters.

        By default, this returns an empty dictionary (no extra parameters).
        """
        return {}

    @property
    def _completion_fn_args(self) -> dict:
        return {
            **self._litellm_extra_parameters,
            "model": self._litellm_model_name,
            # Since all providers covered by LiteLLM use the OpenAI format, but
            # not all support every OpenAI parameter, raise an exception if
            # provider/model uses unsupported parameter
            "drop_params": False,
        }

    def validate_client_setup(self) -> None:
        """Perform client validation. By default only environment variables
        are validated.

        Raises:
            ProviderClientValidationError if validation fails.
        """
        self._validate_environment_variables()
        self._validate_api_key_not_in_config()

    def _validate_environment_variables(self) -> None:
        """Validate that the required environment variables are set."""
        validation_info = validate_environment(self._litellm_model_name)
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

    def _validate_api_key_not_in_config(self) -> None:
        if "api_key" in self._litellm_extra_parameters:
            event_info = (
                "API Key is set through `api_key` extra parameter."
                "Set API keys through environment variables."
            )
            structlogger.error(
                "base_litellm_client.validate_api_key_not_in_config",
                event_info=event_info,
            )
            raise ProviderClientValidationError(event_info)

    @suppress_logs(log_level=logging.WARNING)
    def completion(self, messages: Union[List[str], str]) -> LLMResponse:
        """
        Synchronously generate completions for given list of messages.

        Args:
            messages: List of messages or a single message to generate the
                completion for.
        Returns:
            List of message completions.
        Raises:
            ProviderClientAPIException: If the API request fails.
        """
        try:
            formatted_messages = self._format_messages(messages)
            response = completion(
                messages=formatted_messages, **self._completion_fn_args
            )
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(e)

    @suppress_logs(log_level=logging.WARNING)
    async def acompletion(self, messages: Union[List[str], str]) -> LLMResponse:
        """
        Asynchronously generate completions for given list of messages.

        Args:
            messages: List of messages or a single message to generate the
                completion for.
        Returns:
            List of message completions.
        Raises:
            ProviderClientAPIException: If the API request fails.
        """
        try:
            formatted_messages = self._format_messages(messages)
            response = await acompletion(
                messages=formatted_messages, **self._completion_fn_args
            )
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(e)

    def _format_messages(self, messages: Union[List[str], str]) -> List[Dict[str, str]]:
        """Formats messages (or a single message) to OpenAI format."""
        if isinstance(messages, str):
            messages = [messages]
        return [{"content": message, "role": "user"} for message in messages]

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
        structlogger.debug(
            "base_litellm_client.formatted_response",
            formatted_response=formatted_response.to_dict(),
        )
        return formatted_response

    @staticmethod
    def _ensure_certificates() -> None:
        """
        Configures SSL certificates for LiteLLM. This method is invoked during
        client initialization.

        LiteLLM may utilize `openai` clients or other providers that require
        SSL verification settings through the `SSL_VERIFY` / `SSL_CERTIFICATE`
        environment variables or the `litellm.ssl_verify` /
        `litellm.ssl_certificate` global settings.

        This method ensures proper SSL configuration for both cases.
        """
        ensure_ssl_certificates_for_litellm_non_openai_based_clients()
        ensure_ssl_certificates_for_litellm_openai_based_clients()
