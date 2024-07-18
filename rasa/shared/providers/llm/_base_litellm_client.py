from abc import abstractmethod
from typing import Dict, List, Any, Union

import structlog
from litellm import completion, acompletion, validate_environment

from rasa.shared.exceptions import (
    ProviderClientAPIException,
    ProviderClientValidationError,
)
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMUsage

structlogger = structlog.get_logger()

_VALIDATE_ENVIRONMENT_MISSING_KEYS_KEY = "missing_keys"


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
    def model(self) -> str:
        """Returns the model or deployment name."""
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def model_parameters(self) -> Dict[str, Any]:
        pass

    @property
    def _completion_fn_args(self) -> dict:
        return {
            "model": f"{self.provider}/{self.model}",
            # Since all providers covered by LiteLLM use the OpenAI format, but
            # not all support every OpenAI parameter, drop_params ensures that
            # unsupported request parameters are removed, preventing API
            # exceptions.
            "drop_params": True,
            **self.model_parameters,
        }

    def validate_client_setup(self) -> None:
        """Perform client validation. By default only environment variables
        are validated.

        Raises:
            ProviderClientValidationError if validation fails.
        """
        self._validate_environment_variables()

    def _validate_environment_variables(self) -> None:
        """Validate that the required environment variables are set."""
        validation_info = validate_environment(f"{self.provider}/{self.model}")
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

    def completion(self, messages: Union[List[str], str]) -> LLMResponse:
        """
        Synchronously generate completions for given list of messages.

        Args:
            messages: List of messages or a single message to generate the
                completion for.
        Returns:
            List of message completions.
        """
        try:
            formatted_messages = self._format_messages(messages)
            response = completion(
                messages=formatted_messages, **self._completion_fn_args
            )
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(e)

    async def acompletion(self, messages: Union[List[str], str]) -> LLMResponse:
        """
        Asynchronously generate completions for given list of messages.

        Args:
            messages: List of messages or a single message to generate the
                completion for.
        Returns:
            List of message completions.
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
            formatted_response.usage = LLMUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        structlogger.debug(
            "base_litellm_client.formatted_response",
            formatted_response=formatted_response.to_dict(),
        )
        return formatted_response
