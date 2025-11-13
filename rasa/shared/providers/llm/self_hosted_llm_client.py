from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import structlog
from litellm import atext_completion, text_completion

from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.shared.constants import (
    API_KEY,
    SELF_HOSTED_VLLM_API_KEY_ENV_VAR,
    SELF_HOSTED_VLLM_PREFIX,
)
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.providers._configs.self_hosted_llm_client_config import (
    SelfHostedLLMClientConfig,
)
from rasa.shared.providers.constants import (
    LITE_LLM_API_BASE_FIELD,
    LITE_LLM_API_VERSION_FIELD,
)
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.io import suppress_logs

structlogger = structlog.get_logger()


class SelfHostedLLMClient(_BaseLiteLLMClient):
    """A client for interfacing with Self Hosted LLM endpoints.

    Parameters:
        model (str): The model or deployment name.
        provider (str): The provider of the model.
        api_base (str): The base URL of the API endpoint.
        api_type (Optional[str]): The type of the API endpoint.
        api_version (Optional[str]): The version of the API endpoint.
        use_chat_completions_endpoint (Optional[bool]): Whether to use the chat
            completions endpoint for completions. Defaults to True.
        kwargs: Any: Additional configuration parameters that can include, but
            are not limited to model parameters and lite-llm specific
            parameters. These parameters will be passed to the
            completion/acompletion calls. To see what it can include, visit:

    Raises:
        ProviderClientValidationError: If validation of the client setup fails.
        ProviderClientAPIException: If the API request fails.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_base: str,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
        use_chat_completions_endpoint: Optional[bool] = True,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self._provider = provider
        self._model = model
        self._api_base = api_base
        self._api_type = api_type
        self._api_version = api_version
        self._use_chat_completions_endpoint = use_chat_completions_endpoint
        self._extra_parameters = kwargs or {}
        if self._extra_parameters.get(API_KEY) is None:
            self._apply_dummy_api_key_if_missing()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> SelfHostedLLMClient:
        try:
            client_config = SelfHostedLLMClientConfig.from_dict(config)
        except ValueError as e:
            message = "Cannot instantiate a client from the passed configuration."
            structlogger.error(
                "self_hosted_llm_client.from_config.error",
                message=message,
                config=config,
                original_error=e,
            )
            raise

        return cls(
            model=client_config.model,
            provider=client_config.provider,
            api_base=client_config.api_base,
            api_type=client_config.api_type,
            api_version=client_config.api_version,
            use_chat_completions_endpoint=client_config.use_chat_completions_endpoint,
            **client_config.extra_parameters,
        )

    @property
    def provider(self) -> str:
        """Returns the provider name for the self hosted llm client.

        Returns:
            String representing the provider name.
        """
        return self._provider

    @property
    def model(self) -> str:
        """Returns the model name for the self hosted llm client.

        Returns:
            String representing the model name.
        """
        return self._model

    @property
    def api_base(self) -> str:
        """Returns the base URL for the API endpoint.

        Returns:
            String representing the base URL.
        """
        return self._api_base

    @property
    def api_type(self) -> Optional[str]:
        """Returns the type of the API endpoint. Currently only OpenAI is supported.

        Returns:
            String representing the API type.
        """
        return self._api_type

    @property
    def api_version(self) -> Optional[str]:
        """Returns the version of the API endpoint.

        Returns:
            String representing the API version.
        """
        return self._api_version

    @property
    def config(self) -> Dict:
        """Returns the configuration for the self hosted llm client.

        Returns:
            Dictionary containing the configuration.
        """
        config = SelfHostedLLMClientConfig(
            model=self._model,
            provider=self._provider,
            api_base=self._api_base,
            api_type=self._api_type,
            api_version=self._api_version,
            use_chat_completions_endpoint=self._use_chat_completions_endpoint,
            extra_parameters=self._extra_parameters,
        )
        return config.to_dict()

    @property
    def _litellm_model_name(self) -> str:
        """Returns the value of LiteLLM's model parameter.

        To be used in completion/acompletion in LiteLLM format:
        <hosted_vllm>/<model or deployment name>
        """
        if self.model and f"{SELF_HOSTED_VLLM_PREFIX}/" not in self.model:
            return f"{SELF_HOSTED_VLLM_PREFIX}/{self.model}"
        return self.model

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        """Returns optional configuration parameters.

        Specific to the client provider and deployed model.
        """
        return self._extra_parameters

    @property
    def _completion_fn_args(self) -> Dict[str, Any]:
        """Returns the completion arguments.

        For invoking a call through LiteLLM's completion functions.
        """
        fn_args = super()._completion_fn_args
        fn_args.update(
            {
                LITE_LLM_API_BASE_FIELD: self.api_base,
                LITE_LLM_API_VERSION_FIELD: self.api_version,
            }
        )
        return fn_args

    @suppress_logs(log_level=logging.WARNING)
    def _text_completion(self, prompt: Union[List[str], str]) -> LLMResponse:
        """Synchronously generate completions for given prompt.

        Args:
            prompt: Prompt to generate the completion for.

        Returns:
            List of message completions.

        Raises:
            ProviderClientAPIException: If the API request fails.
        """
        try:
            response = text_completion(prompt=prompt, **self._completion_fn_args)
            return self._format_text_completion_response(response)
        except Exception as e:
            raise ProviderClientAPIException(e)

    @suppress_logs(log_level=logging.WARNING)
    async def _atext_completion(self, prompt: Union[List[str], str]) -> LLMResponse:
        """Asynchronously generate completions for given prompt.

        Args:
            prompt: Prompt to generate the completion for.

        Returns:
            List of message completions.

        Raises:
            ProviderClientAPIException: If the API request fails.
        """
        try:
            timeout = self._litellm_extra_parameters.get(
                "timeout", DEFAULT_REQUEST_TIMEOUT
            )
            response = await asyncio.wait_for(
                atext_completion(prompt=prompt, **self._completion_fn_args),
                timeout=timeout,
            )
            return self._format_text_completion_response(response)
        except asyncio.TimeoutError:
            self._handle_timeout_error()
        except Exception as e:
            raise ProviderClientAPIException(e)

    async def acompletion(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> LLMResponse:
        """Asynchronous completion of the model with the given messages.

        Method overrides the base class method to call the appropriate
        completion method based on the configuration. If the chat completions
        endpoint is enabled, the acompletion method is called. Otherwise, the
        atext_completion method is called.

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
            The completion response.
        """
        if self._use_chat_completions_endpoint:
            return await super().acompletion(messages, **kwargs)
        return await self._atext_completion(messages)

    async def acompletion_stream(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronous streaming completion of the model with the given messages.

        Method overrides the base class method to call the appropriate
        completion method based on the configuration. If the chat completions
        endpoint is enabled, the acompletion_stream method is called. Otherwise,
        falls back to non-streaming completion for text completions endpoint.

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
            An async generator yielding completion response chunks.
            For text completions endpoint, yields a single chunk with the full response.
        """
        if self._use_chat_completions_endpoint:
            async for chunk in super().acompletion_stream(messages, **kwargs):
                yield chunk
        else:
            # Fall back to non-streaming completion for text completions endpoint
            structlogger.warning(
                "self_hosted_llm_client.acompletion_stream.fallback",
                event_info="Streaming not supported for text completions endpoint. "
                "Falling back to non-streaming completion.",
            )
            response = await self._atext_completion(messages)
            yield response

    def completion(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> LLMResponse:
        """Completion of the model with the given messages.

        Method overrides the base class method to call the appropriate
        completion method based on the configuration. If the chat completions
        endpoint is enabled, the completion method is called. Otherwise, the
        text_completion method is called.

        Args:
            messages: The messages to be used for completion.
            **kwargs: Additional parameters to pass to the completion call.

        Returns:
            The completion response.
        """
        if self._use_chat_completions_endpoint:
            return super().completion(messages, **kwargs)
        return self._text_completion(messages)

    @staticmethod
    def _apply_dummy_api_key_if_missing() -> None:
        if not os.getenv(SELF_HOSTED_VLLM_API_KEY_ENV_VAR):
            os.environ[SELF_HOSTED_VLLM_API_KEY_ENV_VAR] = (
                "dummy_self_hosted_llm_api_key"
            )
