from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Union

import structlog

from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.providers._configs.litellm_router_client_config import (
    LiteLLMRouterClientConfig,
)
from rasa.shared.providers.constants import LITE_LLM_MODEL_FIELD
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.providers.router._base_litellm_router_client import (
    _BaseLiteLLMRouterClient,
)
from rasa.shared.utils.io import suppress_logs

structlogger = structlog.get_logger()


class LiteLLMRouterLLMClient(_BaseLiteLLMRouterClient, _BaseLiteLLMClient):
    """A client for interfacing with LiteLLM Router LLM endpoints.

    Parameters:
        model_group_id (str): The model group ID.
        model_configurations (List[Dict[str, Any]]): The list of model configurations.
        router_settings (Dict[str, Any]): The router settings.
        kwargs (Optional[Dict[str, Any]]): Additional configuration parameters.

    Raises:
        ProviderClientValidationError: If validation of the client setup fails.
    """

    def __init__(
        self,
        model_group_id: str,
        model_configurations: List[Dict[str, Any]],
        router_settings: Dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(
            model_group_id, model_configurations, router_settings, **kwargs
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> LiteLLMRouterLLMClient:
        """Instantiates a LiteLLM Router LLM client from a configuration dict.

        Args:
            config: (Dict[str, Any]) The configuration dictionary.

        Returns:
            LiteLLMRouterLLMClient: The instantiated LiteLLM Router LLM client.

        Raises:
            ValueError: If the configuration is invalid.
        """
        try:
            client_config = LiteLLMRouterClientConfig.from_dict(config)
        except (KeyError, ValueError) as e:
            message = "Cannot instantiate a client from the passed configuration."
            structlogger.error(
                "litellm_router_llm_client.from_config.error",
                message=message,
                config=config,
                original_error=e,
            )
            raise

        return cls(
            model_group_id=client_config.model_group_id,
            model_configurations=client_config.litellm_model_list,
            router_settings=client_config.litellm_router_settings,
            use_chat_completions_endpoint=client_config.use_chat_completions_endpoint,
            **client_config.extra_parameters,
        )

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
            structlogger.info(
                "litellm_router_llm_client.text_completion",
                _completion_fn_args=self._completion_fn_args,
            )
            response = self.router_client.text_completion(
                prompt=prompt, **self._completion_fn_args
            )
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
                self.router_client.atext_completion(
                    prompt=prompt, **self._completion_fn_args
                ),
                timeout=timeout,
            )
            return self._format_text_completion_response(response)
        except asyncio.TimeoutError:
            self._handle_timeout_error()
        except Exception as e:
            raise ProviderClientAPIException(e)

    @suppress_logs(log_level=logging.WARNING)
    def completion(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> LLMResponse:
        """Synchronously generate completions for given list of messages.

        Method overrides the base class method to call the appropriate
        completion method based on the configuration. If the chat completions
        endpoint is enabled, the completion method is called. Otherwise, the
        text_completion method is called.

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
        if not self._use_chat_completions_endpoint:
            return self._text_completion(messages)
        try:
            formatted_messages = self._get_formatted_messages(messages)
            response = self.router_client.completion(
                messages=formatted_messages, **{**self._completion_fn_args, **kwargs}
            )
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(e)

    @suppress_logs(log_level=logging.WARNING)
    async def acompletion(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> LLMResponse:
        """Asynchronously generate completions for given list of messages.

        Method overrides the base class method to call the appropriate
        completion method based on the configuration. If the chat completions
        endpoint is enabled, the completion method is called. Otherwise, the
        text_completion method is called.

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
        if not self._use_chat_completions_endpoint:
            return await self._atext_completion(messages)
        try:
            formatted_messages = self._get_formatted_messages(messages)
            timeout = self._litellm_extra_parameters.get(
                "timeout", DEFAULT_REQUEST_TIMEOUT
            )
            response = await asyncio.wait_for(
                self.router_client.acompletion(
                    messages=formatted_messages,
                    **{**self._completion_fn_args, **kwargs},
                ),
                timeout=timeout,
            )
            return self._format_response(response)
        except asyncio.TimeoutError:
            self._handle_timeout_error()
        except Exception as e:
            raise ProviderClientAPIException(e)

    @suppress_logs(log_level=logging.WARNING)
    async def acompletion_stream(
        self, messages: Union[List[dict], List[str], str], **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously generate streaming completions for given list of messages.

        Note: Currently falls back to non-streaming completion for LiteLLM Router
        due to complexity in the router's streaming implementation. This provides
        a graceful degradation until streaming support is properly implemented.

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
            An async generator yielding a single completion response chunk.
        """
        # Fall back to non-streaming completion for LiteLLM Router
        structlogger.warning(
            "litellm_router_llm_client.acompletion_stream.fallback",
            event_info="Streaming not yet supported for LiteLLM Router. "
            "Falling back to non-streaming completion.",
        )
        response = await self.acompletion(messages, **kwargs)
        yield response

    @property
    def _completion_fn_args(self) -> Dict[str, Any]:
        """Returns the completion arguments.

        For invoking a call through LiteLLM's completion functions.
        """
        return {
            **self._litellm_extra_parameters,
            LITE_LLM_MODEL_FIELD: self.model_group_id,
        }
