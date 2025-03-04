from __future__ import annotations

import os
from typing import Any, Dict, List

import structlog
from litellm import Router

from rasa.shared.constants import (
    API_KEY,
    LITELLM_PARAMS_KEY,
    MODEL_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    MODEL_LIST_KEY,
    ROUTER_CONFIG_KEY,
    SELF_HOSTED_VLLM_API_KEY_ENV_VAR,
    SELF_HOSTED_VLLM_PREFIX,
    USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers._configs.litellm_router_client_config import (
    LiteLLMRouterClientConfig,
)
from rasa.shared.utils.io import resolve_environment_variables

structlogger = structlog.get_logger()


class _BaseLiteLLMRouterClient:
    """An abstract base class for LiteLLM Router clients.

    This class defines the interface and common functionality for all the router clients
    based on LiteLLM.

    The class is made private to prevent it from being part of the public-facing
    interface, as it serves as an internal base class for specific implementations of
    router clients that are based on LiteLLM router implementation.

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
        use_chat_completions_endpoint: bool = True,
        **kwargs: Any,
    ):
        self._model_group_id = model_group_id
        self._model_configurations = model_configurations
        self._router_settings = router_settings
        self._use_chat_completions_endpoint = use_chat_completions_endpoint
        self._extra_parameters = kwargs or {}
        self.additional_client_setup()
        try:
            resolved_model_configurations = (
                self._resolve_env_vars_in_model_configurations()
            )
            self._router_client = Router(
                model_list=resolved_model_configurations, **router_settings
            )
        except Exception as e:
            event_info = "Cannot instantiate a router client."
            structlogger.error(
                "_base_litellm_router_client.init.error",
                event_info=event_info,
                model_group_id=model_group_id,
                models=model_configurations,
                router=router_settings,
                original_error=e,
            )
            raise ProviderClientValidationError(f"{event_info} Original error: {e}")

    def additional_client_setup(self) -> None:
        """Additional setup for the LiteLLM Router client."""
        # If the model configuration is self-hosted VLLM, set a dummy API key if not
        # provided. A bug in the LiteLLM library requires an API key to be set even if
        # it is not required.
        for model_configuration in self.model_configurations:
            if (
                f"{SELF_HOSTED_VLLM_PREFIX}/"
                in model_configuration[LITELLM_PARAMS_KEY][MODEL_CONFIG_KEY]
                and API_KEY not in model_configuration[LITELLM_PARAMS_KEY]
                and not os.getenv(SELF_HOSTED_VLLM_API_KEY_ENV_VAR)
            ):
                os.environ[SELF_HOSTED_VLLM_API_KEY_ENV_VAR] = "dummy api key"
                return

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> _BaseLiteLLMRouterClient:
        """Instantiates a LiteLLM Router Embedding client from a configuration dict.

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

    @property
    def model_group_id(self) -> str:
        """Returns the model group ID for the LiteLLM Router client."""
        return self._model_group_id

    @property
    def model_configurations(self) -> List[Dict[str, Any]]:
        """Returns the model configurations for the LiteLLM Router client."""
        return self._model_configurations

    @property
    def router_settings(self) -> Dict[str, Any]:
        """Returns the router settings for the LiteLLM Router client."""
        return self._router_settings

    @property
    def router_client(self) -> Router:
        """Returns the instantiated LiteLLM Router client."""
        return self._router_client

    @property
    def use_chat_completions_endpoint(self) -> bool:
        """Returns whether to use the chat completions endpoint."""
        return self._use_chat_completions_endpoint

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        """
        Returns the extra parameters for the LiteLLM Router client.

        Returns:
            Dictionary containing the model parameters.
        """
        return self._extra_parameters

    @property
    def config(self) -> Dict:
        """Returns the configuration for the LiteLLM Router client in LiteLLM format."""
        return {
            MODEL_GROUP_ID_CONFIG_KEY: self.model_group_id,
            MODEL_LIST_KEY: self.model_configurations,
            ROUTER_CONFIG_KEY: self.router_settings,
            USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY: (
                self.use_chat_completions_endpoint
            ),
            **self._litellm_extra_parameters,
        }

    def _resolve_env_vars_in_model_configurations(self) -> List:
        model_configuration_with_resolved_keys = []
        for model_configuration in self.model_configurations:
            resolved_model_configuration = resolve_environment_variables(
                model_configuration
            )
            model_configuration_with_resolved_keys.append(resolved_model_configuration)
        return model_configuration_with_resolved_keys
