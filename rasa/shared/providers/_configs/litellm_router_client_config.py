from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import structlog

from rasa.shared.constants import (
    API_TYPE_CONFIG_KEY,
    DEPLOYMENT_CONFIG_KEY,
    LITELLM_PARAMS_KEY,
    MODEL_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    MODEL_LIST_KEY,
    MODEL_NAME_CONFIG_KEY,
    MODELS_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    ROUTER_CONFIG_KEY,
    USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY,
)
from rasa.shared.providers._configs.model_group_config import (
    ModelConfig,
    ModelGroupConfig,
)
from rasa.shared.providers.mappings import get_prefix_from_provider
from rasa.shared.utils.llm import DEPLOYMENT_CENTRIC_PROVIDERS

structlogger = structlog.get_logger()

_LITELLM_UNSUPPORTED_KEYS = [
    PROVIDER_CONFIG_KEY,
    DEPLOYMENT_CONFIG_KEY,
    API_TYPE_CONFIG_KEY,
    USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY,
]


@dataclass
class LiteLLMRouterClientConfig:
    """Parses configuration for a LiteLLM Router client.

    The configuration is expected to be in the following format:

    {
        "id": "model_group_id",
        "models": [
            {
                "provider": "provider_name",
                "model": "model_name",
                "api_base": "api_base",
                "api_key": "api_key",
                "api_version": "api_version",
            },
            {
                "provider": "provider_name",
                "model": "model_name",
            },
        "router": {}
    }

    This configuration is converted into the LiteLLM required format:

    {
        "id": "model_group_id",
        "model_list": [
            {
                "model_name": "model_group_id",
                "litellm_params": {
                    "model": "provider_name/model_name",
                    "api_base": "api_base",
                    "api_key": "api_key",
                    "api_version": "api_version",
                },
            },
            {
                "model_name": "model_group_id",
                "litellm_params": {
                    "model": "provider_name/model_name",
                },
            },
        ],
        "router": {},
    }

    Raises:
        ValueError: If the configuration is missing required keys.
    """

    _model_group_config: ModelGroupConfig
    router: Dict[str, Any]
    _use_chat_completions_endpoint: bool = True
    extra_parameters: dict = field(default_factory=dict)

    @property
    def model_group_id(self) -> str:
        return self._model_group_config.model_group_id

    @property
    def models(self) -> List[ModelConfig]:
        return self._model_group_config.models

    @property
    def litellm_model_list(self) -> List[Dict[str, Any]]:
        return self._convert_models_to_litellm_model_list()

    @property
    def litellm_router_settings(self) -> Dict[str, Any]:
        return self._convert_router_to_litellm_router_settings()

    @property
    def use_chat_completions_endpoint(self) -> bool:
        return self._use_chat_completions_endpoint

    def __post_init__(self) -> None:
        if not self.router:
            message = "Router cannot be empty."
            structlogger.error(
                "litellm_router_client_config.validation_error",
                message=message,
                model_group_id=self._model_group_config.model_group_id,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> LiteLLMRouterClientConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            LiteLLMRouterClientConfig
        """
        model_group_config = ModelGroupConfig.from_dict(config)

        # Copy config to avoid mutating the original
        config_copy = copy.deepcopy(config)
        # Pop the keys used by ModelGroupConfig
        config_copy.pop(MODEL_GROUP_ID_CONFIG_KEY, None)
        config_copy.pop(MODELS_CONFIG_KEY, None)
        # Get the router settings
        router_settings = config_copy.pop(ROUTER_CONFIG_KEY, {})
        # Get the use_chat_completions_endpoint setting
        use_chat_completions_endpoint = router_settings.get(
            USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY, True
        )
        # The rest is considered as extra parameters
        extra_parameters = config_copy

        this = LiteLLMRouterClientConfig(
            _model_group_config=model_group_config,
            router=router_settings,
            _use_chat_completions_endpoint=use_chat_completions_endpoint,
            extra_parameters=extra_parameters,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        d = self._model_group_config.to_dict()
        d[ROUTER_CONFIG_KEY] = self.router
        if self.extra_parameters:
            d.update(self.extra_parameters)
        return d

    def to_litellm_dict(self) -> dict:
        return {
            **self.extra_parameters,
            MODEL_GROUP_ID_CONFIG_KEY: self.model_group_id,
            MODEL_LIST_KEY: self._convert_models_to_litellm_model_list(),
            ROUTER_CONFIG_KEY: self._convert_router_to_litellm_router_settings(),
        }

    def _convert_router_to_litellm_router_settings(self) -> Dict[str, Any]:
        _router_settings_copy = copy.deepcopy(self.router)
        _router_settings_copy.pop(USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY, None)
        return _router_settings_copy

    def _convert_models_to_litellm_model_list(self) -> List[Dict[str, Any]]:
        litellm_model_list = []

        for model_config_object in self.models:
            # Convert the model config to a dict representation
            litellm_model_config = model_config_object.to_dict()

            provider = litellm_model_config[PROVIDER_CONFIG_KEY]

            # Get the litellm prefixing for the provider
            prefix = get_prefix_from_provider(provider)

            # Determine whether to use model or deployment key based on the provider.
            litellm_model_name = (
                litellm_model_config[DEPLOYMENT_CONFIG_KEY]
                if provider in DEPLOYMENT_CENTRIC_PROVIDERS
                else litellm_model_config[MODEL_CONFIG_KEY]
            )

            # Set 'model' to a provider prefixed model name e.g. openai/gpt-4
            litellm_model_config[MODEL_CONFIG_KEY] = (
                litellm_model_name
                if f"{prefix}/" in litellm_model_name
                else f"{prefix}/{litellm_model_name}"
            )

            # Remove parameters that are None and not supported by LiteLLM.
            litellm_model_config = {
                key: value
                for key, value in litellm_model_config.items()
                if key not in _LITELLM_UNSUPPORTED_KEYS and value is not None
            }

            litellm_model_list_item = {
                MODEL_NAME_CONFIG_KEY: self.model_group_id,
                LITELLM_PARAMS_KEY: litellm_model_config,
            }

            litellm_model_list.append(litellm_model_list_item)

        return litellm_model_list
