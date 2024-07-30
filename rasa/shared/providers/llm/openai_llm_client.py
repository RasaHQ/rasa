import os
import re
from typing import Dict, Any, Optional

from rasa.shared.constants import (
    OPENAI_API_BASE_ENV_VAR,
    OPENAI_API_VERSION_ENV_VAR,
    OPENAI_API_TYPE_ENV_VAR,
)
from rasa.shared.providers._configs.openai_client_config import OpenAIClientConfig
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient

_OPENAI_PROVIDER = "openai"


class OpenAILLMClient(_BaseLiteLLMClient):
    """
    A client for interfacing with OpenAI LLMs.

    Parameters:
        model (str): The OpenAI model name.
        api_base (Optional[str]): Optional, the base URL for the API endpoints.
            If not provided, it will try to be set via environment variables.
        api_version (Optional[str]): Optional, the version of the API to use.
            If not provided, it will try to be set via environment variable.
        api_type: (Optional[str]): The api type. If not provided, it will be set via
            environment variable (also optional).
        kwargs (Optional[Dict[str, Any]]): Optional configuration parameters specific
            to the model.

    Raises:
        ProviderClientValidationError: If validation of the client setup fails.
    """

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        api_type: Optional[str] = None,
        **kwargs: Any,
    ):
        self._model = model
        self._api_base = api_base or os.environ.get(OPENAI_API_BASE_ENV_VAR, None)
        self._api_version = api_version or os.environ.get(
            OPENAI_API_VERSION_ENV_VAR, None
        )

        # Not used by LiteLLM, here for backward compatibility
        self._api_type = api_type or os.environ.get(OPENAI_API_TYPE_ENV_VAR)

        self._extra_parameters = kwargs or {}
        self.validate_client_setup()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OpenAILLMClient":
        openai_config = OpenAIClientConfig.from_dict(config)
        return cls(
            openai_config.model,
            openai_config.api_base,
            openai_config.api_version,
            openai_config.api_type,
            **openai_config.extra_parameters,
        )

    @property
    def config(self) -> dict:
        config = OpenAIClientConfig(
            self.model,
            self.api_base,
            self.api_version,
            self._litellm_extra_parameters,
            self.api_type,
        )
        return config.to_dict()

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_base(self) -> Optional[str]:
        """
        Returns the base API URL for the openai llm client.
        """
        return self._api_base

    @property
    def api_version(self) -> Optional[str]:
        """
        Returns the API version for the OpenAI LLM client.
        """
        return self._api_version

    @property
    def api_type(self) -> Optional[str]:
        return self._api_type

    @property
    def _litellm_model_name(self) -> str:
        """Returns the value of LiteLLM's model parameter to be used in
        completion/acompletion in LiteLLM format:

        <provider>/<model or deployment name>
        """
        regex_patter = rf"^{_OPENAI_PROVIDER}/"
        if not re.match(regex_patter, self._model):
            return f"{_OPENAI_PROVIDER}/{self._model}"
        return self._model

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        return self._extra_parameters
