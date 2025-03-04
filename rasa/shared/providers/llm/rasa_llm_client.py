from __future__ import annotations

from typing import Any, Dict, Optional

import structlog

from rasa.shared.constants import (
    OPENAI_PROVIDER,
    RASA_PROVIDER,
)
from rasa.shared.providers._configs.rasa_llm_client_config import (
    RasaLLMClientConfig,
)
from rasa.shared.providers.constants import (
    LITE_LLM_API_BASE_FIELD,
    LITE_LLM_API_KEY_FIELD,
)
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient
from rasa.utils.licensing import retrieve_license_from_env

structlogger = structlog.get_logger()


class RasaLLMClient(_BaseLiteLLMClient):
    """A client for interfacing with a Rasa-Hosted LLM endpoint that uses.

    Parameters:
        model (str): The model or deployment name.
        api_base (str): The base URL of the API endpoint.
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
        model: str,
        api_base: str,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self._model = model
        self._api_base = api_base
        self._use_chat_completions_endpoint = True
        self._extra_parameters = kwargs or {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_base(self) -> Optional[str]:
        """Returns the base API URL for the openai llm client."""
        return self._api_base

    @property
    def provider(self) -> str:
        """Returns the provider name for the self hosted llm client.

        Returns:
            String representing the provider name.
        """
        return RASA_PROVIDER

    @property
    def _litellm_model_name(self) -> str:
        return f"{OPENAI_PROVIDER}/{self._model}"

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        return self._extra_parameters

    @property
    def config(self) -> dict:
        return RasaLLMClientConfig(
            model=self._model,
            api_base=self._api_base,
            extra_parameters=self._extra_parameters,
        ).to_dict()

    @property
    def _completion_fn_args(self) -> Dict[str, Any]:
        """Returns the completion arguments for invoking a call using completions."""
        fn_args = super()._completion_fn_args
        fn_args.update(
            {
                LITE_LLM_API_BASE_FIELD: self.api_base,
                LITE_LLM_API_KEY_FIELD: retrieve_license_from_env(),
            }
        )
        return fn_args

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> RasaLLMClient:
        try:
            client_config = RasaLLMClientConfig.from_dict(config)
        except ValueError as e:
            message = "Cannot instantiate a client from the passed configuration."
            structlogger.error(
                "rasa_llm_client.from_config.error",
                message=message,
                config=config,
                original_error=e,
            )
            raise
        return cls(
            model=client_config.model,
            api_base=client_config.api_base,
            **client_config.extra_parameters,
        )
