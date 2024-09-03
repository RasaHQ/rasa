from typing import Any, Dict, Optional
import structlog

from rasa.shared.constants import OPENAI_PROVIDER
from rasa.shared.providers._configs.self_hosted_llm_client_config import (
    SelfHostedLLMClientConfig,
)
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient

structlogger = structlog.get_logger()


class SelfHostedLLMClient(_BaseLiteLLMClient):
    """A client for interfacing with Self Hosted LLM endpoints that uses

    Parameters:
        model (str): The model or deployment name.
        provider (str): The provider of the model.
        api_base (str): The base URL of the API endpoint.
        api_type (Optional[str]): The type of the API endpoint.
        api_version (Optional[str]): The version of the API endpoint.
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
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self._provider = provider
        self._model = model
        self._api_base = api_base
        self._api_type = api_type
        self._api_version = api_version
        self._extra_parameters = kwargs or {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SelfHostedLLMClient":
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
            **client_config.extra_parameters,
        )

    @property
    def provider(self) -> str:
        """
        Returns the provider name for the self hosted llm client.

        Returns:
            String representing the provider name.
        """
        return self._provider

    @property
    def model(self) -> str:
        """
        Returns the model name for the self hosted llm client.

        Returns:
            String representing the model name.
        """
        return self._model

    @property
    def api_base(self) -> str:
        """
        Returns the base URL for the API endpoint.

        Returns:
            String representing the base URL.
        """
        return self._api_base

    @property
    def api_type(self) -> Optional[str]:
        """
        Returns the type of the API endpoint. Currently only OpenAI is supported.

        Returns:
            String representing the API type.
        """
        return self._api_type

    @property
    def api_version(self) -> Optional[str]:
        """
        Returns the version of the API endpoint.

        Returns:
            String representing the API version.
        """
        return self._api_version

    @property
    def config(self) -> Dict:
        """
        Returns the configuration for the self hosted llm client.
        Returns:
            Dictionary containing the configuration.
        """
        config = SelfHostedLLMClientConfig(
            model=self._model,
            provider=self._provider,
            api_base=self._api_base,
            api_type=self._api_type,
            api_version=self._api_version,
            extra_parameters=self._extra_parameters,
        )
        return config.to_dict()

    @property
    def _litellm_model_name(self) -> str:
        """Returns the value of LiteLLM's model parameter to be used in
        completion/acompletion in LiteLLM format:

        <openai>/<model or deployment name>
        """
        if self.model and f"{OPENAI_PROVIDER}/" not in self.model:
            return f"{OPENAI_PROVIDER}/{self.model}"
        return self.model

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        """Returns optional configuration parameters specific
        to the client provider and deployed model.
        """
        return self._extra_parameters

    @property
    def _completion_fn_args(self) -> Dict[str, Any]:
        """Returns the completion arguments for invoking a call through
        LiteLLM's completion functions.
        """
        fn_args = super()._completion_fn_args
        fn_args.update(
            {
                "api_base": self.api_base,
                "api_version": self.api_version,
            }
        )
        return fn_args
