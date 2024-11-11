from typing import Any, Dict, List, Optional, Union

import structlog
import os

from rasa.shared.constants import OPENAI_PROVIDER, OPENAI_API_KEY_ENV_VAR
from rasa.shared.providers._configs.rasa_llm_client_config import (
    RasaLLMClientConfig,
)
from rasa.utils.licensing import retrieve_license_from_env
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.providers.llm.self_hosted_llm_client import SelfHostedLLMClient
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMUsage
from rasa.shared.utils.io import suppress_logs

structlogger = structlog.get_logger()


class RasaLLMClient(SelfHostedLLMClient):
    """A client for interfacing with a Rasa-Hosted LLM endpoint that uses

    Parameters:
        model (str): The model or deployment name.
        provider (str): The provider of the model.
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
        super().__init__(
            provider=provider,
            model=model,
            api_base=api_base,
            api_type=api_type,
            api_version=api_version,
            use_chat_completions_endpoint=use_chat_completions_endpoint,
            **kwargs
        )

    @classmethod
    def set_rasa_pro_license_as_openai_api_key(cls):
        os.environ[OPENAI_API_KEY_ENV_VAR] = retrieve_license_from_env()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RasaLLMClient":
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
        cls.set_rasa_pro_license_as_openai_api_key()
        return cls(
            model=client_config.model,
            provider="self-hosted",
            api_base=client_config.api_base,
            api_type=client_config.api_type,
            api_version=client_config.api_version,
            use_chat_completions_endpoint=client_config.use_chat_completions_endpoint,
            **client_config.extra_parameters,
        )    