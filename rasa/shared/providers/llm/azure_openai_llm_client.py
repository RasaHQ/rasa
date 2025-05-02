from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import structlog

from rasa.shared.constants import (
    API_KEY,
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_KEY_ENV_VAR,
    AZURE_API_TYPE_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    AZURE_OPENAI_PROVIDER,
    OPENAI_API_BASE_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_API_TYPE_ENV_VAR,
    OPENAI_API_VERSION_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers._configs.azure_openai_client_config import (
    AzureEntraIDOAuthConfig,
    AzureOpenAIClientConfig,
)
from rasa.shared.providers._utils import validate_azure_client_setup
from rasa.shared.providers.constants import (
    DEFAULT_AZURE_API_KEY_NAME,
    LITE_LLM_API_BASE_FIELD,
    LITE_LLM_API_KEY_FIELD,
    LITE_LLM_API_VERSION_FIELD,
    LITE_LLM_AZURE_AD_TOKEN,
)
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient
from rasa.shared.utils.io import raise_deprecation_warning

structlogger = structlog.get_logger()

AZURE_CLIENT_ID = "AZURE_CLIENT_ID"
AZURE_CLIENT_SECRET = "AZURE_CLIENT_SECRET"
AZURE_TENANT_ID = "AZURE_TENANT_ID"
CLIENT_SECRET_VARS = (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)

AZURE_CLIENT_CERTIFICATE_PATH = "AZURE_CLIENT_CERTIFICATE_PATH"
AZURE_CLIENT_CERTIFICATE_PASSWORD = "AZURE_CLIENT_CERTIFICATE_PASSWORD"
AZURE_CLIENT_SEND_CERTIFICATE_CHAIN = "AZURE_CLIENT_SEND_CERTIFICATE_CHAIN"
CERT_VARS = (AZURE_CLIENT_ID, AZURE_CLIENT_CERTIFICATE_PATH, AZURE_TENANT_ID)


class AzureADConfig:
    def __init__(
        self, client_id: str, client_secret: str, tenant_id: str, scopes: str
    ) -> None:
        self.scopes = scopes


class AzureOpenAILLMClient(_BaseLiteLLMClient):
    """A client for interfacing with Azure's OpenAI LLM deployments.

    Parameters:
        deployment (str): The deployment name.
        model (Optional[str]): The name of the deployed model.
        api_type: (Optional[str]): The api type. If not provided, it will be set via
            environment variable.
        api_base (Optional[str]): The base URL for the API endpoints. If not provided,
            it will be set via environment variables.
        api_version (Optional[str]): The version of the API to use. If not provided,
            it will be set via environment variable.

        kwargs (Optional[Dict[str, Any]]): Optional configuration parameters specific
            to the model deployment.

    Raises:
        ProviderClientValidationError: If validation of the client setup fails.
        DeprecationWarning: If deprecated environment variables are used for
            configuration.
    """

    def __init__(
        self,
        deployment: str,
        model: Optional[str] = None,
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        oauth: Optional[AzureEntraIDOAuthConfig] = None,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self._deployment = deployment
        self._model = model
        self._extra_parameters = kwargs or {}

        # Set api_base with the following priority:
        # parameter -> Azure Env Var -> (deprecated) OpenAI Env Var
        self._api_base = (
            api_base
            or os.getenv(AZURE_API_BASE_ENV_VAR)
            or os.getenv(OPENAI_API_BASE_ENV_VAR)
        )

        # Set api_version with the following priority:
        # parameter -> Azure Env Var -> (deprecated) OpenAI Env Var
        self._api_version = (
            api_version
            or os.getenv(AZURE_API_VERSION_ENV_VAR)
            or os.getenv(OPENAI_API_VERSION_ENV_VAR)
        )

        # Not used by LiteLLM, here for backward compatibility
        self._api_type = (
            api_type
            or os.getenv(AZURE_API_TYPE_ENV_VAR)
            or os.getenv(OPENAI_API_TYPE_ENV_VAR)
        )

        os.unsetenv("OPENAI_API_KEY")
        os.unsetenv("AZURE_API_KEY")

        self._oauth = oauth

        if self._oauth:
            os.unsetenv(DEFAULT_AZURE_API_KEY_NAME)
            os.unsetenv(AZURE_API_KEY_ENV_VAR)
            os.unsetenv(OPENAI_API_KEY_ENV_VAR)
        self._api_key_env_var = (
            self._resolve_api_key_env_var() if not self._oauth else None
        )

        # Run helper function to check and raise deprecation warning if
        # deprecated environment variables were used for initialization of the
        # client settings
        self._raise_evn_var_deprecation_warnings()

        # validate the client settings
        self.validate_client_setup()

    def _raise_evn_var_deprecation_warnings(self) -> None:
        """Helper function to check and raise deprecation warning if
        deprecated environment variables were used for initialization of
        some client settings.
        """
        deprecation_mapping = {
            "API Base": {
                "current_value": self.api_base,
                "env_var": AZURE_API_BASE_ENV_VAR,
                "deprecated_var": OPENAI_API_BASE_ENV_VAR,
            },
            "API Version": {
                "current_value": self.api_version,
                "env_var": AZURE_API_VERSION_ENV_VAR,
                "deprecated_var": OPENAI_API_VERSION_ENV_VAR,
            },
        }

        deprecation_warning_message = (
            "Usage of {deprecated_env_var} environment "
            "variable for setting the {setting} for Azure "
            "OpenAI client is deprecated and will be removed "
            "in 4.0.0. "
        )
        deprecation_warning_replacement_message = (
            "Please use {env_var} environment variable."
        )

        for setting in deprecation_mapping.keys():
            current_value = deprecation_mapping[setting]["current_value"]
            env_var = deprecation_mapping[setting]["env_var"]
            deprecated_var = deprecation_mapping[setting]["deprecated_var"]

            # Value is set through the non-deprecated env var
            if current_value == os.getenv(env_var):
                continue

            # Value is set through the deprecated env var
            if current_value == os.getenv(deprecated_var):
                message = deprecation_warning_message.format(
                    setting=setting, deprecated_env_var=deprecated_var
                )
                if env_var is not None:
                    message += deprecation_warning_replacement_message.format(
                        env_var=env_var
                    )
                raise_deprecation_warning(message=message)

    def _resolve_api_key_env_var(self) -> str:
        """Resolves the environment variable to use for the API key.

        Returns:
            str: The env variable in dollar syntax format to use for the API key.
        """
        if API_KEY in self._extra_parameters:
            # API key is set to an env var in the config itself
            # in case the model is defined in the endpoints.yml
            return self._extra_parameters[API_KEY]

        if os.getenv(AZURE_API_KEY_ENV_VAR) is not None:
            return f"${{{DEFAULT_AZURE_API_KEY_NAME}}}"

        if os.getenv(OPENAI_API_KEY_ENV_VAR) is not None:
            # API key can be set through OPENAI_API_KEY too,
            # because of the backward compatibility
            raise_deprecation_warning(
                message=(
                    f"Usage of '{OPENAI_API_KEY_ENV_VAR}' environment variable "
                    "for setting the API key for Azure OpenAI "
                    "client is deprecated and will be removed "
                    f"in 4.0.0. Please use '{AZURE_API_KEY_ENV_VAR}' "
                    "environment variable."
                )
            )
            return "${OPENAI_API_KEY}"

        structlogger.error(
            "azure_openai_llm_client.api_key_not_set",
            event_info=(
                "API key not set, it is required for API calls. "
                f"Set it either via the environment variable"
                f"'{AZURE_API_KEY_ENV_VAR}' or directly"
                f"via the config key '{API_KEY}'."
            ),
        )
        raise ProviderClientValidationError(
            f"Missing required environment variable/config key '{API_KEY}' for "
            f"API calls."
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> AzureOpenAILLMClient:
        """Initializes the client from given configuration.

        Args:
            config (Dict[str, Any]): Configuration.

        Raises:
            ValueError:
                If any of the required configuration keys are missing.
                If `api_type` has a value different from `azure`.

        Returns:
            AzureOpenAILLMClient: Initialized client.
        """
        try:
            azure_openai_config = AzureOpenAIClientConfig.from_dict(config)
        except ValueError as e:
            message = "Cannot instantiate a client from the passed configuration."
            structlogger.error(
                "azure_openai_llm_client.from_config.error",
                message=message,
                config=config,
                original_error=e,
            )
            raise

        return cls(
            deployment=azure_openai_config.deployment,
            model=azure_openai_config.model,
            api_type=azure_openai_config.api_type,
            api_base=azure_openai_config.api_base,
            api_version=azure_openai_config.api_version,
            oauth=azure_openai_config.oauth,
            **azure_openai_config.extra_parameters,
        )

    @property
    def config(self) -> dict:
        """Returns the configuration for that the llm client
        in dictionary form.
        """
        config = AzureOpenAIClientConfig(
            deployment=self._deployment,
            model=self._model,
            api_base=self._api_base,
            api_version=self._api_version,
            api_type=self._api_type,
            oauth=self._oauth,
            extra_parameters=self._extra_parameters,
        )
        return config.to_dict()

    @property
    def deployment(self) -> str:
        return self._deployment

    @property
    def model(self) -> Optional[str]:
        """Returns the name of the model deployed on Azure."""
        return self._model

    @property
    def api_base(self) -> Optional[str]:
        """Returns the API base URL for the Azure OpenAI llm client."""
        return self._api_base

    @property
    def api_version(self) -> Optional[str]:
        """Returns the API version for the Azure OpenAI llm client."""
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
        regex_pattern = rf"^{AZURE_OPENAI_PROVIDER}/"
        if not re.match(regex_pattern, self._deployment):
            return f"{AZURE_OPENAI_PROVIDER}/{self._deployment}"
        return self._deployment

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        return self._extra_parameters

    @property
    def _completion_fn_args(self) -> Dict[str, Any]:
        """Returns the completion arguments for invoking a call through
        LiteLLM's completion functions.
        """
        # Set the API key env var to None if OAuth is used
        auth_parameter: Dict[str, str] = {}

        if self._oauth:
            auth_parameter = {
                **auth_parameter,
                LITE_LLM_AZURE_AD_TOKEN: self._oauth.get_bearer_token(),
            }
        elif self._api_key_env_var:
            auth_parameter = {LITE_LLM_API_KEY_FIELD: self._api_key_env_var}

        fn_args = super()._completion_fn_args
        fn_args.update(
            {
                LITE_LLM_API_BASE_FIELD: self.api_base,
                LITE_LLM_API_VERSION_FIELD: self.api_version,
                **auth_parameter,
            }
        )
        return fn_args

    def validate_client_setup(self) -> None:
        """Validates that all required configuration parameters are set."""

        return validate_azure_client_setup(
            api_base=self.api_base,
            api_version=self.api_version,
            deployment=self.deployment,
        )
