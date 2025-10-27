from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Dict,
    Optional,
    Set,
)

import structlog
from pydantic import BaseModel

from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    API_KEY,
    API_TYPE_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    AZURE_API_TYPE,
    AZURE_OPENAI_PROVIDER,
    DEPLOYMENT_CONFIG_KEY,
    DEPLOYMENT_NAME_CONFIG_KEY,
    ENGINE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
    MAX_COMPLETION_TOKENS_CONFIG_KEY,
    MAX_TOKENS_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
    OPENAI_API_BASE_CONFIG_KEY,
    OPENAI_API_TYPE_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
    REQUEST_TIMEOUT_CONFIG_KEY,
    STREAM_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.providers._configs.azure_entra_id_config import (
    AzureEntraIDOAuthConfig,
    AzureEntraIDOAuthType,
)
from rasa.shared.providers._configs.oauth_config import (
    OAUTH_KEY,
    OAUTH_TYPE_FIELD,
    OAuth,
)
from rasa.shared.utils.common import class_from_module_path
from rasa.shared.utils.configs import (
    raise_deprecation_warnings,
    resolve_aliases,
    validate_forbidden_keys,
    validate_required_keys,
)

structlogger = structlog.get_logger()

DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING = {
    # Deployment name aliases
    DEPLOYMENT_NAME_CONFIG_KEY: DEPLOYMENT_CONFIG_KEY,
    ENGINE_CONFIG_KEY: DEPLOYMENT_CONFIG_KEY,
    # Provider aliases
    RASA_TYPE_CONFIG_KEY: PROVIDER_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY: PROVIDER_CONFIG_KEY,
    # API type aliases
    OPENAI_API_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    # API base aliases
    OPENAI_API_BASE_CONFIG_KEY: API_BASE_CONFIG_KEY,
    # API version aliases
    OPENAI_API_VERSION_CONFIG_KEY: API_VERSION_CONFIG_KEY,
    # Model name aliases
    MODEL_NAME_CONFIG_KEY: MODEL_CONFIG_KEY,
    # Timeout aliases
    REQUEST_TIMEOUT_CONFIG_KEY: TIMEOUT_CONFIG_KEY,
    # Max tokens aliases
    MAX_TOKENS_CONFIG_KEY: MAX_COMPLETION_TOKENS_CONFIG_KEY,
}

REQUIRED_KEYS = [DEPLOYMENT_CONFIG_KEY]

FORBIDDEN_KEYS = [
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
]


class OAuthConfigWrapper(OAuth, BaseModel):
    """Wrapper for OAuth configuration.

    It's main purpose is to provide to_dict method which is used to serialize
    the oauth configuration to the original format.

    """

    # Pydantic configuration to allow arbitrary user defined types
    class Config:
        arbitrary_types_allowed = True

    oauth: OAuth
    original_config: Dict[str, Any]

    def get_bearer_token(self) -> str:
        """Returns a bearer token."""
        return self.oauth.get_bearer_token()

    def to_dict(self) -> Dict[str, Any]:
        """Converts the OAuth configuration to the original format."""
        return self.original_config

    @staticmethod
    def _valid_type_values() -> Set[str]:
        """Returns the valid built-in values for the `type` field in the `oauth`."""
        return AzureEntraIDOAuthType.valid_string_values()

    @classmethod
    def from_dict(cls, oauth_config: Dict[str, Any]) -> OAuthConfigWrapper:
        """Initializes a dataclass from the passed config.

        Args:
            oauth_config: (dict) The config from which to initialize.

        Returns:
            AzureOAuthConfig
        """
        original_config = deepcopy(oauth_config)

        oauth_type: Optional[str] = oauth_config.get(OAUTH_TYPE_FIELD, None)

        if oauth_type is None:
            message = (
                "Oauth configuration must contain "
                f"'{OAUTH_TYPE_FIELD}' field and it must be set to one of the "
                f"following values: {OAuthConfigWrapper._valid_type_values()}, "
                f"or to the path of module which is "
                f"implementing {OAuth.__name__} protocol."
            )
            structlogger.error(
                "azure_oauth_config.missing_oauth_type",
                message=message,
            )
            raise ValueError(message)

        if oauth_type in AzureEntraIDOAuthType.valid_string_values():
            return cls(
                oauth=AzureEntraIDOAuthConfig.from_dict(oauth_config),
                original_config=original_config,
            )

        module = class_from_module_path(oauth_type)

        if not issubclass(module, OAuth):
            message = (
                f"Module {oauth_type} does not implement "
                f"{OAuth.__name__} interface."
            )
            structlogger.error(
                "azure_oauth_config.invalid_oauth_module",
                message=message,
            )
            raise ValueError(message)

        return cls(
            oauth=module.from_dict(oauth_config), original_config=original_config
        )


@dataclass
class AzureOpenAIClientConfig:
    """Parses configuration for Azure OpenAI client.

    Resolves aliases and raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
            - If `api_type` has a value different from `azure`.
    """

    deployment: str

    model: Optional[str]
    api_base: Optional[str]
    api_version: Optional[str]
    # API Type is not used by LiteLLM backend, but we define
    # it here for backward compatibility.
    api_type: Optional[str] = AZURE_API_TYPE
    # Provider is not used by LiteLLM backend, but we define it here since it's
    # used as switch between different clients.
    provider: str = AZURE_OPENAI_PROVIDER

    # OAuth related parameters
    oauth: Optional[OAuthConfigWrapper] = None

    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider != AZURE_OPENAI_PROVIDER:
            message = f"Provider must be set to '{AZURE_OPENAI_PROVIDER}'."
            structlogger.error(
                "azure_openai_client_config.validation_error",
                message=message,
                provider=self.provider,
            )
            raise ValueError(message)
        if self.deployment is None:
            message = "Deployment cannot be set to None."
            structlogger.error(
                "azure_openai_client_config.validation_error",
                message=message,
                deployment=self.deployment,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> AzureOpenAIClientConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Raised in cases of invalid configuration:
                - If any of the required configuration keys are missing.
                - If `api_type` has a value different from `azure`.

        Returns:
            AzureOpenAIClientConfig
        """
        # Check for deprecated keys
        raise_deprecation_warnings(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Resolve any potential aliases
        config = cls.resolve_config_aliases(config)
        # Validate that required keys are set
        validate_required_keys(config, REQUIRED_KEYS)
        # Validate that the forbidden keys are not present
        validate_forbidden_keys(config, FORBIDDEN_KEYS)
        # Init client config

        cls._validate_authentication_configuration(config)

        has_oauth_key = config.get(OAUTH_KEY, None) is not None
        oauth = (
            OAuthConfigWrapper.from_dict(config.pop(OAUTH_KEY))
            if has_oauth_key
            else None
        )

        this = AzureOpenAIClientConfig(
            # Required parameters
            deployment=config.pop(DEPLOYMENT_CONFIG_KEY),
            # Pop the 'provider' key. Currently, it's *optional* because of
            # backward compatibility with older versions.
            provider=config.pop(PROVIDER_CONFIG_KEY, AZURE_OPENAI_PROVIDER),
            # Optional
            api_type=config.pop(API_TYPE_CONFIG_KEY, AZURE_API_TYPE),
            model=config.pop(MODEL_CONFIG_KEY, None),
            # Optional, can also be set through environment variables
            # in clients.
            api_base=config.pop(API_BASE_CONFIG_KEY, None),
            api_version=config.pop(API_VERSION_CONFIG_KEY, None),
            # OAuth related parameters, set only if auth_type is set to 'entra_id'
            oauth=oauth,
            # The rest of parameters (e.g. model parameters) are considered
            # as extra parameters (this also includes timeout).
            extra_parameters=config,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        d = asdict(self)
        # Extra parameters should also be on the top level
        d.pop("extra_parameters", None)
        d.update(self.extra_parameters)

        d.pop("oauth", None)
        d.update({"oauth": self.oauth.to_dict()} if self.oauth else {})
        return d

    @staticmethod
    def resolve_config_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_aliases(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)

    @staticmethod
    def _validate_authentication_configuration(config: Dict[str, Any]) -> None:
        """Validates the authentication configuration."""
        has_api_key = config.get(API_KEY, None) is not None
        has_oauth_key = config.get(OAUTH_KEY, None) is not None

        if has_api_key and has_oauth_key:
            message = (
                "Azure OpenAI client configuration cannot contain "
                f"both '{API_KEY}' and '{OAUTH_KEY}' fields. Please provide either "
                f"'{API_KEY}' or '{OAUTH_KEY}' fields."
            )
            structlogger.error(
                "azure_openai_client_config.multiple_auth_types_specified",
                message=message,
            )
            raise ValueError(message)


def is_azure_openai_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure an Azure OpenAI client."""
    # Resolve any aliases that are specific to Azure OpenAI configuration
    config = AzureOpenAIClientConfig.resolve_config_aliases(config)

    # Case: Configuration contains `provider: azure`.
    if config.get(PROVIDER_CONFIG_KEY) == AZURE_OPENAI_PROVIDER:
        return True

    # Case: Configuration contains `deployment` key
    # (specific to Azure OpenAI configuration)
    if (
        config.get(DEPLOYMENT_CONFIG_KEY) is not None
        and config.get(PROVIDER_CONFIG_KEY) is None
    ):
        return True

    return False
