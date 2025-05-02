from typing import Any, Dict, Optional

import structlog
from litellm import validate_environment

from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_ENV_VAR,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_REGION_NAME_ENV_VAR,
    AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_VAR,
    AWS_SESSION_TOKEN_CONFIG_KEY,
    AWS_SESSION_TOKEN_ENV_VAR,
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    DEPLOYMENT_CONFIG_KEY,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.embedding._base_litellm_embedding_client import (
    _VALIDATE_ENVIRONMENT_MISSING_KEYS_KEY,
)

structlogger = structlog.get_logger()


def validate_aws_setup_for_litellm_clients(
    litellm_model_name: str, litellm_call_kwargs: dict, source_log: str
) -> None:
    """Validates the AWS setup for LiteLLM clients to ensure all required
    environment variables or corresponding call kwargs are set.

    Args:
        litellm_model_name (str): The name of the LiteLLM model being validated.
        litellm_call_kwargs (dict): Additional keyword arguments passed to the client,
            which may include configuration values for AWS credentials.
        source_log (str): The source log identifier for structured logging.

    Raises:
        ProviderClientValidationError: If any required AWS environment variable
            or corresponding configuration key is missing.
    """

    # Mapping of environment variable names to their corresponding config keys
    envs_to_args = {
        AWS_ACCESS_KEY_ID_ENV_VAR: AWS_ACCESS_KEY_ID_CONFIG_KEY,
        AWS_SECRET_ACCESS_KEY_ENV_VAR: AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
        AWS_REGION_NAME_ENV_VAR: AWS_REGION_NAME_CONFIG_KEY,
        AWS_SESSION_TOKEN_ENV_VAR: AWS_SESSION_TOKEN_CONFIG_KEY,
    }

    # Validate the environment setup for the model
    validation_info = validate_environment(litellm_model_name)
    missing_environment_variables = validation_info.get(
        _VALIDATE_ENVIRONMENT_MISSING_KEYS_KEY, []
    )
    # Filter out missing environment variables that have been set trough arguments
    # in extra parameters
    missing_environment_variables = [
        missing_env_var
        for missing_env_var in missing_environment_variables
        if litellm_call_kwargs.get(envs_to_args.get(missing_env_var)) is None
    ]

    if missing_environment_variables:
        missing_environment_details = [
            (
                f"'{missing_env_var}' environment variable or "
                f"'{envs_to_args.get(missing_env_var)}' config key"
            )
            for missing_env_var in missing_environment_variables
        ]
        event_info = (
            f"The following environment variables or configuration keys are "
            f"missing: "
            f"{', '.join(missing_environment_details)}. "
            f"These settings are required for API calls."
        )
        structlogger.error(
            f"{source_log}.validate_aws_environment_variables",
            event_info=event_info,
            missing_environment_variables=missing_environment_variables,
        )
        raise ProviderClientValidationError(event_info)


def validate_azure_client_setup(
    api_base: Optional[str],
    api_version: Optional[str],
    deployment: Optional[str],
) -> None:
    """Validates the Azure setup for LiteLLM Router clients to ensure
     that all required configuration parameters are set.
    Raises:
        ProviderClientValidationError: If any required Azure configurations
            is missing.
    """

    def generate_event_info_for_missing_setting(
        setting: str,
        setting_env_var: Optional[str] = None,
        setting_config_key: Optional[str] = None,
    ) -> str:
        """Generate a part of the message with instructions on what to set
        for the missing client setting.
        """
        info = "Set {setting} with {options}. "
        options = ""
        if setting_env_var is not None:
            options += f"environment variable '{setting_env_var}'"
        if setting_config_key is not None and setting_env_var is not None:
            options += " or "
        if setting_config_key is not None:
            options += f"config key '{setting_config_key}'"

        return info.format(setting=setting, options=options)

    # All required settings for Azure OpenAI client
    settings: Dict[str, Dict[str, Any]] = {
        "API Base": {
            "current_value": api_base,
            "env_var": AZURE_API_BASE_ENV_VAR,
            "config_key": API_BASE_CONFIG_KEY,
        },
        "API Version": {
            "current_value": api_version,
            "env_var": AZURE_API_VERSION_ENV_VAR,
            "config_key": API_VERSION_CONFIG_KEY,
        },
        "Deployment Name": {
            "current_value": deployment,
            "env_var": None,
            "config_key": DEPLOYMENT_CONFIG_KEY,
        },
    }

    missing_settings = [
        setting_name
        for setting_name, setting_info in settings.items()
        if setting_info["current_value"] is None
    ]

    if missing_settings:
        event_info = f"Client settings not set: {', '.join(missing_settings)}. "

        for missing_setting in missing_settings:
            if settings[missing_setting]["current_value"] is not None:
                continue
            event_info += generate_event_info_for_missing_setting(
                missing_setting,
                settings[missing_setting]["env_var"],
                settings[missing_setting]["config_key"],
            )

        structlogger.error(
            "azure_openai_llm_client.not_configured",
            event_info=event_info,
            missing_settings=missing_settings,
        )
        raise ProviderClientValidationError(event_info)
