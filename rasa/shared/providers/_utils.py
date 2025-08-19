from typing import Any, Dict, Optional

import boto3
import structlog
from botocore.exceptions import BotoCoreError, ClientError

from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_CONFIG_KEY,
    AWS_BEDROCK_PROVIDER,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_SAGEMAKER_CHAT_PROVIDER,
    AWS_SAGEMAKER_PROVIDER,
    AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
    AWS_SESSION_TOKEN_CONFIG_KEY,
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    DEPLOYMENT_CONFIG_KEY,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.utils.io import resolve_environment_variables

structlogger = structlog.get_logger()


def validate_aws_setup_for_litellm_clients(
    litellm_model_name: str, litellm_call_kwargs: Dict, source_log: str, provider: str
) -> None:
    """Validates the AWS setup for LiteLLM clients to ensure credentials are set.

    Args:
        litellm_model_name (str): The name of the LiteLLM model being validated.
        litellm_call_kwargs (dict): Additional keyword arguments passed to the client,
            which may include configuration values for AWS credentials.
        source_log (str): The source log identifier for structured logging.
        provider (str): The provider for which the validation is being performed.

    Raises:
        ProviderClientValidationError: If any required AWS environment variable
            or corresponding configuration key is missing.
    """
    # expand environment variables if referenced in the config
    resolved_litellm_call_kwargs: Dict = resolve_environment_variables(
        litellm_call_kwargs
    )  # type: ignore[assignment]

    # boto3 only accepts bedrock and sagemaker as valid clients
    # therefore we need to convert the provider name if it is defined
    # as sagemaker_chat
    provider = (
        AWS_SAGEMAKER_PROVIDER if provider == AWS_SAGEMAKER_CHAT_PROVIDER else provider
    )

    # if the AWS credentials are defined in the endpoints yaml model config,
    # either as referenced secret env vars or direct values, we need to pass them
    # to the boto3 client to ensure that the client can connect to the AWS service.
    additional_kwargs: Dict[str, Any] = {}
    if AWS_ACCESS_KEY_ID_CONFIG_KEY in resolved_litellm_call_kwargs:
        additional_kwargs[AWS_ACCESS_KEY_ID_CONFIG_KEY] = resolved_litellm_call_kwargs[
            AWS_ACCESS_KEY_ID_CONFIG_KEY
        ]
    if AWS_SECRET_ACCESS_KEY_CONFIG_KEY in resolved_litellm_call_kwargs:
        additional_kwargs[AWS_SECRET_ACCESS_KEY_CONFIG_KEY] = (
            resolved_litellm_call_kwargs[AWS_SECRET_ACCESS_KEY_CONFIG_KEY]
        )
    if AWS_SESSION_TOKEN_CONFIG_KEY in resolved_litellm_call_kwargs:
        additional_kwargs[AWS_SESSION_TOKEN_CONFIG_KEY] = resolved_litellm_call_kwargs[
            AWS_SESSION_TOKEN_CONFIG_KEY
        ]
    if AWS_REGION_NAME_CONFIG_KEY in resolved_litellm_call_kwargs:
        additional_kwargs["region_name"] = resolved_litellm_call_kwargs[
            AWS_REGION_NAME_CONFIG_KEY
        ]

    try:
        # We are using the boto3 client because it can discover the AWS credentials
        # from the environment variables, credentials file, or IAM roles.
        # This is necessary to ensure that the client can connect to the AWS service.
        aws_client = boto3.client(provider, **additional_kwargs)

        # Using different method calls available to different AWS clients
        # to test the connection
        if provider == AWS_SAGEMAKER_PROVIDER:
            aws_client.list_models()
        elif provider == AWS_BEDROCK_PROVIDER:
            aws_client.get_model_invocation_logging_configuration()

    except (ClientError, BotoCoreError) as exc:
        event_info = (
            f"Failed to validate AWS setup for LiteLLM clients: {exc}. "
            f"Ensure that you are using one of the available authentication methods:"
            f"credentials file, environment variables, or IAM roles. "
            f"Also, ensure that the AWS region is set correctly. "
        )
        structlogger.error(
            f"{source_log}.validate_aws_credentials_for_litellm_clients",
            event_info=event_info,
            exception=str(exc),
            model_name=litellm_model_name,
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
