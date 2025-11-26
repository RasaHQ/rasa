import os
from typing import Any, Dict, Optional, Set

import boto3
import structlog
from botocore.exceptions import BotoCoreError, ClientError
from google.auth.environment_vars import AWS_DEFAULT_REGION

from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_ENV_VAR,
    AWS_BEDROCK_PROVIDER,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_REGION_NAME_ENV_VAR,
    AWS_SAGEMAKER_CHAT_PROVIDER,
    AWS_SAGEMAKER_PROVIDER,
    AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_VAR,
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
    from rasa.shared.utils.health_check.health_check import is_api_health_check_enabled

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

    if is_api_health_check_enabled():
        structlogger.debug(
            f"{source_log}.validating_aws_credentials_for_litellm_clients_via_aws_client",
            model_name=litellm_model_name,
        )
        _validate_credentials_with_aws_client(
            provider,
            additional_kwargs,
            litellm_model_name,
            source_log,
        )
        return None

    return _validate_credentials_exist(
        resolved_litellm_call_kwargs, litellm_model_name, source_log
    )


def _validate_credentials_exist(
    resolved_litellm_call_kwargs: Dict[str, Any],
    litellm_model_name: str,
    source_log: str,
) -> None:
    """Validates that AWS credentials are provided.

    Args:
        resolved_litellm_call_kwargs (Dict[str, Any]): The resolved keyword arguments
            containing AWS credentials.
        litellm_model_name (str): The name of the LiteLLM model being validated.
        source_log (str): The source log identifier for structured logging.

    Raises:
        ProviderClientValidationError: If any required AWS credentials are missing.
    """
    required_iam_credential_keys = {"model_id", AWS_REGION_NAME_CONFIG_KEY}
    required_env_var_secrets = {
        AWS_ACCESS_KEY_ID_CONFIG_KEY,
        AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
        AWS_REGION_NAME_CONFIG_KEY,
    }

    provided_credentials: Dict[str, Any] = _add_env_vars_credentials_if_defined(
        resolved_litellm_call_kwargs
    )
    provided_credentials_keys = set(provided_credentials.keys())
    common_credentials = (
        required_iam_credential_keys | required_env_var_secrets
    ) & provided_credentials_keys

    if not common_credentials:
        event_info = (
            "Missing AWS credentials for LiteLLM clients. "
            "Ensure that you are using one of the available authentication methods: "
            "endpoints yml, environment variables, or IAM roles. "
        )
        structlogger.error(
            f"{source_log}.validate_aws_credentials_existence_for_litellm_clients",
            event_info=event_info,
            model_name=litellm_model_name,
        )
        raise ProviderClientValidationError(event_info)

    # if the chosen auth method is via env vars,
    # ensure that all required env vars are set
    if common_credentials.issubset(required_env_var_secrets):
        _verify_missing_credentials(
            required_env_var_secrets, common_credentials, litellm_model_name, source_log
        )

    if common_credentials.issubset(required_iam_credential_keys):
        _verify_missing_credentials(
            required_iam_credential_keys,
            common_credentials,
            litellm_model_name,
            source_log,
        )


def _add_env_vars_credentials_if_defined(
    resolved_litellm_call_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Adds AWS credentials from environment variables to the resolved call kwargs.

    Args:
        resolved_litellm_call_kwargs (Dict[str, Any]): The resolved keyword arguments.

    Returns:
        Dict[str, Any]: The updated keyword arguments with AWS credentials from
            environment variables if they were defined.
    """
    if AWS_ACCESS_KEY_ID_CONFIG_KEY not in resolved_litellm_call_kwargs:
        env_var_value = os.getenv(AWS_ACCESS_KEY_ID_ENV_VAR)
        if env_var_value is not None:
            resolved_litellm_call_kwargs[AWS_ACCESS_KEY_ID_CONFIG_KEY] = env_var_value

    if AWS_SECRET_ACCESS_KEY_CONFIG_KEY not in resolved_litellm_call_kwargs:
        env_var_value = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_VAR)
        if env_var_value is not None:
            resolved_litellm_call_kwargs[AWS_SECRET_ACCESS_KEY_CONFIG_KEY] = (
                env_var_value
            )

    if AWS_REGION_NAME_CONFIG_KEY not in resolved_litellm_call_kwargs:
        env_var_value = os.getenv(
            AWS_REGION_NAME_ENV_VAR, os.getenv(AWS_DEFAULT_REGION)
        )
        if env_var_value is not None:
            resolved_litellm_call_kwargs[AWS_REGION_NAME_CONFIG_KEY] = env_var_value

    return resolved_litellm_call_kwargs


def _verify_missing_credentials(
    required_credentials: Set[str],
    present_credentials: Set[str],
    litellm_model_name: str,
    source_log: str,
) -> None:
    """Verifies if any required credentials are missing.

    Args:
        present_credentials (set): The set of essential credential keys
            that are present.
        required_credentials (set): The set of required credential keys.
        litellm_model_name (str): The name of the LiteLLM model being validated.
        source_log (str): The source log identifier for structured logging.

    Raises:
        ProviderClientValidationError: If any required credentials are missing.
    """
    missing_credentials = required_credentials - present_credentials
    if missing_credentials:
        event_info = (
            f"Missing AWS credentials for "
            f"LiteLLM clients: {', '.join(missing_credentials)}. "
        )
        structlogger.error(
            f"{source_log}.validate_aws_credentials_existence_for_litellm_clients",
            event_info=event_info,
            model_name=litellm_model_name,
        )
        raise ProviderClientValidationError(event_info)


def _validate_credentials_with_aws_client(
    provider: str,
    additional_kwargs: Dict[str, Any],
    litellm_model_name: str,
    source_log: str,
) -> None:
    """Creates an AWS client with the provided credentials.

    Args:
        provider (str): The AWS service provider.
        additional_kwargs (Dict[str, Any]): Additional keyword arguments for the client.
        litellm_model_name (str): The name of the LiteLLM model being validated.
        source_log (str): The source log identifier for structured logging.
    """
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
            f"{source_log}.validate_aws_credentials_for_litellm_clients_via_aws_client.failed",
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
