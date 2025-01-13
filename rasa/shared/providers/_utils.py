import structlog
from litellm import validate_environment

from rasa.shared.constants import (
    AWS_ACCESS_KEY_ID_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_ENV_VAR,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_REGION_NAME_ENV_VAR,
    AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_VAR,
    AWS_SESSION_TOKEN_CONFIG_KEY,
    AWS_SESSION_TOKEN_ENV_VAR,
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
