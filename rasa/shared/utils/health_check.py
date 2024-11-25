import os
from typing import Optional, Dict, Any

from rasa.shared.constants import (
    LLM_API_HEALTH_CHECK_ENV_VAR,
    MODELS_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    LLM_API_HEALTH_CHECK_DEFAULT_VALUE,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.utils.cli import print_error_and_exit
from rasa.shared.utils.llm import llm_factory, structlogger, embedder_factory


def try_instantiate_llm_client(
    custom_llm_config: Optional[Dict],
    default_llm_config: Optional[Dict],
    log_source_function: str,
    log_source_component: str,
) -> LLMClient:
    """Validate llm configuration."""
    try:
        return llm_factory(custom_llm_config, default_llm_config)
    except (ProviderClientValidationError, ValueError) as e:
        structlogger.error(
            f"{log_source_function}.llm_instantiation_failed",
            message="Unable to instantiate LLM client.",
            error=e,
        )
        print_error_and_exit(
            f"Unable to create the LLM client for component - {log_source_component}. "
            f"Please make sure you specified the required environment variables "
            f"and configuration keys. "
            f"Error: {e}"
        )


def try_instantiate_embedder(
    custom_embeddings_config: Optional[Dict],
    default_embeddings_config: Optional[Dict],
    log_source_function: str,
    log_source_component: str,
) -> EmbeddingClient:
    """Validate embeddings configuration."""
    try:
        return embedder_factory(custom_embeddings_config, default_embeddings_config)
    except (ProviderClientValidationError, ValueError) as e:
        structlogger.error(
            f"{log_source_function}.embedder_instantiation_failed",
            message="Unable to instantiate Embedding client.",
            error=e,
        )
        print_error_and_exit(
            f"Unable to create the Embedding client for component - "
            f"{log_source_component}. Please make sure you specified the required "
            f"environment variables and configuration keys. Error: {e}"
        )


def perform_training_time_llm_health_check(
    custom_config: Optional[Dict[str, Any]],
    default_config: Dict[str, Any],
    log_source_function: str,
    log_source_component: str,
) -> Optional[str]:
    """Try to instantiate the LLM Client to validate the provided config.
    If the LLM_API_HEALTH_CHECK environment variable is true, perform a test call
    to the LLM API. If config contains multiple models, perform a consistency
    check for the model group.

    This method supports both single model configurations and model group configurations
    (configs that have the `models` key).

    Returns:
        model name from the API response or `None` if LLM_API_HEALTH_CHECK is false
    """
    llm_client = try_instantiate_llm_client(
        custom_config, default_config, log_source_function, log_source_component
    )

    if (
        os.getenv(
            LLM_API_HEALTH_CHECK_ENV_VAR, LLM_API_HEALTH_CHECK_DEFAULT_VALUE
        ).lower()
        == "true"
    ):
        train_model_name: Optional[str] = None
        if (
            custom_config
            and MODELS_CONFIG_KEY in custom_config
            and len(custom_config[MODELS_CONFIG_KEY]) > 1
        ):
            train_model_name = perform_llm_model_group_consistency_check(
                custom_config,
                default_config,
                log_source_function,
                log_source_component,
            )
        else:
            train_model_name = send_test_llm_api_request(
                llm_client,
                log_source_function,
                log_source_component,
            )
        return train_model_name
    else:
        structlogger.warning(
            f"{log_source_function}.perform_training_time_llm_health_check.disabled",
            event_info=(
                f"The {LLM_API_HEALTH_CHECK_ENV_VAR} environment variable is set "
                f"to false, which will disable model consistency check. "
                f"It is recommended to set this variable to true in production "
                f"environments."
            ),
        )
        return None


def perform_training_time_embeddings_health_check(
    custom_config: Optional[Dict[str, Any]],
    default_config: Dict[str, Any],
    log_source_function: str,
    log_source_component: str,
) -> Optional[str]:
    """Try to instantiate the Embedder to validate the provided config.
    If the LLM_API_HEALTH_CHECK environment variable is true, perform a test call
    to the Embeddings API. If config contains multiple models, perform a consistency
    check for the model group.

    This method supports both single model configurations and model group configurations
    (configs that have the `models` key).

    Returns:
        model name from the API response or `None` if LLM_API_HEALTH_CHECK is false
    """
    if (
        os.getenv(
            LLM_API_HEALTH_CHECK_ENV_VAR, LLM_API_HEALTH_CHECK_DEFAULT_VALUE
        ).lower()
        == "true"
    ):
        train_model_name: Optional[str] = None
        if (
            custom_config
            and MODELS_CONFIG_KEY in custom_config
            and len(custom_config[MODELS_CONFIG_KEY]) > 1
        ):
            train_model_name = perform_embeddings_model_group_consistency_check(
                custom_config,
                default_config,
                log_source_function,
                log_source_component,
            )
        else:
            embedder = try_instantiate_embedder(
                custom_config, default_config, log_source_function, log_source_component
            )
            train_model_name = send_test_embeddings_api_request(
                embedder, log_source_function, log_source_component
            )
        return train_model_name
    else:
        structlogger.warning(
            f"{log_source_function}"
            f".perform_training_time_embeddings_health_check.disabled",
            event_info=(
                f"The {LLM_API_HEALTH_CHECK_ENV_VAR} environment variable is set "
                f"to false, which will disable model consistency check. "
                f"It is recommended to set this variable to true in production "
                f"environments."
            ),
        )
        return None


def perform_inference_time_llm_health_check(
    custom_config: Optional[Dict[str, Any]],
    default_config: Dict[str, Any],
    train_model_name: Optional[str],
    log_source_function: str,
    log_source_component: str,
) -> None:
    """If the LLM_API_HEALTH_CHECK environment variable is true, perform a test call
    to the LLM API. If config contains multiple models, perform a consistency
    check for the model group.
    Compare the model name from the API response with the model name used for training
    (if available) and raise an exception if they are different.

    This method supports both single model configurations and model group configurations
    (configs that have the `models` key).
    """
    if (
        os.getenv(
            LLM_API_HEALTH_CHECK_ENV_VAR, LLM_API_HEALTH_CHECK_DEFAULT_VALUE
        ).lower()
        == "true"
    ):
        structlogger.info(
            f"{log_source_function}.perform_inference_time_llm_health_check",
            event_info=(
                f"Performing an inference-time health check on the LLM API for "
                f"the component - {log_source_component}."
            ),
            config=custom_config,
        )

        inference_model_name: Optional[str] = None
        if (
            custom_config
            and MODELS_CONFIG_KEY in custom_config
            and len(custom_config[MODELS_CONFIG_KEY]) > 1
        ):
            inference_model_name = perform_llm_model_group_consistency_check(
                custom_config,
                default_config,
                log_source_function,
                log_source_component,
            )
        else:
            llm_client = try_instantiate_llm_client(
                custom_config,
                default_config,
                log_source_function,
                log_source_component,
            )
            inference_model_name = send_test_llm_api_request(
                llm_client,
                log_source_function,
                log_source_component,
            )

        if not inference_model_name:
            structlogger.warning(
                f"{log_source_function}"
                f".perform_inference_time_llm_health_check.no_inference_model",
                event_info=(
                    "Failed to perform model consistency check: "
                    "the API response does not contain a model name."
                ),
            )
        elif not train_model_name:
            structlogger.warning(
                f"{log_source_function}"
                f".perform_inference_time_llm_health_check.no_train_model",
                event_info=(
                    f"The model was trained with {LLM_API_HEALTH_CHECK_ENV_VAR} "
                    f"environment variable set to false, so the model "
                    f"consistency check is not available."
                ),
            )
        elif inference_model_name != train_model_name:
            error_message = (
                f"The LLM used to train the {log_source_component} "
                f"({train_model_name}) is not the same as the LLM used for inference "
                f"({inference_model_name}). Please verify your configuration."
            )
            structlogger.error(
                f"{log_source_function}.train_inference_llm_model_mismatch",
                event_info=error_message,
            )
            print_error_and_exit(error_message)
    else:
        structlogger.warning(
            f"{log_source_function}.perform_inference_time_llm_health_check.disabled",
            event_info=(
                f"The {LLM_API_HEALTH_CHECK_ENV_VAR} environment variable is set "
                f"to false, which will disable model consistency check. "
                f"It is recommended to set this variable to true in production "
                f"environments."
            ),
        )


def perform_inference_time_embeddings_health_check(
    custom_config: Optional[Dict[str, Any]],
    default_config: Dict[str, Any],
    train_model_name: Optional[str],
    log_source_function: str,
    log_source_component: str,
) -> None:
    """If the LLM_API_HEALTH_CHECK environment variable is true, perform a test call
    to the Embeddings API. If config contains multiple models, perform a consistency
    check for the model group.
    Compare the model name from the API response with the model name used for training
    (if available) and raise an exception if they are different.

    This method supports both single model configurations and model group configurations
    (configs that have the `models` key).
    """
    if (
        os.getenv(
            LLM_API_HEALTH_CHECK_ENV_VAR, LLM_API_HEALTH_CHECK_DEFAULT_VALUE
        ).lower()
        == "true"
    ):
        structlogger.info(
            f"{log_source_function}.perform_inference_time_embeddings_health_check",
            event_info=(
                f"Performing an inference-time health check on the Embeddings API for "
                f"the component - {log_source_component}."
            ),
            config=custom_config,
        )

        inference_model_name: Optional[str] = None
        if (
            custom_config
            and MODELS_CONFIG_KEY in custom_config
            and len(custom_config[MODELS_CONFIG_KEY]) > 1
        ):
            inference_model_name = perform_embeddings_model_group_consistency_check(
                custom_config,
                default_config,
                log_source_function,
                log_source_component,
            )
        else:
            embedder = try_instantiate_embedder(
                custom_config, default_config, log_source_function, log_source_component
            )
            inference_model_name = send_test_embeddings_api_request(
                embedder, log_source_function, log_source_component
            )

        if not inference_model_name:
            structlogger.warning(
                f"{log_source_function}"
                f".perform_inference_time_embeddings_health_check.no_inference_model",
                event_info=(
                    "Failed to perform embeddings model consistency check: "
                    "the API response does not contain a model name."
                ),
            )
        elif not train_model_name:
            structlogger.warning(
                f"{log_source_function}"
                f".perform_inference_time_embeddings_health_check.no_train_model",
                event_info=(
                    f"The model was trained with {LLM_API_HEALTH_CHECK_ENV_VAR} "
                    f"environment variable set to false, so the model "
                    f"consistency check is not available."
                ),
            )
        elif inference_model_name != train_model_name:
            error_message = (
                f"The Embeddings model used to train the {log_source_component} "
                f"({train_model_name}) is not the same as the model used for inference "
                f"({inference_model_name}). Please verify your configuration."
            )
            structlogger.error(
                f"{log_source_function}.train_inference_embeddings_model_mismatch",
                event_info=error_message,
            )
            print_error_and_exit(error_message)
    else:
        structlogger.warning(
            f"{log_source_function}"
            f".perform_inference_time_embeddings_health_check.disabled",
            event_info=(
                f"The {LLM_API_HEALTH_CHECK_ENV_VAR} environment variable is set "
                f"to false, which will disable model consistency check. "
                f"It is recommended to set this variable to true in production "
                f"environments."
            ),
        )


def perform_llm_model_group_consistency_check(
    custom_config: Dict[str, Any],
    default_config: Dict[str, Any],
    log_source_function: str,
    log_source_component: str,
) -> Optional[str]:
    """Perform a consistency check for multiple models inside a model group.

    This function checks if all models within a model group are consistent by verifying
    that they all return the same model name from the LLM API health check.

    This method supports only model group configuration (config that have the `models`
    key) and will ignore single model configuration.

    Returns:
        The model name if all models are consistent, otherwise raises
        an InvalidConfigException.

    Raises:
        InvalidConfigException: If the model group contains different models.
    """
    if not custom_config or MODELS_CONFIG_KEY not in custom_config:
        return None

    model_names = set()
    for model_config in custom_config[MODELS_CONFIG_KEY]:
        llm_client = try_instantiate_llm_client(
            model_config, default_config, log_source_function, log_source_component
        )
        model_name = send_test_llm_api_request(
            llm_client, log_source_function, log_source_component
        )
        model_names.add(model_name)

    if len(model_names) > 1:
        error_message = (
            f"The model group {custom_config.get(MODEL_GROUP_ID_CONFIG_KEY)} used by "
            f"{log_source_component} component is inconsistent. "
            f"It contains different models: {model_names}. "
            f"Please verify your configuration."
        )
        structlogger.error(
            f"{log_source_function}.inconsistent_model_group", event_info=error_message
        )
        print_error_and_exit(error_message)

    return model_names.pop() if len(model_names) > 0 else None


def send_test_llm_api_request(
    llm_client: LLMClient, log_source_function: str, log_source_component: str
) -> Optional[str]:
    """Sends a test request to the LLM API to perform a health check.

    Returns:
        The model name from the API response.

    Raises:
        Exception: If the API call fails.
    """
    structlogger.info(
        f"{log_source_function}.send_test_llm_api_request",
        event_info=(
            f"Sending a test LLM API request for the component - "
            f"{log_source_component}."
        ),
        config=llm_client.config,
    )
    try:
        response = llm_client.completion("hello")
        return response.model
    except Exception as e:
        structlogger.error(
            f"{log_source_function}.send_test_llm_api_request_failed",
            event_info="Test call to the LLM API failed.",
            error=e,
        )
        print_error_and_exit(
            f"Test call to the LLM API failed for component - {log_source_component}. "
            f"Error: {e}"
        )


def send_test_embeddings_api_request(
    embedder: EmbeddingClient, log_source_function: str, log_source_component: str
) -> Optional[str]:
    """Sends a test request to the Embeddings API to perform a health check.

    Returns:
        The model name from the API response.

    Raises:
        Exception: If the API call fails.
    """
    structlogger.info(
        f"{log_source_function}.send_test_embeddings_api_request",
        event_info=(
            f"Sending a test Embeddings API request for the component - "
            f"{log_source_component}."
        ),
        config=embedder.config,
    )
    try:
        response = embedder.embed(["hello"])
        return response.model
    except Exception as e:
        structlogger.error(
            f"{log_source_function}.send_test_llm_api_request_failed",
            event_info="Test call to the Embeddings API failed.",
            error=e,
        )
        print_error_and_exit(
            f"Test call to the Embeddings API failed for component - "
            f"{log_source_component}. Error: {e}"
        )


def perform_embeddings_model_group_consistency_check(
    custom_config: Dict[str, Any],
    default_config: Dict[str, Any],
    log_source_function: str,
    log_source_component: str,
) -> Optional[str]:
    """Perform a consistency check for multiple embeddings models inside a model group.

    This function checks if all models within a model group are consistent by verifying
    that they all return the same model name from the Embeddings API health check.

    This method supports only model group configuration (config that have the `models`
    key) and will ignore single model configuration.

    Returns:
        The model name if all models are consistent, otherwise raises
        an InvalidConfigException.

    Raises:
        InvalidConfigException: If the model group contains different models.
    """
    if not custom_config or MODELS_CONFIG_KEY not in custom_config:
        return None

    model_names = set()
    for model_config in custom_config[MODELS_CONFIG_KEY]:
        embedder = try_instantiate_embedder(
            custom_config, default_config, log_source_function, log_source_component
        )
        model_name = send_test_embeddings_api_request(
            embedder, log_source_function, log_source_component
        )
        model_names.add(model_name)

    if len(model_names) > 1:
        error_message = (
            f"The embeddings model group {custom_config.get(MODEL_GROUP_ID_CONFIG_KEY)}"
            f" used by {log_source_component} component is inconsistent. "
            f"It contains different models: {model_names}. "
            f"Please verify your configuration."
        )
        structlogger.error(
            f"{log_source_function}.inconsistent_embeddings", event_info=error_message
        )
        print_error_and_exit(error_message)

    return model_names.pop() if len(model_names) > 0 else None
