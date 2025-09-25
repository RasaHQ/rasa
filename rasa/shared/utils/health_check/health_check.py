import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from rasa.exceptions import HealthCheckError

if TYPE_CHECKING:
    pass
from rasa.shared.constants import (
    LLM_API_HEALTH_CHECK_DEFAULT_VALUE,
    LLM_API_HEALTH_CHECK_ENV_VAR,
    MODELS_CONFIG_KEY,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.utils.llm import embedder_factory, llm_factory, structlogger


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
        raise HealthCheckError(
            code=f"{log_source_function}.llm_instantiation_failed",
            event_info=(
                f"Unable to create the LLM client for component - "
                f"{log_source_component}. "
                f"Please make sure you specified the required environment variables "
                f"and configuration keys. "
            ),
            error=e,
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
        raise HealthCheckError(
            code=f"{log_source_function}.embedder_instantiation_failed",
            event_info=(
                f"Unable to create the Embedding client for component - "
                f"{log_source_component}. Please make sure you specified the required "
                f"environment variables and configuration keys."
            ),
            error=e,
        )


def perform_llm_health_check(
    custom_config: Optional[Dict[str, Any]],
    default_config: Dict[str, Any],
    log_source_function: str,
    log_source_component: str,
) -> None:
    """Try to instantiate the LLM Client to validate the provided config.

    If the LLM_API_HEALTH_CHECK environment variable is true, perform a test call
    to the LLM API. If config contains multiple models, perform a test call for each
    model in the model group.

    This method supports both single model configurations and model group configurations
    (configs that have the `models` key).
    """
    # Instantiate the LLM client or Router LLM client to validate the provided config.
    llm_client = try_instantiate_llm_client(
        custom_config, default_config, log_source_function, log_source_component
    )

    if is_api_health_check_enabled():
        if (
            custom_config
            and MODELS_CONFIG_KEY in custom_config
            and len(custom_config[MODELS_CONFIG_KEY]) > 1
        ):
            # If the config uses a router, instantiate the LLM client for each model
            # in the model group. This is required to perform a test api call for each
            # model in the group.
            # Note: The Router LLM client is not used here as we need to perform a test
            # api call and not load balance the requests.
            for model_config in custom_config[MODELS_CONFIG_KEY]:
                llm_client = try_instantiate_llm_client(
                    model_config,
                    default_config,
                    log_source_function,
                    log_source_component,
                )
                send_test_llm_api_request(
                    llm_client, log_source_function, log_source_component
                )
        else:
            # Make a test api call to perform a health check for the LLM client.
            # LLM config from config file and model group config from endpoint config
            # without router are handled here.
            send_test_llm_api_request(
                llm_client,
                log_source_function,
                log_source_component,
            )
    else:
        structlogger.warning(
            f"{log_source_function}.perform_llm_health_check.disabled",
            event_info=(
                f"The {LLM_API_HEALTH_CHECK_ENV_VAR} environment variable is set "
                f"to false, which will disable LLM health check. "
                f"It is recommended to set this variable to true in production "
                f"environments."
            ),
        )
        return None


def perform_embeddings_health_check(
    custom_config: Optional[Dict[str, Any]],
    default_config: Dict[str, Any],
    log_source_function: str,
    log_source_component: str,
) -> None:
    """Try to instantiate the Embedder to validate the provided config.

    If the LLM_API_HEALTH_CHECK environment variable is true, perform a test call
    to the Embeddings API. If config contains multiple models, perform a test call for
    each model in the model group.

    This method supports both single model configurations and model group configurations
    (configs that have the `models` key).
    """
    # Instantiate the Embedder client or the Embedder Router client to validate the
    # provided config. Deprecation warnings and errors are logged here.
    embedder = try_instantiate_embedder(
        custom_config, default_config, log_source_function, log_source_component
    )

    if is_api_health_check_enabled():
        if (
            custom_config
            and MODELS_CONFIG_KEY in custom_config
            and len(custom_config[MODELS_CONFIG_KEY]) > 1
        ):
            # If the config uses a router, instantiate the Embedder client for each
            # model in the model group. This is required to perform a test api call
            # for every model in the group.
            # Note: The Router Embedding client is not used here as we need to perform
            # a test API call and not load balance the requests.
            for model_config in custom_config[MODELS_CONFIG_KEY]:
                embedder = try_instantiate_embedder(
                    model_config,
                    default_config,
                    log_source_function,
                    log_source_component,
                )
                send_test_embeddings_api_request(
                    embedder, log_source_function, log_source_component
                )
        else:
            # Make a test api call to perform a health check for the Embedding client.
            # Embeddings config from config file and model group config from endpoint
            # config without router are handled here.
            send_test_embeddings_api_request(
                embedder, log_source_function, log_source_component
            )
    else:
        structlogger.warning(
            f"{log_source_function}" f".perform_embeddings_health_check.disabled",
            event_info=(
                f"The {LLM_API_HEALTH_CHECK_ENV_VAR} environment variable is set "
                f"to false, which will disable embeddings API health check. "
                f"It is recommended to set this variable to true in production "
                f"environments."
            ),
        )
        return None


def send_test_llm_api_request(
    llm_client: LLMClient, log_source_function: str, log_source_component: str
) -> None:
    """Sends a test request to the LLM API to perform a health check.

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
        llm_client.completion("hello")
    except Exception as e:
        raise HealthCheckError(
            code=f"{log_source_function}.send_test_llm_api_request_failed",
            event_info=(
                f"Test call to the LLM API failed for component - "
                f"{log_source_component}."
            ),
            config=llm_client.config,
            error=e,
        )


def send_test_embeddings_api_request(
    embedder: EmbeddingClient, log_source_function: str, log_source_component: str
) -> None:
    """Sends a test request to the Embeddings API to perform a health check.

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
        embedder.embed(["hello"])
    except Exception as e:
        raise HealthCheckError(
            code=f"{log_source_function}.send_test_llm_api_request_failed",
            event_info=(
                f"Test call to the Embeddings API failed for component - "
                f"{log_source_component}."
            ),
            config=embedder.config,
            error=e,
        )


def is_api_health_check_enabled() -> bool:
    """Determines whether the API health check is enabled.

    Returns:
        bool: True if the API health check is enabled, False otherwise.
    """
    return (
        os.getenv(
            LLM_API_HEALTH_CHECK_ENV_VAR, LLM_API_HEALTH_CHECK_DEFAULT_VALUE
        ).lower()
        == "true"
    )
