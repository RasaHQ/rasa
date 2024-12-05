from typing import Dict, Any

from unittest.mock import patch, Mock

from rasa.shared.utils.health_check.health_check import (
    perform_embeddings_health_check,
)
import os
from rasa.shared.utils.health_check.health_check import (
    perform_llm_health_check,
)


def test_embeddings_health_check_disabled() -> None:
    custom_config = {"provider": "azure", "deployment": "my_deployment_1"}
    default_config: Dict[str, Any] = {}
    log_source_function = "test_function"
    log_source_component = "test_component"

    mock_try_instantiate_embedder = Mock()
    mock_send_test_embeddings_api_request = Mock(return_value="consistent_model")

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder",
            mock_try_instantiate_embedder,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request",
            mock_send_test_embeddings_api_request,
        ),
    ):
        perform_embeddings_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

    assert mock_try_instantiate_embedder.call_count == 1
    assert mock_send_test_embeddings_api_request.call_count == 0


def test_embeddings_health_check_single_model() -> None:
    custom_config = {"provider": "azure", "deployment": "my_deployment_1"}
    default_config: Dict[str, Any] = {}
    log_source_function = "test_function"
    log_source_component = "test_component"

    mock_try_instantiate_embedder = Mock()
    mock_send_test_embeddings_api_request = Mock(return_value="consistent_model")

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder",
            mock_try_instantiate_embedder,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request",
            mock_send_test_embeddings_api_request,
        ),
    ):
        perform_embeddings_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

    assert mock_try_instantiate_embedder.call_count == 1
    assert mock_send_test_embeddings_api_request.call_count == 1


def test_embeddings_health_check_multiple_models() -> None:
    custom_config = {
        "id": "test_model_group",
        "models": [
            {"model_1": {"provider": "azure", "deployment": "my_deployment_1"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_2"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_3"}},
        ],
    }
    default_config: Dict[str, Any] = {}
    log_source_function = "test_function"
    log_source_component = "test_component"

    mock_try_instantiate_embedder = Mock()
    mock_send_test_embeddings_api_request = Mock(return_value="consistent_model")

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder",
            mock_try_instantiate_embedder,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request",
            mock_send_test_embeddings_api_request,
        ),
    ):
        perform_embeddings_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

    # one instantiation for the multiple models config and 3 for each modelseparately
    assert mock_try_instantiate_embedder.call_count == 4
    # one call for each model
    assert mock_send_test_embeddings_api_request.call_count == 3


def test_llm_health_check_disabled() -> None:
    custom_config = {
        "id": "test_model_group",
        "models": [
            {"model_1": {"provider": "azure", "deployment": "my_deployment_1"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_2"}},
        ],
    }
    default_config: Dict[str, Any] = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    mock_try_instantiate_llm_client = Mock()
    mock_send_test_llm_api_request = Mock(return_value="consistent_model")
    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client",
            mock_try_instantiate_llm_client,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request",
            mock_send_test_llm_api_request,
        ),
    ):
        perform_llm_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

    assert mock_try_instantiate_llm_client.call_count == 1
    assert mock_send_test_llm_api_request.call_count == 0


def test_llm_health_check_single_model() -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config: Dict[str, Any] = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    mock_try_instantiate_llm_client = Mock()
    mock_send_test_llm_api_request = Mock(return_value="consistent_model")
    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client",
            mock_try_instantiate_llm_client,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request",
            mock_send_test_llm_api_request,
        ),
    ):
        perform_llm_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

    assert mock_try_instantiate_llm_client.call_count == 1
    assert mock_send_test_llm_api_request.call_count == 1


def test_llm_health_check_multiple_models() -> None:
    custom_config = {
        "id": "test_model_group",
        "models": [
            {"model_1": {"provider": "azure", "deployment": "my_deployment_1"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_2"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_3"}},
        ],
    }
    default_config: Dict[str, Any] = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    mock_try_instantiate_llm_client = Mock()
    mock_send_test_llm_api_request = Mock(return_value="consistent_model")
    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client",
            mock_try_instantiate_llm_client,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request",
            mock_send_test_llm_api_request,
        ),
    ):
        perform_llm_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

    # one instantiation for the multiple models config and 3 for each model separately
    assert mock_try_instantiate_llm_client.call_count == 4
    # one call for each model
    assert mock_send_test_llm_api_request.call_count == 3
