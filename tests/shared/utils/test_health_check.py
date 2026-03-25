import os
from typing import Any, Dict
from unittest.mock import Mock, patch

from rasa.shared.utils.health_check.health_check import (
    HealthCheckPhase,
    _needs_model_probe,
    perform_embeddings_health_check,
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


# ===========================================================================
# HealthCheckPhase enum
# ===========================================================================


def test_health_check_phase_values() -> None:
    assert HealthCheckPhase.TRAIN.value == "train"
    assert HealthCheckPhase.INFERENCE.value == "inference"


# ===========================================================================
# _needs_model_probe
# ===========================================================================


def test_needs_model_probe_azure_deployment_only() -> None:
    """Azure client with deployment but no model → needs probe."""
    client = Mock(spec=_azure_llm_client_class())
    client._is_deployment_only = True
    client._extra_parameters = {}
    assert _needs_model_probe(client) is True


def test_needs_model_probe_azure_with_model() -> None:
    """Azure client with an explicit model → does NOT need probe."""
    client = Mock(spec=_azure_llm_client_class())
    client._is_deployment_only = False
    client._extra_parameters = {}
    assert _needs_model_probe(client) is False


def test_needs_model_probe_non_azure_client() -> None:
    """Non-Azure LLM client → does NOT need probe."""
    client = Mock()
    assert _needs_model_probe(client) is False


def test_needs_model_probe_azure_deployment_only_with_user_reasoning_effort() -> None:
    """Azure deployment-only but user already set reasoning_effort → skip probe."""
    client = Mock(spec=_azure_llm_client_class())
    client._is_deployment_only = True
    client._extra_parameters = {"reasoning_effort": "high"}
    assert _needs_model_probe(client) is False


def _azure_llm_client_class() -> type:
    """Return the real AzureOpenAILLMClient class for isinstance checks."""
    from rasa.shared.providers.llm.azure_openai_llm_client import (
        AzureOpenAILLMClient,
    )

    return AzureOpenAILLMClient


# ===========================================================================
# perform_llm_health_check — phase-aware behaviour
# ===========================================================================

_HC_MOD = "rasa.shared.utils.health_check.health_check"


def test_llm_hc_disabled_inference_azure_deployment_only_probes() -> None:
    """HC disabled + INFERENCE + Azure deployment-only → probe fires."""
    mock_client = Mock(_is_deployment_only=True, _extra_parameters={})
    mock_client.__class__ = _azure_llm_client_class()
    mock_instantiate = Mock(return_value=mock_client)
    mock_send = Mock()

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            {"provider": "azure", "deployment": "my-dep"},
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    assert mock_send.call_count == 1


def test_llm_hc_disabled_train_azure_deployment_only_no_probe() -> None:
    """HC disabled + TRAIN + Azure deployment-only → NO probe."""
    mock_client = Mock(_is_deployment_only=True, _extra_parameters={})
    mock_client.__class__ = _azure_llm_client_class()
    mock_instantiate = Mock(return_value=mock_client)
    mock_send = Mock()

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            {"provider": "azure", "deployment": "my-dep"},
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.TRAIN,
        )

    assert mock_send.call_count == 0


def test_llm_hc_disabled_inference_non_azure_no_probe() -> None:
    """HC disabled + INFERENCE + non-Azure client → NO probe."""
    mock_client = Mock()
    mock_instantiate = Mock(return_value=mock_client)
    mock_send = Mock()

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            {"model": "gpt-5.1-2025-11-13"},
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    assert mock_send.call_count == 0


def test_llm_hc_disabled_inference_azure_with_model_no_probe() -> None:
    """HC disabled + INFERENCE + Azure with explicit model → NO probe."""
    mock_client = Mock(_is_deployment_only=False)
    mock_client.__class__ = _azure_llm_client_class()
    mock_instantiate = Mock(return_value=mock_client)
    mock_send = Mock()

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            {
                "provider": "azure",
                "deployment": "my-dep",
                "model": "gpt-5.1-2025-11-13",
            },
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    assert mock_send.call_count == 0


def test_llm_hc_enabled_inference_azure_deployment_only_still_probes() -> None:
    """HC enabled + INFERENCE + Azure deployment-only → normal HC fires."""
    mock_client = Mock(_is_deployment_only=True, _extra_parameters={})
    mock_client.__class__ = _azure_llm_client_class()
    mock_instantiate = Mock(return_value=mock_client)
    mock_send = Mock()

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            {"provider": "azure", "deployment": "my-dep"},
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    assert mock_send.call_count == 1


def test_llm_hc_disabled_inference_azure_deploy_only_user_effort_no_probe() -> None:
    """HC disabled + INFERENCE + Azure deployment-only
    + user set reasoning_effort → NO probe.
    """
    mock_client = Mock(
        _is_deployment_only=True,
        _extra_parameters={"reasoning_effort": "high"},
    )
    mock_client.__class__ = _azure_llm_client_class()
    mock_instantiate = Mock(return_value=mock_client)
    mock_send = Mock()

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            {
                "provider": "azure",
                "deployment": "my-dep",
                "reasoning_effort": "high",
            },
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    assert mock_send.call_count == 0


def test_llm_hc_default_phase_is_train() -> None:
    """Existing tests pass without phase; default should be TRAIN."""
    mock_client = Mock(_is_deployment_only=True, _extra_parameters={})
    mock_client.__class__ = _azure_llm_client_class()
    mock_instantiate = Mock(return_value=mock_client)
    mock_send = Mock()

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        # No explicit phase argument — default is TRAIN
        perform_llm_health_check(
            {"provider": "azure", "deployment": "my-dep"},
            {},
            "test_fn",
            "TestComponent",
        )

    # Default phase is TRAIN, so no probe even for Azure deployment-only
    assert mock_send.call_count == 0


# ===========================================================================
# Router / model-group configs — inference probe
# ===========================================================================


def test_llm_hc_disabled_inference_router_probes_azure_only() -> None:
    """HC disabled + INFERENCE + router with mixed models.

    Only the Azure deployment-only model should be probed.
    """
    azure_deploy_client = Mock(_is_deployment_only=True, _extra_parameters={})
    azure_deploy_client.__class__ = _azure_llm_client_class()

    non_azure_client = Mock()

    # First call returns router client; subsequent calls return
    # individual clients for each model in the group.
    router_client = Mock()
    mock_instantiate = Mock(
        side_effect=[router_client, azure_deploy_client, non_azure_client]
    )
    mock_send = Mock()

    router_config = {
        "id": "test_group",
        "models": [
            {"provider": "azure", "deployment": "dep-a"},
            {"provider": "openai", "model": "gpt-5.1"},
        ],
    }

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            router_config,
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    # 1 for the router + 2 for individual models
    assert mock_instantiate.call_count == 3
    # Only the Azure deployment-only model gets probed
    assert mock_send.call_count == 1
    mock_send.assert_called_once_with(azure_deploy_client, "test_fn", "TestComponent")


def test_llm_hc_disabled_inference_router_no_azure_no_probe() -> None:
    """HC disabled + INFERENCE + router with no Azure deployment-only.

    No probes should fire; disabled warning should be logged.
    """
    non_azure_1 = Mock()
    non_azure_2 = Mock()

    router_client = Mock()
    mock_instantiate = Mock(side_effect=[router_client, non_azure_1, non_azure_2])
    mock_send = Mock()

    router_config = {
        "id": "test_group",
        "models": [
            {"provider": "openai", "model": "gpt-5.1"},
            {"provider": "openai", "model": "gpt-5-mini"},
        ],
    }

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            router_config,
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    # 1 for the router + 2 for individual models
    assert mock_instantiate.call_count == 3
    # No Azure deployment-only → no probes
    assert mock_send.call_count == 0


def test_llm_hc_enabled_router_all_models_checked() -> None:
    """HC enabled + router: every model gets a health check regardless."""
    client_a = Mock()
    client_b = Mock()

    router_client = Mock()
    mock_instantiate = Mock(side_effect=[router_client, client_a, client_b])
    mock_send = Mock()

    router_config = {
        "id": "test_group",
        "models": [
            {"provider": "openai", "model": "gpt-5.1"},
            {"provider": "azure", "deployment": "dep-a"},
        ],
    }

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            router_config,
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    assert mock_instantiate.call_count == 3
    assert mock_send.call_count == 2


def test_llm_hc_disabled_train_router_azure_no_probe() -> None:
    """HC disabled + TRAIN + router with Azure deployment-only.

    TRAIN phase never probes, even for Azure deployment-only models.
    """
    azure_client = Mock(_is_deployment_only=True, _extra_parameters={})
    azure_client.__class__ = _azure_llm_client_class()

    router_client = Mock()
    mock_instantiate = Mock(side_effect=[router_client, azure_client])
    mock_send = Mock()

    router_config = {
        "id": "test_group",
        "models": [
            {"provider": "azure", "deployment": "dep-a"},
            {"provider": "azure", "deployment": "dep-b"},
        ],
    }

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            router_config,
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.TRAIN,
        )

    # Only the top-level router client is instantiated; no iteration
    assert mock_instantiate.call_count == 1
    assert mock_send.call_count == 0


def test_llm_hc_disabled_inference_router_all_azure_all_probed() -> None:
    """HC disabled + INFERENCE + router where all models are Azure
    deployment-only. Every entry should be probed.
    """
    azure_a = Mock(_is_deployment_only=True, _extra_parameters={})
    azure_a.__class__ = _azure_llm_client_class()

    azure_b = Mock(_is_deployment_only=True, _extra_parameters={})
    azure_b.__class__ = _azure_llm_client_class()

    router_client = Mock()
    mock_instantiate = Mock(side_effect=[router_client, azure_a, azure_b])
    mock_send = Mock()

    router_config = {
        "id": "test_group",
        "models": [
            {"provider": "azure", "deployment": "dep-a"},
            {"provider": "azure", "deployment": "dep-b"},
        ],
    }

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(f"{_HC_MOD}.try_instantiate_llm_client", mock_instantiate),
        patch(f"{_HC_MOD}.send_test_llm_api_request", mock_send),
    ):
        perform_llm_health_check(
            router_config,
            {},
            "test_fn",
            "TestComponent",
            phase=HealthCheckPhase.INFERENCE,
        )

    assert mock_instantiate.call_count == 3
    assert mock_send.call_count == 2
