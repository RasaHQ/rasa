from typing import Dict, Any

import pytest
from unittest.mock import patch, Mock
from _pytest.capture import CaptureFixture

from rasa.shared.utils.health_check.health_check import (
    perform_embeddings_model_group_consistency_check,
    perform_llm_model_group_consistency_check,
    perform_training_time_embeddings_health_check,
    perform_inference_time_llm_health_check,
)
from rasa.shared.utils.health_check.health_check import (
    perform_inference_time_embeddings_health_check,
)
import os
from rasa.shared.utils.health_check.health_check import (
    perform_training_time_llm_health_check,
)


def test_perform_embeddings_model_group_consistency_check_success() -> None:
    custom_config = {
        "id": "test_model_group",
        "models": [
            {"model_1": {"provider": "azure", "deployment": "my_deployment_1"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_2"}},
        ],
    }
    default_config: Dict[str, Any] = {}
    log_source_function = "test_function"
    log_source_component = "test_component"

    mock_try_instantiate_embedder = Mock()
    mock_send_test_embeddings_api_request = Mock(return_value="consistent_model")

    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder",
            mock_try_instantiate_embedder,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request",
            mock_send_test_embeddings_api_request,
        ),
    ):
        retrieved_model_name = perform_embeddings_model_group_consistency_check(
            custom_config, default_config, log_source_function, log_source_component
        )

    assert retrieved_model_name == "consistent_model"
    assert mock_try_instantiate_embedder.call_count == 2
    assert mock_send_test_embeddings_api_request.call_count == 2


def test_perform_embeddings_model_group_consistency_check_inconsistent_models(
    capsys: CaptureFixture,
) -> None:
    custom_config = {
        "id": "test_model_group",
        "models": [
            {"model_1": {"provider": "azure", "deployment": "my_deployment_1"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_2"}},
        ],
    }
    default_config: Dict[str, Any] = {}
    log_source_function = "test_function"
    log_source_component = "test_component"

    mock_try_instantiate_embedder = Mock()
    mock_send_test_embeddings_api_request = Mock()
    mock_send_test_embeddings_api_request.side_effect = ["model_1", "model_2"]
    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder",
            mock_try_instantiate_embedder,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request",
            mock_send_test_embeddings_api_request,
        ),
    ):
        with pytest.raises(SystemExit):
            perform_embeddings_model_group_consistency_check(
                custom_config, default_config, log_source_function, log_source_component
            )

    assert mock_try_instantiate_embedder.call_count == 2
    assert mock_send_test_embeddings_api_request.call_count == 2
    captured = capsys.readouterr()
    expected_error = (
        "The embeddings model group test_model_group used by "
        "test_component component is inconsistent. It contains different models"
    )
    assert expected_error in captured.out


def test_perform_llm_model_group_consistency_check_success() -> None:
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
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client",
            mock_try_instantiate_llm_client,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request",
            mock_send_test_llm_api_request,
        ),
    ):
        retrieved_model_name = perform_llm_model_group_consistency_check(
            custom_config, default_config, log_source_function, log_source_component
        )

    assert retrieved_model_name == "consistent_model"
    assert mock_try_instantiate_llm_client.call_count == 2
    assert mock_send_test_llm_api_request.call_count == 2


def test_perform_llm_model_group_consistency_check_inconsistent_models(
    capsys: CaptureFixture,
) -> None:
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
    mock_send_test_llm_api_request = Mock()
    mock_send_test_llm_api_request.side_effect = ["model_1", "model_2"]
    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client",
            mock_try_instantiate_llm_client,
        ),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request",
            mock_send_test_llm_api_request,
        ),
    ):
        with pytest.raises(SystemExit):
            perform_llm_model_group_consistency_check(
                custom_config, default_config, log_source_function, log_source_component
            )

    assert mock_try_instantiate_llm_client.call_count == 2
    assert mock_send_test_llm_api_request.call_count == 2
    captured = capsys.readouterr()
    expected_error = (
        "The model group test_model_group used by test_component component "
        "is inconsistent. It contains different models"
    )
    assert expected_error in captured.out


def test_perform_training_time_llm_health_check_env_var_not_set():
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client"
        ) as mock_try_instantiate_llm_client,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request"
        ) as mock_send_test_llm_api_request,
    ):
        model_name = perform_training_time_llm_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

        assert model_name is None
        mock_try_instantiate_llm_client.assert_called_once()
        mock_send_test_llm_api_request.assert_not_called()


def test_perform_training_time_llm_health_check_success():
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client"
        ) as mock_try_instantiate_llm_client,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request"
        ) as mock_send_test_llm_api_request,
    ):
        mock_try_instantiate_llm_client.return_value = Mock()
        mock_send_test_llm_api_request.return_value = "consistent_model"

        model_name = perform_training_time_llm_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

        assert model_name == "consistent_model"
        mock_try_instantiate_llm_client.assert_called_once()
        mock_send_test_llm_api_request.assert_called_once()


def test_perform_training_time_llm_health_check_inconsistent_models(
    capsys: CaptureFixture,
):
    custom_config = {
        "id": "test_model_group",
        "models": [
            {"model_1": {"provider": "azure", "deployment": "my_deployment_1"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_2"}},
        ],
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client"
        ) as mock_try_instantiate_llm_client,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request"
        ) as mock_send_test_llm_api_request,
    ):
        mock_try_instantiate_llm_client.return_value = Mock()
        mock_send_test_llm_api_request.side_effect = ["model_1", "model_2"]

        with pytest.raises(SystemExit):
            perform_training_time_llm_health_check(
                custom_config, default_config, log_source_function, log_source_component
            )

        # two calls for each model of model group, additional call for the whole config
        assert mock_try_instantiate_llm_client.call_count == 3
        assert mock_send_test_llm_api_request.call_count == 2
        captured = capsys.readouterr()
        expected_error = (
            "The model group test_model_group used by test_component component "
            "is inconsistent. It contains different models"
        )
        assert expected_error in captured.out


def test_perform_training_time_embeddings_health_check_env_var_not_set():
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "false"}),
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request"
        ) as mock_send_test_embeddings_api_request,
    ):
        model_name = perform_training_time_embeddings_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

        assert model_name is None
        mock_send_test_embeddings_api_request.assert_not_called()


def test_perform_training_time_embeddings_health_check_success():
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder"
        ) as mock_try_instantiate_embedder,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request"
        ) as mock_send_test_embeddings_api_request,
    ):
        mock_try_instantiate_embedder.return_value = Mock()
        mock_send_test_embeddings_api_request.return_value = "consistent_model"

        model_name = perform_training_time_embeddings_health_check(
            custom_config, default_config, log_source_function, log_source_component
        )

        assert model_name == "consistent_model"
        mock_try_instantiate_embedder.assert_called_once()
        mock_send_test_embeddings_api_request.assert_called_once()


def test_perform_training_time_embeddings_health_check_inconsistent_models(
    capsys: CaptureFixture,
):
    custom_config = {
        "id": "test_model_group",
        "models": [
            {"model_1": {"provider": "azure", "deployment": "my_deployment_1"}},
            {"model_2": {"provider": "azure", "deployment": "my_deployment_2"}},
        ],
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"

    with (
        patch.dict(os.environ, {"LLM_API_HEALTH_CHECK": "true"}),
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder"
        ) as mock_try_instantiate_embedder,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request"
        ) as mock_send_test_embeddings_api_request,
    ):
        mock_try_instantiate_embedder.return_value = Mock()
        mock_send_test_embeddings_api_request.side_effect = ["model_1", "model_2"]

        with pytest.raises(SystemExit):
            perform_training_time_embeddings_health_check(
                custom_config, default_config, log_source_function, log_source_component
            )

        assert mock_try_instantiate_embedder.call_count == 2
        assert mock_send_test_embeddings_api_request.call_count == 2
        captured = capsys.readouterr()
        expected_error = (
            "The embeddings model group test_model_group used by test_component "
            "component is inconsistent. It contains different models"
        )
        assert expected_error in captured.out


def test_perform_inference_time_llm_health_check_env_var_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"
    train_model_name = "consistent_model"

    monkeypatch.setenv("LLM_API_HEALTH_CHECK", "false")

    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client"
        ) as mock_try_instantiate_llm_client,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request"
        ) as mock_send_test_llm_api_request,
    ):
        perform_inference_time_llm_health_check(
            custom_config,
            default_config,
            train_model_name,
            log_source_function,
            log_source_component,
        )

        mock_try_instantiate_llm_client.assert_not_called()
        mock_send_test_llm_api_request.assert_not_called()


def test_perform_inference_time_llm_health_check_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"
    train_model_name = "consistent_model"

    monkeypatch.setenv("LLM_API_HEALTH_CHECK", "true")

    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client"
        ) as mock_try_instantiate_llm_client,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request"
        ) as mock_send_test_llm_api_request,
    ):
        mock_try_instantiate_llm_client.return_value = Mock()
        mock_send_test_llm_api_request.return_value = "consistent_model"

        perform_inference_time_llm_health_check(
            custom_config,
            default_config,
            train_model_name,
            log_source_function,
            log_source_component,
        )

        mock_try_instantiate_llm_client.assert_called_once()
        mock_send_test_llm_api_request.assert_called_once()


def test_perform_inference_time_llm_health_check_inconsistent_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture,
) -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"
    train_model_name = "model_1"

    monkeypatch.setenv("LLM_API_HEALTH_CHECK", "true")

    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client"
        ) as mock_try_instantiate_llm_client,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request"
        ) as mock_send_test_llm_api_request,
    ):
        mock_try_instantiate_llm_client.return_value = Mock()
        mock_send_test_llm_api_request.return_value = "model_2"

        with pytest.raises(SystemExit):
            perform_inference_time_llm_health_check(
                custom_config,
                default_config,
                train_model_name,
                log_source_function,
                log_source_component,
            )

        mock_try_instantiate_llm_client.assert_called_once()
        mock_send_test_llm_api_request.assert_called_once()
        captured = capsys.readouterr()
        expected_error = (
            "The LLM used to train the test_component (model_1) is not the same as "
            "the LLM used for inference (model_2)"
        )
        assert expected_error in captured.out


def test_perform_inference_time_llm_health_check_no_train_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"
    train_model_name = None

    monkeypatch.setenv("LLM_API_HEALTH_CHECK", "true")

    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_llm_client"
        ) as mock_try_instantiate_llm_client,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_llm_api_request"
        ) as mock_send_test_llm_api_request,
    ):
        mock_try_instantiate_llm_client.return_value = Mock()
        mock_send_test_llm_api_request.return_value = "consistent_model"

        perform_inference_time_llm_health_check(
            custom_config,
            default_config,
            train_model_name,
            log_source_function,
            log_source_component,
        )

        mock_try_instantiate_llm_client.assert_called_once()
        mock_send_test_llm_api_request.assert_called_once()


def test_perform_inference_time_embeddings_health_check_env_var_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"
    train_model_name = "consistent_model"

    monkeypatch.setenv("LLM_API_HEALTH_CHECK", "false")

    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder"
        ) as mock_try_instantiate_embedder,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request"
        ) as mock_send_test_embeddings_api_request,
    ):
        perform_inference_time_embeddings_health_check(
            custom_config,
            default_config,
            train_model_name,
            log_source_function,
            log_source_component,
        )

        mock_try_instantiate_embedder.assert_not_called()
        mock_send_test_embeddings_api_request.assert_not_called()


def test_perform_inference_time_embeddings_health_check_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"
    train_model_name = "consistent_model"

    monkeypatch.setenv("LLM_API_HEALTH_CHECK", "true")

    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder"
        ) as mock_try_instantiate_embedder,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request"
        ) as mock_send_test_embeddings_api_request,
    ):
        mock_try_instantiate_embedder.return_value = Mock()
        mock_send_test_embeddings_api_request.return_value = "consistent_model"

        perform_inference_time_embeddings_health_check(
            custom_config,
            default_config,
            train_model_name,
            log_source_function,
            log_source_component,
        )

        mock_try_instantiate_embedder.assert_called_once()
        mock_send_test_embeddings_api_request.assert_called_once()


def test_perform_inference_time_embeddings_health_check_inconsistent_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture,
) -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"
    train_model_name = "model_1"

    monkeypatch.setenv("LLM_API_HEALTH_CHECK", "true")
    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder"
        ) as mock_try_instantiate_embedder,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request"
        ) as mock_send_test_embeddings_api_request,
    ):
        mock_try_instantiate_embedder.return_value = Mock()
        mock_send_test_embeddings_api_request.return_value = "model_2"

        with pytest.raises(SystemExit):
            perform_inference_time_embeddings_health_check(
                custom_config,
                default_config,
                train_model_name,
                log_source_function,
                log_source_component,
            )

        mock_try_instantiate_embedder.assert_called_once()
        mock_send_test_embeddings_api_request.assert_called_once()
        captured = capsys.readouterr()
        expected_error = (
            "The Embeddings model used to train the test_component (model_1) is "
            "not the same as the model used for inference (model_2)"
        )
        assert expected_error in captured.out


def test_perform_inference_time_embeddings_health_check_no_train_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    default_config = {
        "provider": "azure",
        "deployment": "my_deployment_1",
    }
    log_source_function = "test_function"
    log_source_component = "test_component"
    train_model_name = None

    monkeypatch.setenv("LLM_API_HEALTH_CHECK", "true")

    with (
        patch(
            "rasa.shared.utils.health_check.health_check.try_instantiate_embedder"
        ) as mock_try_instantiate_embedder,
        patch(
            "rasa.shared.utils.health_check.health_check.send_test_embeddings_api_request"
        ) as mock_send_test_embeddings_api_request,
    ):
        mock_try_instantiate_embedder.return_value = Mock()
        mock_send_test_embeddings_api_request.return_value = "consistent_model"

        perform_inference_time_embeddings_health_check(
            custom_config,
            default_config,
            train_model_name,
            log_source_function,
            log_source_component,
        )

        mock_try_instantiate_embedder.assert_called_once()
        mock_send_test_embeddings_api_request.assert_called_once()
