import pytest

from rasa.shared.providers._configs.model_group_config import (
    ModelGroupConfig,
    ModelConfig,
)


def test_model_group_config_init() -> None:
    model_group_config = ModelGroupConfig(
        model_group_id="test-model-group-id",
        models=[
            ModelConfig.from_dict(
                {"provider": "cohere", "model": "cohere/test-cohere"}
            ),
            ModelConfig.from_dict({"provider": "openai", "model": "openai/gpt-4"}),
        ],
    )
    assert model_group_config.model_group_id == "test-model-group-id"
    assert model_group_config.models[0].provider == "cohere"
    assert model_group_config.models[0].model == "cohere/test-cohere"
    assert model_group_config.models[1].provider == "openai"
    assert model_group_config.models[1].model == "openai/gpt-4"


def test_model_group_config_post_init_raises_error_missing_model_group_id() -> None:
    with pytest.raises(ValueError):
        ModelGroupConfig(
            model_group_id=None,
            models=[
                ModelConfig.from_dict(
                    {"provider": "cohere", "model": "cohere/test-cohere"}
                ),
                ModelConfig.from_dict({"provider": "openai", "model": "openai/gpt-4"}),
            ],
        )


def test_model_group_config_post_init_raises_error_missing_models() -> None:
    with pytest.raises(ValueError):
        ModelGroupConfig(model_group_id="test-model-group-id", models=[])


@pytest.mark.parametrize(
    "model_config,"
    "expected_provider,"
    "expected_model,"
    "expected_deployment,"
    "expected_api_base,"
    "expected_api_version,"
    "expected_extra_parameters",
    [
        (
            {
                "provider": "cohere",
                "model": "cohere/test-cohere",
                "timeout": 0.7,
                "num_retries": 7,
            },
            "cohere",
            "cohere/test-cohere",
            None,
            None,
            None,
            {"timeout": 0.7, "num_retries": 7},
        ),
        (
            {
                "provider": "openai",
                "model": "openai/gpt-4",
                "timeout": 0.7,
                "num_retries": 7,
            },
            "openai",
            "openai/gpt-4",
            None,
            None,
            None,
            {"timeout": 0.7, "num_retries": 7},
        ),
        (
            {
                "provider": "azure",
                "deployment": "my-test-gpt-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "timeout": 0.7,
                "num_retries": 7,
            },
            "azure",
            None,
            "my-test-gpt-deployment-on-azure",
            "https://my-test-base",
            "v1",
            {"timeout": 0.7, "num_retries": 7},
        ),
        # Deprecated openai configuration
        (
            {
                "_type": "openai",
                "model_name": "gpt-4",
                "request_timeout": 0.7,
            },
            "openai",
            "gpt-4",
            None,
            None,
            None,
            {"timeout": 0.7},
        ),
        # Deprecated azure configuration
        (
            {
                "_type": "azure",
                "deployment_name": "azure/my-test-gpt-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
            },
            "azure",
            None,
            "azure/my-test-gpt-deployment-on-azure",
            "https://my-test-base",
            "v1",
            {},
        ),
        (
            {
                "type": "azure",
                "engine": "my-test-gpt-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
            },
            "azure",
            None,
            "my-test-gpt-deployment-on-azure",
            "https://my-test-base",
            "v1",
            {},
        ),
    ],
)
def test_model_group_config_from_dict(
    model_config: dict,
    expected_provider: str,
    expected_model: str,
    expected_deployment: str,
    expected_api_base: str,
    expected_api_version: str,
    expected_extra_parameters: dict,
):
    # Given
    model_group_config = {"id": "test-model-group-id", "models": [model_config]}

    # When
    model_group_config_object = ModelGroupConfig.from_dict(model_group_config)

    # Then
    assert model_group_config_object.model_group_id == "test-model-group-id"
    assert len(model_group_config_object.models) == 1

    assert model_group_config_object.models[0].provider == expected_provider
    assert model_group_config_object.models[0].model == expected_model
    assert model_group_config_object.models[0].deployment == expected_deployment
    assert model_group_config_object.models[0].api_base == expected_api_base
    assert model_group_config_object.models[0].api_version == expected_api_version
    assert (
        model_group_config_object.models[0].extra_parameters
        == expected_extra_parameters
    )


@pytest.mark.parametrize(
    "given_model_config," "expected_model_config",
    [
        (
            {
                "provider": "cohere",
                "model": "cohere/test-cohere",
                "timeout": 0.7,
                "num_retries": 7,
            },
            {
                "provider": "cohere",
                "model": "cohere/test-cohere",
                "timeout": 0.7,
                "num_retries": 7,
            },
        ),
        (
            {
                "provider": "openai",
                "model": "openai/gpt-4",
                "timeout": 0.7,
                "num_retries": 7,
            },
            {
                "provider": "openai",
                "model": "openai/gpt-4",
                "timeout": 0.7,
                "num_retries": 7,
                # legacy, automatically set
                "api_type": "openai",
            },
        ),
        (
            {
                "provider": "azure",
                "deployment": "my-test-gpt-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "timeout": 0.7,
                "num_retries": 7,
            },
            {
                "provider": "azure",
                "deployment": "my-test-gpt-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "timeout": 0.7,
                "num_retries": 7,
                # legacy, automatically set
                "api_type": "azure",
            },
        ),
        # Deprecated openai configuration
        (
            {
                "_type": "openai",
                "model_name": "gpt-4",
                "request_timeout": 0.7,
            },
            {
                "provider": "openai",
                "model": "gpt-4",
                "timeout": 0.7,
                # legacy, automatically set
                "api_type": "openai",
            },
        ),
        # Deprecated azure configuration
        (
            {
                "_type": "azure",
                "deployment_name": "azure/my-test-gpt-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "azure/my-test-gpt-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        (
            {
                "type": "azure",
                "engine": "my-test-gpt-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-gpt-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
    ],
)
def test_model_group_config_to_dict(given_model_config, expected_model_config):
    # Given
    model_group_config = {"id": "test-model-group-id", "models": [given_model_config]}

    # When
    model_group_config_object = ModelGroupConfig.from_dict(model_group_config).to_dict()
    parsed_model_config = model_group_config_object["models"][0]

    # Then
    assert parsed_model_config == expected_model_config


def test_model_config_from_dict_and_to_dict_interoperability():
    # Given
    config = {"provider": "cohere", "model": "test-cohere"}

    # When
    model_config_object = ModelConfig.from_dict(config)
    parsed_config = model_config_object.to_dict()
    model_config_object_again = ModelConfig.from_dict(parsed_config)

    # Then
    assert model_config_object_again.provider == "cohere"
    assert model_config_object_again.model == "test-cohere"


def test_model_group_config_from_dict_and_to_dict_interoperability():
    # Given
    config = {
        "id": "test-model-group-id",
        "models": [
            {"provider": "cohere", "model": "test-cohere"},
            {"provider": "openai", "model": "test-gpt"},
        ],
    }

    # When
    model_group_config_object = ModelGroupConfig.from_dict(config)
    parsed_config = model_group_config_object.to_dict()
    model_group_config_object_again = ModelGroupConfig.from_dict(parsed_config)

    # Then
    assert model_group_config_object_again.model_group_id == "test-model-group-id"
    assert model_group_config_object_again.models[0].provider == "cohere"
    assert model_group_config_object_again.models[0].model == "test-cohere"
    assert model_group_config_object_again.models[1].provider == "openai"
    assert model_group_config_object_again.models[1].model == "test-gpt"
