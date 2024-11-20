import pytest

from rasa.shared.providers._configs.litellm_router_client_config import (
    LiteLLMRouterClientConfig,
)
from rasa.shared.providers._configs.model_group_config import (
    ModelGroupConfig,
    ModelConfig,
)


class TestLiteLLMRouterClientConfig:
    def test_init(self):
        router_client_config = LiteLLMRouterClientConfig(
            _model_group_config=ModelGroupConfig.from_dict(
                {
                    "id": "test-model-group-id",
                    "models": [
                        {"provider": "cohere", "model": "cohere/test-cohere"},
                        {"provider": "openai", "model": "openai/gpt-4"},
                    ],
                },
            ),
            router={"routing_strategy": "test"},
        )
        assert router_client_config.model_group_id == "test-model-group-id"
        assert router_client_config.models == [
            ModelConfig.from_dict(
                {"provider": "cohere", "model": "cohere/test-cohere"}
            ),
            ModelConfig.from_dict({"provider": "openai", "model": "openai/gpt-4"}),
        ]
        assert router_client_config.router == {"routing_strategy": "test"}

    def test_init_raises_error_missing_model_group_id(self):
        with pytest.raises(ValueError):
            LiteLLMRouterClientConfig(
                _model_group_config=ModelGroupConfig.from_dict(
                    {
                        "models": [
                            {"provider": "cohere", "model": "cohere/test-cohere"},
                            {"provider": "openai", "model": "openai/gpt-4"},
                        ],
                    },
                ),
                router={"routing_strategy": "test"},
            )

    def test_init_raises_error_missing_models(self):
        with pytest.raises(ValueError):
            LiteLLMRouterClientConfig(
                _model_group_config=ModelGroupConfig.from_dict(
                    {
                        "id": "test-model-group-id",
                    },
                ),
                router={"routing_strategy": "test"},
            )

    def test_init_raises_error_missing_router_settings(self):
        with pytest.raises(ValueError):
            LiteLLMRouterClientConfig(
                _model_group_config=ModelGroupConfig.from_dict(
                    {
                        "id": "test-model-group-id",
                        "models": [
                            {"provider": "cohere", "model": "cohere/test-cohere"},
                            {"provider": "openai", "model": "openai/gpt-4"},
                        ],
                    },
                ),
                router={},
            )

    def test_from_dict(self):
        router_client_config = LiteLLMRouterClientConfig.from_dict(
            {
                "id": "test-model-group-id",
                "models": [
                    {"provider": "cohere", "model": "test-cohere"},
                    {"provider": "openai", "model": "gpt-4"},
                    {"provider": "azure", "deployment": "test-deployment"},
                    {
                        "provider": "azure",
                        "deployment": "test-deployment",
                        "api_base": "test-api-base",
                        "api_type": "azure",
                        "api_version": "test-api-version",
                        "timeout": 10,
                    },
                ],
                "router": {"routing_strategy": "test"},
            }
        )

        assert router_client_config.model_group_id == "test-model-group-id"
        assert router_client_config.router == {"routing_strategy": "test"}

        assert len(router_client_config.models) == 4

        assert router_client_config.models[0].provider == "cohere"
        assert router_client_config.models[0].model == "test-cohere"

        assert router_client_config.models[1].provider == "openai"
        assert router_client_config.models[1].model == "gpt-4"

        assert router_client_config.models[2].provider == "azure"
        assert router_client_config.models[2].deployment == "test-deployment"

        assert router_client_config.models[3].provider == "azure"
        assert router_client_config.models[3].deployment == "test-deployment"
        assert router_client_config.models[3].api_base == "test-api-base"
        assert router_client_config.models[3].api_version == "test-api-version"
        assert router_client_config.models[3].api_type == "azure"
        assert router_client_config.models[3].extra_parameters == {"timeout": 10}

    def test_from_dict_to_dict_interoperability(self):
        source_dict = {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    # legacy, this is going to be automatically added in parsing.
                    "api_type": "openai",
                    "provider_prefixed_model": "openai/gpt-4",
                },
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_base": "test-api-base",
                    "api_version": "test-api-version",
                    "timeout": 10,
                    # legacy, this is going to be automatically added in parsing.
                    "api_type": "azure",
                    "provider_prefixed_model": "azure/test-deployment",
                },
            ],
            "router": {"routing_strategy": "test"},
            "some_extra_parameter": "test-extra-parameter-1",
            "another_extra_parameter": "test-extra-parameter-2",
        }
        router_client_config = LiteLLMRouterClientConfig.from_dict(source_dict)
        destination_dict = router_client_config.to_dict()

        assert destination_dict == source_dict

    def test_to_litellm_config(self):
        # Given
        source_dict = {
            "id": "test-model-group-id",
            "models": [
                {"provider": "cohere", "model": "test-cohere"},
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    # Api type should not be present in litellm config
                    "api_type": "openai",
                },
                {"provider": "azure", "deployment": "test-deployment"},
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_base": "test-api-base",
                    "api_version": "test-api-version",
                    "timeout": 10,
                    # Api type should not be present in litellm config
                    "api_type": "azure",
                },
            ],
            "router": {"routing_strategy": "test"},
        }
        expected_litellm_dict = {
            "id": "test-model-group-id",
            "model_list": [
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {"model": "cohere/test-cohere"},
                },
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {"model": "openai/gpt-4"},
                },
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {"model": "azure/test-deployment"},
                },
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {
                        "model": "azure/test-deployment",
                        "api_base": "test-api-base",
                        "api_version": "test-api-version",
                        "timeout": 10,
                    },
                },
            ],
            "router": {"routing_strategy": "test"},
        }
        router_client_config = LiteLLMRouterClientConfig.from_dict(source_dict)
        destination_dict = router_client_config.to_litellm_dict()

        assert destination_dict == expected_litellm_dict

    def test_litellm_model_list(self):
        source_dict = {
            "id": "test-model-group-id",
            "models": [
                {"provider": "cohere", "model": "test-cohere"},
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    # Api type should not be present in litellm config
                    "api_type": "openai",
                },
                {"provider": "azure", "deployment": "test-deployment"},
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_base": "test-api-base",
                    "api_version": "test-api-version",
                    "timeout": 10,
                    # Api type should not be present in litellm config
                    "api_type": "azure",
                },
            ],
            "router": {"routing_strategy": "test"},
        }
        router_client_config = LiteLLMRouterClientConfig.from_dict(source_dict)

        expected_litellm_model_list = [
            {
                "model_name": "test-model-group-id",
                "litellm_params": {"model": "cohere/test-cohere"},
            },
            {
                "model_name": "test-model-group-id",
                "litellm_params": {"model": "openai/gpt-4"},
            },
            {
                "model_name": "test-model-group-id",
                "litellm_params": {"model": "azure/test-deployment"},
            },
            {
                "model_name": "test-model-group-id",
                "litellm_params": {
                    "model": "azure/test-deployment",
                    "api_base": "test-api-base",
                    "api_version": "test-api-version",
                    "timeout": 10,
                },
            },
        ]

        assert router_client_config.litellm_model_list == expected_litellm_model_list
