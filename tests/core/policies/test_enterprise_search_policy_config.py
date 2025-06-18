from typing import Any, Dict, List

import pytest
import structlog

from rasa.core.constants import POLICY_MAX_HISTORY
from rasa.core.policies.enterprise_search_policy import (
    EnterpriseSearchPolicyConfig,
)
from rasa.core.policies.enterprise_search_policy_config import (
    CHECK_RELEVANCY_PROPERTY,
    CITATION_ENABLED_PROPERTY,
    DEFAULT_CHECK_RELEVANCY_PROPERTY,
    DEFAULT_CITATION_ENABLED_PROPERTY,
    DEFAULT_EMBEDDINGS_CONFIG,
    DEFAULT_LLM_CONFIG,
    DEFAULT_MAX_MESSAGES_IN_QUERY,
    DEFAULT_TRACE_PROMPT_TOKEN_PROPERTY,
    DEFAULT_USE_LLM_PROPERTY,
    DEFAULT_VECTOR_STORE,
    DEFAULT_VECTOR_STORE_THRESHOLD,
    MAX_MESSAGES_IN_QUERY_KEY,
    SOURCE_PROPERTY,
    TRACE_TOKENS_PROPERTY,
    USE_LLM_PROPERTY,
    VECTOR_STORE_PROPERTY,
    VECTOR_STORE_TYPE_PROPERTY,
)
from rasa.shared.constants import (
    EMBEDDINGS_CONFIG_KEY,
    LLM_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
)
from tests.utilities import filter_logs


class MockAvailableEndpoints:
    @staticmethod
    def get_instance():
        return MockAvailableEndpoints()

    def __init__(self):
        self.model_groups = [
            {
                "id": "openai-test-gpt-direct",
                "models": [
                    {"provider": "openai", "model": "test-gpt"},
                ],
            },
            {
                "id": "openai-test-embeddings-direct",
                "models": [
                    {"provider": "openai", "model": "text-embeddings"},
                ],
            },
        ]


@pytest.mark.parametrize(
    "config, expected_attributes_and_values",
    [
        (
            # Default config
            {},
            {
                "use_generative_llm": DEFAULT_USE_LLM_PROPERTY,
                "enable_citation": DEFAULT_CITATION_ENABLED_PROPERTY,
                "check_relevancy": DEFAULT_CHECK_RELEVANCY_PROPERTY,
                "prompt_template": None,
                "llm_config": DEFAULT_LLM_CONFIG,
                "embeddings_config": DEFAULT_EMBEDDINGS_CONFIG,
                "vector_store_config": DEFAULT_VECTOR_STORE,
                "max_history": None,
                "max_messages_in_query": DEFAULT_MAX_MESSAGES_IN_QUERY,
                "trace_prompt_tokens": DEFAULT_TRACE_PROMPT_TOKEN_PROPERTY,
            },
        ),
        # With deprecated alias: 'prompt' instead of 'prompt_template'
        (
            {
                "prompt": "path/to/the/test/prompt.jinja2",
            },
            {
                "prompt_template": "path/to/the/test/prompt.jinja2",
            },
        ),
        # With custom LLM and embeddings config (direct)
        (
            {
                LLM_CONFIG_KEY: {
                    PROVIDER_CONFIG_KEY: "openai",
                    MODEL_CONFIG_KEY: "test-gpt",
                },
                EMBEDDINGS_CONFIG_KEY: {
                    PROVIDER_CONFIG_KEY: "openai",
                    MODEL_CONFIG_KEY: "test-embeddings",
                },
            },
            {
                "llm_config": {
                    PROVIDER_CONFIG_KEY: "openai",
                    MODEL_CONFIG_KEY: "test-gpt",
                },
                "embeddings_config": {
                    PROVIDER_CONFIG_KEY: "openai",
                    MODEL_CONFIG_KEY: "test-embeddings",
                },
            },
        ),
        # With custom LLM and embeddings config (model groups)
        (
            {
                LLM_CONFIG_KEY: {
                    MODEL_GROUP_CONFIG_KEY: "openai-test-gpt-direct",
                },
                EMBEDDINGS_CONFIG_KEY: {
                    MODEL_GROUP_CONFIG_KEY: "openai-test-embeddings-direct",
                },
            },
            {
                "llm_config": {
                    "id": "openai-test-gpt-direct",
                    "models": [{"provider": "openai", "model": "test-gpt"}],
                },
                "embeddings_config": {
                    "id": "openai-test-embeddings-direct",
                    "models": [{"provider": "openai", "model": "text-embeddings"}],
                },
            },
        ),
        # Use FAISS vector store with no source defined
        (
            {VECTOR_STORE_PROPERTY: {VECTOR_STORE_TYPE_PROPERTY: "faiss"}},
            {
                "vector_store_config": {
                    VECTOR_STORE_TYPE_PROPERTY: "faiss",
                },
                "vector_store_type": "faiss",
                "vector_store_threshold": DEFAULT_VECTOR_STORE_THRESHOLD,
                "vector_store_source": None,
            },
        ),
        # Use FAISS vector store with the custom source defined
        (
            {
                VECTOR_STORE_PROPERTY: {
                    VECTOR_STORE_TYPE_PROPERTY: "faiss",
                    SOURCE_PROPERTY: "./some/test/path/docs/",
                }
            },
            {
                "vector_store_config": {
                    VECTOR_STORE_TYPE_PROPERTY: "faiss",
                    SOURCE_PROPERTY: "./some/test/path/docs/",
                },
                "vector_store_type": "faiss",
                "vector_store_threshold": DEFAULT_VECTOR_STORE_THRESHOLD,
                "vector_store_source": "./some/test/path/docs/",
            },
        ),
        # Use non-FAISS vector store
        (
            {
                VECTOR_STORE_PROPERTY: {
                    VECTOR_STORE_TYPE_PROPERTY: "test-vector-store",
                }
            },
            {
                "vector_store_config": {
                    VECTOR_STORE_TYPE_PROPERTY: "test-vector-store",
                },
                "vector_store_type": "test-vector-store",
                "vector_store_threshold": DEFAULT_VECTOR_STORE_THRESHOLD,
                "vector_store_source": None,
            },
        ),
        # Other custom settings
        (
            {
                POLICY_MAX_HISTORY: 50,
                MAX_MESSAGES_IN_QUERY_KEY: 10,
                TRACE_TOKENS_PROPERTY: True,
            },
            {
                "max_history": 50,
                "max_messages_in_query": 10,
                "trace_prompt_tokens": True,
            },
        ),
    ],
)
def test_enterprise_search_policy_config_from_dict(
    config: Dict[str, Any],
    expected_attributes_and_values: Dict[str, Any],
    monkeypatch,
) -> None:
    # Given
    mock_endpoints = MockAvailableEndpoints()
    monkeypatch.setattr("rasa.shared.utils.llm.AvailableEndpoints", mock_endpoints)

    # When
    parsed_config = EnterpriseSearchPolicyConfig.from_dict(config)

    # Then
    for attribute, expected_value in expected_attributes_and_values.items():
        actual_value = getattr(parsed_config, attribute)
        assert actual_value == expected_value


@pytest.mark.parametrize(
    "config, expected_warnings",
    [
        (
            {"prompt": "test/path/to/prompt"},
            [
                "'prompt' is deprecated and will be removed in 4.0.0. "
                "Use 'prompt_template' instead."
            ],
        )
    ],
)
def test_enterprise_search_policy_config_warns_about_deprecated_keys(
    config: Dict[str, Any], expected_warnings: List[str]
) -> None:
    with pytest.warns(FutureWarning) as record:
        EnterpriseSearchPolicyConfig.from_dict(config)

        all_warning_messages = [str(warning.message) for warning in record]

        for expected_substring in expected_warnings:
            assert any(expected_substring in actual for actual in all_warning_messages)


def test_enterprise_search_policy_config_warns_when_relevancy_check_is_enabled_but_generative_search_is_disabled():  # noqa: E501
    # Given
    config = {USE_LLM_PROPERTY: False, CHECK_RELEVANCY_PROPERTY: True}
    expected_log_level = "warning"
    expected_event = (
        "enterprise_search_policy"
        ".relevancy_check_enabled_with_disabled_generative_search"
    )

    with structlog.testing.capture_logs() as caplog:
        # When
        EnterpriseSearchPolicyConfig.from_dict(config)
        logs = filter_logs(caplog, expected_event, expected_log_level)

    # Then
    assert len(logs) == 1


def test_enterprise_search_policy_config_warns_when_citation_is_enabled_but_generative_search_is_disabled():  # noqa: E501
    # Given
    config = {USE_LLM_PROPERTY: False, CITATION_ENABLED_PROPERTY: True}
    expected_log_level = "warning"
    expected_event = (
        "enterprise_search_policy" ".citation_enabled_with_disabled_generative_search"
    )

    with structlog.testing.capture_logs() as caplog:
        # When
        EnterpriseSearchPolicyConfig.from_dict(config)
        logs = filter_logs(caplog, expected_event, expected_log_level)

    # Then
    assert len(logs) == 1
