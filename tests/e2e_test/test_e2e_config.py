from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from rasa.core.config.configuration import Configuration
from rasa.e2e_test.constants import DEFAULT_E2E_TESTING_MODEL, KEY_LLM_JUDGE
from rasa.e2e_test.e2e_config import (
    BaseModelConfig,
    LLME2ETestConverterConfig,
    LLMJudgeConfig,
    create_llm_e2e_test_converter_config,
    create_llm_judge_config,
    get_conftest_path,
    read_conftest_file,
)
from rasa.shared.constants import (
    AZURE_OPENAI_PROVIDER,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
    OPENAI_PROVIDER,
)
from rasa.shared.utils.yaml import YamlValidationException, write_yaml


@pytest.fixture(autouse=True)
def test_case_path(tmp_path: Path) -> Path:
    test_case_path = (
        tmp_path / "e2e_tests" / "test_transfer_money" / "test_transfer_money.yml"
    )
    test_case_path.mkdir(parents=True)

    config_path = tmp_path / "config.yml"
    config_path.write_text("config")
    return test_case_path


def test_create_llm_judge_config() -> None:
    test_case_path = Path(
        "data/test_e2e_config/valid_llm_config/dummy_test_case_file.yml"
    )
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model="gpt-4",
            extra_parameters={},
        ),
        embeddings=None,
    )


def test_create_llm_judge_config_no_conftest_detected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("config")

    test_case_path = tmp_path / "no_conftest_detected"
    test_case_path.mkdir()
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model=DEFAULT_E2E_TESTING_MODEL,
            extra_parameters={},
        ),
        embeddings=None,
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_without_llm_judge_key(
    tmp_path: Path, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("")
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model=DEFAULT_E2E_TESTING_MODEL,
            extra_parameters={},
        ),
        embeddings=None,
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_with_custom_config(
    tmp_path: Path, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_judge:
        llm:
            provider: openai
            model: gpt-4
        embeddings:
            provider: openai
            model: text-embedding-3-large
    """)
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model="gpt-4",
            extra_parameters={},
        ),
        embeddings=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model="text-embedding-3-large",
            extra_parameters={},
        ),
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_with_different_llm_provider(
    tmp_path: Path, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_judge:
        llm:
            provider: anthropic
            model: claude-2.1
    """)
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider="anthropic", model="claude-2.1", extra_parameters={}
        ),
        embeddings=None,
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_with_llm_model_group(
    tmp_path: Path, monkeypatch: MonkeyPatch, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_judge:
        llm:
            model_group: openai-direct-gpt-4
        embeddings:
            provider: openai
            model: text-embedding-3-large
    """)

    endpoints_path = tmp_path / "endpoints.yml"
    endpoints_path.write_text("""
    model_groups:
        - id: openai-direct-gpt-4
          models:
            - provider: openai
              model: gpt-4
              timeout: 7
              temperature: 0.0
              top_p: 0.0
    """)
    endpoints = Configuration.initialise_endpoints(
        endpoints_path=endpoints_path
    ).endpoints
    assert endpoints.model_groups is not None

    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model="gpt-4",
            extra_parameters={
                "timeout": 7,
                "temperature": 0.0,
                "top_p": 0.0,
            },
        ),
        embeddings=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model="text-embedding-3-large",
        ),
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_with_embeddings_model_group(
    tmp_path: Path, monkeypatch: MonkeyPatch, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_judge:
        llm:
            provider: openai
            model: gpt-4
        embeddings:
            model_group: open_ai_text_embedding
    """)

    endpoints_path = tmp_path / "endpoints.yml"
    endpoints_path.write_text("""
    model_groups:
        - id: open_ai_text_embedding
          models:
            - provider: openai
              model: text-embedding-3-large
    """)
    endpoints = Configuration.initialise_endpoints(
        endpoints_path=endpoints_path
    ).endpoints
    assert endpoints.model_groups is not None

    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model="gpt-4",
        ),
        embeddings=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model="text-embedding-3-large",
            extra_parameters={},
        ),
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_with_model_groups(
    tmp_path: Path, monkeypatch: MonkeyPatch, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_judge:
        llm:
            model_group: openai-direct-gpt-4
        embeddings:
            model_group: huggingface_local
    """)

    endpoints_path = tmp_path / "endpoints.yml"
    endpoints_path.write_text("""
    model_groups:
        - id: openai-direct-gpt-4
          models:
            - provider: openai
              model: gpt-4
              timeout: 7
              temperature: 0.0
              top_p: 0.0
        - id: huggingface_local
          models:
            - provider: huggingface_local
              model: BAAI/bge-small-en-v1.5
              model_kwargs:
                device: "cpu"
              encode_kwargs:
                normalize_embeddings: true
              timeout: 7
    """)
    endpoints = Configuration.initialise_endpoints(
        endpoints_path=endpoints_path
    ).endpoints
    assert endpoints.model_groups is not None

    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=OPENAI_PROVIDER,
            model="gpt-4",
            extra_parameters={
                "timeout": 7,
                "temperature": 0.0,
                "top_p": 0.0,
            },
        ),
        embeddings=BaseModelConfig(
            provider=HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
            model="BAAI/bge-small-en-v1.5",
            extra_parameters={
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True},
                "timeout": 7,
            },
        ),
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_with_deployment_conftest_with_model_groups(
    tmp_path: Path, monkeypatch: MonkeyPatch, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_judge:
        llm:
            model_group: azure_4o_model_group
        embeddings:
            model_group: azure_embedding_model_group
    """)

    endpoints_path = tmp_path / "endpoints.yml"
    endpoints_path.write_text("""
    model_groups:
        - id: azure_4o_model_group
          models:
            - provider: azure
              deployment: gpt-4o-rasa-dev-samllm
              timeout: 25
              api_version: 2024-06-01
              api_key: test
              api_base: https://my-azure-base/
        - id: azure_embedding_model_group
          models:
            - provider: azure
              deployment: text-embedding-3-large
              timeout: 20
              api_version: 2024-06-01
              api_key: test
              api_base: https://my-azure-base/
    """)
    endpoints = Configuration.initialise_endpoints(
        endpoints_path=endpoints_path
    ).endpoints
    assert endpoints.model_groups is not None

    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=AZURE_OPENAI_PROVIDER,
            extra_parameters={
                "deployment": "gpt-4o-rasa-dev-samllm",
                "timeout": 25,
                "api_version": "2024-06-01",
                "api_key": "test",
                "api_base": "https://my-azure-base/",
            },
        ),
        embeddings=BaseModelConfig(
            provider=AZURE_OPENAI_PROVIDER,
            extra_parameters={
                "deployment": "text-embedding-3-large",
                "timeout": 20,
                "api_version": "2024-06-01",
                "api_key": "test",
                "api_base": "https://my-azure-base/",
            },
        ),
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_with_deployment_conftest_without_model_groups(
    tmp_path: Path, monkeypatch: MonkeyPatch, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_judge:
        llm:
            provider: azure
            deployment: gpt-4o-rasa-dev-samllm
            timeout: 25
            api_version: 2024-06-01
            api_key: test
            api_base: https://my-azure-base/
        embeddings:
            provider: azure
            deployment: text-embedding-3-large
            timeout: 20
            api_version: 2024-06-01
            api_key: test
            api_base: https://my-azure-base/
    """)

    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        llm_config=BaseModelConfig(
            provider=AZURE_OPENAI_PROVIDER,
            extra_parameters={
                "deployment": "gpt-4o-rasa-dev-samllm",
                "timeout": 25,
                "api_version": "2024-06-01",
                "api_key": "test",
                "api_base": "https://my-azure-base/",
            },
        ),
        embeddings=BaseModelConfig(
            provider=AZURE_OPENAI_PROVIDER,
            extra_parameters={
                "deployment": "text-embedding-3-large",
                "timeout": 20,
                "api_version": "2024-06-01",
                "api_key": "test",
                "api_base": "https://my-azure-base/",
            },
        ),
    )


@pytest.mark.parametrize("model_value", [1, False, True, [], {}, None, set()])
def test_read_conftest_file_raises_yaml_validation_error(
    tmp_path: Path,
    model_value: Any,
) -> None:
    conftest_data = {KEY_LLM_JUDGE: {"model": model_value}}
    test_case_path = tmp_path / "conftest.yml"
    write_yaml(conftest_data, test_case_path)

    with pytest.raises(YamlValidationException):
        read_conftest_file(test_case_path)


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_get_conftest_path_found(
    tmp_path: Path, test_case_path: Path, conftest_file_name: str
) -> None:
    conftest_path = tmp_path / "e2e_tests" / conftest_file_name
    conftest_path.write_text("")
    assert get_conftest_path(test_case_path) == conftest_path

    # teardown
    conftest_path.unlink()


def test_get_conftest_path_not_found(tmp_path: Path, test_case_path: Path) -> None:
    assert get_conftest_path(test_case_path) is None


def test_llm_judge_config_from_dict_valid_with_defaults() -> None:
    judge_config = LLMJudgeConfig.from_dict({})
    assert judge_config.llm_config.provider == OPENAI_PROVIDER
    assert judge_config.llm_config.model == DEFAULT_E2E_TESTING_MODEL
    assert judge_config.embeddings is None


def test_llm_judge_config_from_dict_valid() -> None:
    judge_config = LLMJudgeConfig.from_dict(
        {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
            }
        }
    )

    assert judge_config.llm_config.model == "gpt-4"
    assert judge_config.embeddings is None


def test_llm_judge_config_as_dict_provider_config() -> None:
    judge_config = LLMJudgeConfig.from_dict(
        {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
            }
        }
    )

    assert judge_config.llm_config_as_dict == {
        "provider": "openai",
        "model": "gpt-4",
    }


def test_llm_judge_config_as_dict_provider_with_extra_parameters() -> None:
    judge_config = LLMJudgeConfig.from_dict(
        {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.0,
            }
        }
    )

    assert judge_config.llm_config_as_dict == {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.0,
    }


def test_create_llm_e2e_test_converter_config_no_conftest(tmp_path: Path):
    config_path = tmp_path / "assistant" / "config.yml"
    assert create_llm_e2e_test_converter_config(
        config_path
    ) == LLME2ETestConverterConfig(
        provider=None, model=None, deployment=None, api_base=None, extra_parameters={}
    )


def test_create_llm_e2e_test_converter_config_with_conftest(tmp_path: Path):
    conftest_path = tmp_path / "conftest.yml"
    model = "gpt-4"
    provider = "openai"
    config_yaml_string = (
        f"llm_e2e_test_conversion:\n  model: {model}\n  provider: {provider}"
    )
    conftest_path.write_text(config_yaml_string)
    assert create_llm_e2e_test_converter_config(
        conftest_path
    ) == LLME2ETestConverterConfig(
        provider=provider,
        model=model,
        deployment=None,
        api_base=None,
        extra_parameters={},
    )


def test_create_llm_e2e_test_converter_config_empty_conftest(tmp_path: Path):
    config_path = tmp_path / "conftest.yml"
    assert create_llm_e2e_test_converter_config(
        config_path
    ) == LLME2ETestConverterConfig(
        provider=None, model=None, deployment=None, api_base=None, extra_parameters={}
    )


def test_llm_e2e_test_converter_config_from_dict_valid_with_defaults(
    monkeypatch: MonkeyPatch,
) -> None:
    converter_config = LLME2ETestConverterConfig.from_dict({})

    assert converter_config.provider is None
    assert converter_config.model is None
    assert converter_config.deployment is None
    assert converter_config.api_base is None
    assert converter_config.extra_parameters == {}


def test_llm_e2e_test_converter_config_from_dict_valid():
    converter_config = LLME2ETestConverterConfig.from_dict(
        {
            "provider": "openai",
            "model": "gpt-4o",
            "deployment": "v1",
            "api_base": "https://api.openai.com/v1",
            "custom_key": "custom_value",
        }
    )

    assert converter_config.provider == "openai"
    assert converter_config.model == "gpt-4o"
    assert converter_config.deployment == "v1"
    assert converter_config.api_base == "https://api.openai.com/v1"
    assert converter_config.extra_parameters == {"custom_key": "custom_value"}


def test_llm_e2e_test_converter_config_as_dict():
    converter_config = LLME2ETestConverterConfig.from_dict(
        {
            "provider": "openai",
            "model": "gpt-4o",
            "deployment": "v1",
            "api_base": "https://api.openai.com/v1",
            "custom_key": "custom_value",
        }
    )

    assert converter_config.as_dict() == {
        "provider": "openai",
        "model": "gpt-4o",
        "deployment": "v1",
        "api_base": "https://api.openai.com/v1",
        "extra_parameters": {"custom_key": "custom_value"},
    }
