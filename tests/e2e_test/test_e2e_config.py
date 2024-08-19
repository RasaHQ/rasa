from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from rasa.e2e_test.constants import KEY_LLM_AS_JUDGE
from rasa.e2e_test.e2e_config import (
    ComputeMethodType,
    InvalidLLMConfiguration,
    LLMJudgeConfig,
    LLME2ETestConverterConfig,
    create_llm_judge_config,
    create_llm_e2e_test_converter_config,
    get_conftest_path,
    read_conftest_file,
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


def test_create_llm_judge_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    test_case_path = Path(
        "data/test_e2e_config/valid_llm_config/dummy_test_case_file.yml"
    )
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        model="gpt-4",
        openai_api_key="test",
    )


def test_create_llm_judge_config_no_conftest_detected(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    config_path = tmp_path / "config.yml"
    config_path.write_text("config")

    test_case_path = tmp_path / "no_conftest_detected"
    test_case_path.mkdir()
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        model="gpt-4o-mini", openai_api_key="test", embedding_compute_method="local"
    )


def test_create_llm_judge_config_conftest_without_llm_judge_key(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    test_case_path = tmp_path / "conftest.yml"
    test_case_path.write_text("")
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        model="gpt-4o-mini", openai_api_key=None, embedding_compute_method="local"
    )


@pytest.mark.parametrize("model_value", [1, False, True, [], {}, None, set()])
def test_read_conftest_file_raises_yaml_validation_error(
    tmp_path: Path,
    model_value: Any,
) -> None:
    conftest_data = {KEY_LLM_AS_JUDGE: {"model": model_value}}
    test_case_path = tmp_path / "conftest.yml"
    write_yaml(conftest_data, test_case_path)

    with pytest.raises(YamlValidationException):
        read_conftest_file(test_case_path)


def test_get_conftest_path_found(tmp_path: Path, test_case_path: Path) -> None:
    conftest_path = tmp_path / "e2e_tests" / "conftest.yml"
    conftest_path.write_text("")
    assert get_conftest_path(test_case_path) == conftest_path

    # teardown
    conftest_path.unlink()


def test_get_conftest_path_not_found(tmp_path: Path, test_case_path: Path) -> None:
    assert get_conftest_path(test_case_path) is None


def test_llm_judge_config_from_dict_raises_invalid_llm_exception_no_embedding_token(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    config_data = {"embedding_compute_method": "api"}
    error_msg = "No API token for the embedding model was set."
    with pytest.raises(InvalidLLMConfiguration, match=error_msg):
        LLMJudgeConfig.from_dict(config_data)


def test_llm_judge_config_from_dict_raises_invalid_llm_exception_no_embedding_url(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("EMBEDDING_MODEL_API_TOKEN", "test")

    config_data = {"embedding_compute_method": "api"}
    error_msg = "No embedding model URL was set."
    with pytest.raises(InvalidLLMConfiguration, match=error_msg):
        LLMJudgeConfig.from_dict(config_data)


def test_llm_judge_config_from_dict_valid_with_defaults(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    judge_config = LLMJudgeConfig.from_dict({})

    assert judge_config.model == "gpt-4o-mini"
    assert judge_config.openai_api_key == "test"
    assert judge_config.embedding_compute_method == ComputeMethodType.LOCAL.value

    assert judge_config.logs_folder is None
    assert judge_config.seed is None
    assert judge_config.rpm_limit is None
    assert judge_config.tpm_limit is None

    assert judge_config.anthropic_api_key is None
    assert judge_config.azure_api_key is None
    assert judge_config.cohere_api_key is None
    assert judge_config.huggingface_api_key is None
    assert judge_config.replicate_api_token is None
    assert judge_config.anyscale_api_key is None
    assert judge_config.together_api_key is None
    assert judge_config.mistral_api_key is None
    assert judge_config.embedding_model_api_token is None

    assert judge_config.azure_api_version is None
    assert judge_config.azure_api_base is None


def test_llm_judge_config_from_dict_valid(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    judge_config = LLMJudgeConfig.from_dict(
        {
            "type": "anthropic",
            "model": "claude-2.1",
            "logs_folder": "logs",
            "seed": 42,
        }
    )

    assert judge_config.model == "claude-2.1"
    assert judge_config.anthropic_api_key == "test"
    assert judge_config.logs_folder == "logs"
    assert judge_config.seed == 42
    assert judge_config.embedding_compute_method == ComputeMethodType.LOCAL.value

    for key in [
        "rpm_limit",
        "tpm_limit",
        "cohere_api_key",
        "huggingface_api_key",
        "replicate_api_token",
        "anyscale_api_key",
        "together_api_key",
        "mistral_api_key",
        "embedding_model_api_token",
        "openai_api_key",
        "embedding_model_url",
        "embedding_model_api_token",
    ]:
        assert hasattr(judge_config, key)
        assert getattr(judge_config, key) is None


def test_llm_judge_config_as_dict(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    judge_config = LLMJudgeConfig.from_dict(
        {
            "type": "anthropic",
            "model": "claude-2.1",
            "logs_folder": "logs",
            "seed": 42,
        }
    )

    assert judge_config.as_dict() == {
        "model": "claude-2.1",
        "logs_folder": "logs",
        "seed": 42,
        "anthropic_api_key": "test",
        "embedding_compute_method": "local",
    }


def test_create_llm_e2e_test_converter_config_no_conftest(tmp_path: Path):
    config_path = tmp_path / "assistant" / "config.yml"
    assert create_llm_e2e_test_converter_config(
        config_path
    ) == LLME2ETestConverterConfig(
        api_type=None, model=None, deployment=None, api_base=None, extra_parameters={}
    )


def test_create_llm_e2e_test_converter_config_with_conftest(tmp_path: Path):
    conftest_path = tmp_path / "conftest.yml"
    model = "gpt-4"
    api_type = "openai"
    config_yaml_string = (
        f"llm_e2e_test_conversion:\n  model: {model}\n  api_type: {api_type}"
    )
    conftest_path.write_text(config_yaml_string)
    assert create_llm_e2e_test_converter_config(
        conftest_path
    ) == LLME2ETestConverterConfig(
        api_type=api_type,
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
        api_type=None, model=None, deployment=None, api_base=None, extra_parameters={}
    )


def test_llm_e2e_test_converter_config_from_dict_valid_with_defaults(
    monkeypatch: MonkeyPatch,
) -> None:
    converter_config = LLME2ETestConverterConfig.from_dict({})

    assert converter_config.api_type is None
    assert converter_config.model is None
    assert converter_config.deployment is None
    assert converter_config.api_base is None
    assert converter_config.extra_parameters == {}


def test_llm_e2e_test_converter_config_from_dict_valid():
    converter_config = LLME2ETestConverterConfig.from_dict(
        {
            "api_type": "openai",
            "model": "gpt-3.5-turbo",
            "deployment": "v1",
            "api_base": "https://api.openai.com/v1",
            "custom_key": "custom_value",
        }
    )

    assert converter_config.api_type == "openai"
    assert converter_config.model == "gpt-3.5-turbo"
    assert converter_config.deployment == "v1"
    assert converter_config.api_base == "https://api.openai.com/v1"
    assert converter_config.extra_parameters == {"custom_key": "custom_value"}


def test_llm_e2e_test_converter_config_as_dict():
    converter_config = LLME2ETestConverterConfig.from_dict(
        {
            "api_type": "openai",
            "model": "gpt-3.5-turbo",
            "deployment": "v1",
            "api_base": "https://api.openai.com/v1",
            "custom_key": "custom_value",
        }
    )

    assert converter_config.as_dict() == {
        "api_type": "openai",
        "model": "gpt-3.5-turbo",
        "deployment": "v1",
        "api_base": "https://api.openai.com/v1",
        "extra_parameters": {"custom_key": "custom_value"},
    }
