from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from rasa.e2e_test.constants import KEY_LLM_AS_JUDGE
from rasa.e2e_test.e2e_config import (
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


def test_create_llm_judge_config() -> None:
    test_case_path = Path(
        "data/test_e2e_config/valid_llm_config/dummy_test_case_file.yml"
    )
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        api_type="openai",
        model="gpt-4",
    )


def test_create_llm_judge_config_no_conftest_detected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("config")

    test_case_path = tmp_path / "no_conftest_detected"
    test_case_path.mkdir()
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        api_type="openai",
        model="gpt-4o-mini",
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_without_llm_judge_key(
    tmp_path: Path, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("")
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        api_type="openai",
        model="gpt-4o-mini",
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_with_custom_config(
    tmp_path: Path, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_as_judge:
        api_type: openai
        model: gpt-4
    """)
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        api_type="openai",
        model="gpt-4",
    )


@pytest.mark.parametrize("conftest_file_name", ["conftest.yaml", "conftest.yml"])
def test_create_llm_judge_config_conftest_with_invalid_llm_config(
    tmp_path: Path, conftest_file_name: str
) -> None:
    test_case_path = tmp_path / conftest_file_name
    test_case_path.write_text("""
    llm_as_judge:
        api_type: anthropic
        model: claude-2.1
    """)
    # fallback to default configuration
    assert create_llm_judge_config(test_case_path) == LLMJudgeConfig(
        api_type="openai",
        model="gpt-4o-mini",
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
    assert judge_config.api_type == "openai"
    assert judge_config.model == "gpt-4o-mini"


def test_llm_judge_config_from_dict_valid() -> None:
    judge_config = LLMJudgeConfig.from_dict(
        {
            "api_type": "openai",
            "model": "gpt-4",
        }
    )

    assert judge_config.model == "gpt-4"


def test_llm_judge_config_from_dict_invalid() -> None:
    with pytest.raises(
        InvalidLLMConfiguration,
        match="Invalid LLM type 'anthropic'. Only 'openai' is supported.",
    ):
        LLMJudgeConfig.from_dict(
            {
                "api_type": "anthropic",
                "model": "claude-2.1",
            }
        )


def test_llm_judge_config_as_dict() -> None:
    judge_config = LLMJudgeConfig.from_dict(
        {
            "api_type": "openai",
            "model": "gpt-4",
        }
    )

    assert judge_config.as_dict() == {
        "api_type": "openai",
        "model": "gpt-4",
    }


def test_llm_judge_config_get_model_uri() -> None:
    judge_config = LLMJudgeConfig.from_dict(
        {
            "api_type": "openai",
            "model": "gpt-3.5-turbo",
        }
    )

    assert judge_config.get_model_uri() == "openai:/gpt-3.5-turbo"


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
            "model": "gpt-3.5-turbo",
            "deployment": "v1",
            "api_base": "https://api.openai.com/v1",
            "custom_key": "custom_value",
        }
    )

    assert converter_config.provider == "openai"
    assert converter_config.model == "gpt-3.5-turbo"
    assert converter_config.deployment == "v1"
    assert converter_config.api_base == "https://api.openai.com/v1"
    assert converter_config.extra_parameters == {"custom_key": "custom_value"}


def test_llm_e2e_test_converter_config_as_dict():
    converter_config = LLME2ETestConverterConfig.from_dict(
        {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "deployment": "v1",
            "api_base": "https://api.openai.com/v1",
            "custom_key": "custom_value",
        }
    )

    assert converter_config.as_dict() == {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "deployment": "v1",
        "api_base": "https://api.openai.com/v1",
        "extra_parameters": {"custom_key": "custom_value"},
    }
