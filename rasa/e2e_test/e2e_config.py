from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, Optional, Dict, Any

import structlog
from pydantic import BaseModel

from rasa.e2e_test.constants import (
    CONFTEST_FILE_NAME,
    E2E_CONFIG_SCHEMA_FILE_PATH,
    KEY_LLM_AS_JUDGE,
    KEY_LLM_E2E_TEST_CONVERSION,
)
from rasa.shared.constants import (
    API_TYPE_CONFIG_KEY,
    API_BASE_CONFIG_KEY,
    DEPLOYMENT_CONFIG_KEY,
    MODEL_CONFIG_KEY,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.yaml import (
    parse_raw_yaml,
    read_schema_file,
    validate_yaml_content_using_schema,
)

structlogger = structlog.get_logger()

LLM_PROVIDER_TO_API_KEY = {
    "openai": "OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "anyscale": "ANYSCALE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "azure": "AZURE_API_KEY",
}


class ComputeMethodType(Enum):
    LOCAL = "local"
    REPLICATE = "replicate"
    API = "api"


class InvalidLLMConfiguration(RasaException):
    """Exception raised when the LLM configuration is invalid."""

    def __init__(self, error_message: str) -> None:
        """Creates a `InvalidLLMConfiguration`."""
        super().__init__(error_message)


@dataclass
class LLMJudgeConfig:
    """Class for storing the configuration of the LLM-As-Judge.

    The LLM-As-Judge is used to measure the factual accuracy
    (i.e., how grounded in the source documents the response is),
     or relevance of the generated response during E2E testing.
    """

    model: str = "gpt-4o-mini"

    # Embedding model
    embedding_compute_method: str = ComputeMethodType.LOCAL.value
    embedding_model_url: Optional[str] = None
    embedding_model_api_token: Optional[str] = None

    logs_folder: Optional[str] = None
    seed: Optional[int] = None

    # Rate limits
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None

    # Custom LLM provider
    custom_llm_provider: Optional[str] = None
    api_base: Optional[str] = None

    # External API keys
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    replicate_api_token: Optional[str] = None
    anyscale_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None

    azure_api_key: Optional[str] = None
    azure_api_base: Optional[str] = None
    azure_api_version: Optional[str] = None

    @staticmethod
    def from_dict(config_data: Dict[str, Any]) -> LLMJudgeConfig:
        """Loads the configuration from a dictionary."""
        llm_type = config_data.pop("type", "openai")
        external_api_keys = load_external_api_keys(llm_type)
        config_data.update(external_api_keys)

        validate_config(config_data)

        return LLMJudgeConfig(**config_data)

    @staticmethod
    def clean_up_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from the configuration."""
        return {key: value for key, value in config_data.items() if value is not None}

    def as_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        return self.clean_up_config(data)


class LLME2ETestConverterConfig(BaseModel):
    """Class for storing the LLM configuration of the E2ETestConverter.

    This configuration is used to initialize the LiteLLM client.
    """

    api_type: Optional[str]
    model: Optional[str]
    deployment: Optional[str]
    api_base: Optional[str]
    extra_parameters: Optional[Dict[str, Any]]

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> LLME2ETestConverterConfig:
        """Loads the configuration from a dictionary."""
        expected_fields = [
            API_TYPE_CONFIG_KEY,
            API_BASE_CONFIG_KEY,
            DEPLOYMENT_CONFIG_KEY,
            MODEL_CONFIG_KEY,
        ]
        kwargs = {
            expected_field: config_data.pop(expected_field, None)
            for expected_field in expected_fields
        }
        return cls(extra_parameters=config_data, **kwargs)

    @classmethod
    def get_default_config(cls) -> LLME2ETestConverterConfig:
        default_llm_config = {"api_type": "openai", "model": "gpt-4o-mini"}
        return cls.from_dict(default_llm_config)

    @staticmethod
    def _clean_up_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from the configuration."""
        return {key: value for key, value in config_data.items() if value}

    def as_dict(self) -> Dict[str, Any]:
        return self._clean_up_config(dict(self))


def load_external_api_keys(llm_type: str) -> Dict[str, Optional[str]]:
    """Load the environment variables for the external API keys."""
    environment_variable_name = LLM_PROVIDER_TO_API_KEY.get(llm_type)

    if environment_variable_name is None:
        raise InvalidLLMConfiguration(f"Unsupported LLM provider '{llm_type}'.")

    api_keys = {
        environment_variable_name.lower(): os.environ.get(environment_variable_name),
        "embedding_model_api_token": os.environ.get("EMBEDDING_MODEL_API_TOKEN"),
    }

    return api_keys


def validate_config(config_data: Dict[str, Any]) -> None:
    """Validate the configuration."""
    if config_data.get("embedding_compute_method") == ComputeMethodType.API.value:
        if config_data.get("embedding_model_api_token") is None:
            raise InvalidLLMConfiguration(
                "No API token for the embedding model was set."
            )

        if config_data.get("embedding_model_url") is None:
            raise InvalidLLMConfiguration("No embedding model URL was set.")


def get_conftest_path(test_case_path: Optional[Path]) -> Optional[Path]:
    """Get the path to the conftest.yml file.

    This assumes that the conftest.yml file is in the assistant project.
    """
    if test_case_path is None:
        return None

    while True:
        if test_case_path.is_file():
            test_case_path = test_case_path.parent

        matches = find_conftest_path(test_case_path)
        try:
            match = next(matches)
            structlogger.debug("e2e_config.get_conftest_path.found", match=match)
            return match
        except StopIteration:
            pass

        test_case_path = test_case_path.parent

        plausible_config_paths = [
            test_case_path / "config.yml",
            test_case_path / "config",
        ]
        for plausible_config_path in plausible_config_paths:
            if plausible_config_path.exists():
                # we reached the root of the assistant project
                return None

        # In case of an invalid path outside the assistant project,
        # break the loop if we reach the root
        if test_case_path == Path("."):
            return None


def find_conftest_path(path: Path) -> Generator[Path, None, None]:
    """Find the path to the conftest.yml file."""
    for file_path in path.rglob(CONFTEST_FILE_NAME):
        yield file_path


def create_llm_judge_config(test_case_path: Optional[Path]) -> LLMJudgeConfig:
    """Create the LLM-Judge configuration from the dictionary."""
    config_data = read_conftest_file(test_case_path)
    if not config_data:
        structlogger.debug("e2e_config.create_llm_judge_config.no_conftest_detected")
        return LLMJudgeConfig.from_dict(config_data)

    llm_judge_config_data = config_data.get(KEY_LLM_AS_JUDGE, {})
    if not llm_judge_config_data:
        structlogger.debug("e2e_config.create_llm_judge_config.no_llm_as_judge_key")

    structlogger.info(
        "e2e_config.create_llm_judge_config.success",
        llm_judge_config_data=llm_judge_config_data,
    )

    return LLMJudgeConfig.from_dict(llm_judge_config_data)


def create_llm_e2e_test_converter_config(
    config_path: Path,
) -> LLME2ETestConverterConfig:
    """Create the LLME2ETestConverterConfig configuration from the dictionary."""
    config_data = read_conftest_file(config_path)
    if not config_data:
        structlogger.debug(
            "e2e_config.create_llm_e2e_test_converter_config.no_conftest_detected"
        )
        return LLME2ETestConverterConfig.from_dict(config_data)

    llm_e2e_test_converter_config_data = config_data.get(
        KEY_LLM_E2E_TEST_CONVERSION, {}
    )
    if not llm_e2e_test_converter_config_data:
        structlogger.debug(
            "e2e_config.create_llm_e2e_test_converter_config.no_llm_e2e_test_converter_config_key"
        )

    structlogger.info(
        "e2e_config.create_llm_e2e_test_converter_config.success",
        llm_e2e_test_converter_config_data=llm_e2e_test_converter_config_data,
    )

    return LLME2ETestConverterConfig.from_dict(llm_e2e_test_converter_config_data)


def read_conftest_file(test_case_path: Optional[Path]) -> Dict[str, Any]:
    """Read the conftest.yml file."""
    conftest_path = get_conftest_path(test_case_path)
    if conftest_path is None:
        return {}

    e2e_config_schema = read_schema_file(E2E_CONFIG_SCHEMA_FILE_PATH)
    config_data = parse_raw_yaml(conftest_path.read_text())
    validate_yaml_content_using_schema(config_data, e2e_config_schema)

    return config_data
