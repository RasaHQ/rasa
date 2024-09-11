from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Dict, Any

import structlog
from pydantic import BaseModel

from rasa.e2e_test.constants import (
    E2E_CONFIG_SCHEMA_FILE_PATH,
    KEY_LLM_AS_JUDGE,
    KEY_LLM_E2E_TEST_CONVERSION,
)
from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    DEPLOYMENT_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    OPENAI_PROVIDER,
    PROVIDER_CONFIG_KEY,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.yaml import (
    parse_raw_yaml,
    read_schema_file,
    validate_yaml_content_using_schema,
)

structlogger = structlog.get_logger()

CONFTEST_PATTERNS = ["conftest.yml", "conftest.yaml"]


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

    api_type: str = "openai"
    model: str = "gpt-4o-mini"

    @staticmethod
    def from_dict(config_data: Dict[str, Any]) -> LLMJudgeConfig:
        """Loads the configuration from a dictionary."""
        llm_type = config_data.pop("api_type", "openai")
        if llm_type != "openai":
            raise InvalidLLMConfiguration(
                f"Invalid LLM type '{llm_type}'. Only 'openai' is supported."
            )

        return LLMJudgeConfig(**config_data)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def get_model_uri(self) -> str:
        return f"{self.api_type}:/{self.model}"


class LLME2ETestConverterConfig(BaseModel):
    """Class for storing the LLM configuration of the E2ETestConverter.

    This configuration is used to initialize the LiteLLM client.
    """

    provider: Optional[str]
    model: Optional[str]
    deployment: Optional[str]
    api_base: Optional[str]
    extra_parameters: Optional[Dict[str, Any]]

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> LLME2ETestConverterConfig:
        """Loads the configuration from a dictionary."""
        expected_fields = [
            PROVIDER_CONFIG_KEY,
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
    def get_default_config(cls) -> Dict[str, Any]:
        return {PROVIDER_CONFIG_KEY: OPENAI_PROVIDER, MODEL_CONFIG_KEY: "gpt-4o-mini"}

    @staticmethod
    def _clean_up_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from the configuration."""
        return {key: value for key, value in config_data.items() if value}

    def as_dict(self) -> Dict[str, Any]:
        return self._clean_up_config(dict(self))


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

        test_case_path = test_case_path.parent


def find_conftest_path(path: Path) -> Generator[Path, None, None]:
    """Find the path to the conftest.yml file."""
    for pattern in CONFTEST_PATTERNS:
        for file_path in path.rglob(pattern):
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

    try:
        return LLMJudgeConfig.from_dict(llm_judge_config_data)
    except InvalidLLMConfiguration as e:
        structlogger.error(
            "e2e_config.create_llm_judge_config.invalid_llm_configuration",
            error_message=str(e),
            event_info="Falling back to default configuration.",
        )
        return LLMJudgeConfig()


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
