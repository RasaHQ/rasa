from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

from rasa.e2e_test.constants import (
    DEFAULT_E2E_TESTING_MODEL,
    E2E_CONFIG_SCHEMA_FILE_PATH,
    KEY_EXTRA_PARAMETERS,
    KEY_LLM_E2E_TEST_CONVERSION,
    KEY_LLM_JUDGE,
)
from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    DEPLOYMENT_CONFIG_KEY,
    EMBEDDINGS_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODELS_CONFIG_KEY,
    OPENAI_PROVIDER,
    PROVIDER_CONFIG_KEY,
)
from rasa.shared.utils.llm import (
    combine_custom_and_default_config,
    resolve_model_client_config,
)
from rasa.shared.utils.yaml import (
    parse_raw_yaml,
    read_schema_file,
    validate_yaml_content_using_schema,
)

structlogger = structlog.get_logger()

CONFTEST_PATTERNS = ["conftest.yml", "conftest.yaml"]


class BaseModelConfig(BaseModel):
    """Base class for model configurations used by generative assertions."""

    provider: Optional[str] = None
    model: Optional[str] = None
    extra_parameters: Dict[str, Any] = Field(default_factory=dict)
    model_group: Optional[str] = None


class LLMJudgeConfig(BaseModel):
    """Class for storing the configuration of the LLM-Judge.

    The LLM-Judge is used to measure the factual correctness
    (i.e., how grounded in the source documents the response is),
     or relevance of the generated response during E2E testing.
    """

    llm_config: BaseModelConfig
    embeddings: Optional[BaseModelConfig] = None

    @classmethod
    def get_default_llm_config(cls) -> Dict[str, Any]:
        return {
            PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
            MODEL_CONFIG_KEY: DEFAULT_E2E_TESTING_MODEL,
        }

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> LLMJudgeConfig:
        """Loads the configuration from a dictionary."""
        embeddings = config_data.pop(EMBEDDINGS_CONFIG_KEY, {})
        llm_config = config_data.pop("llm", {})

        llm_config = resolve_model_client_config(llm_config)
        llm_config, llm_extra_parameters = cls.extract_attributes(llm_config)
        if not llm_config:
            llm_config = combine_custom_and_default_config(
                llm_config, cls.get_default_llm_config()
            )
        embeddings_config = resolve_model_client_config(embeddings)
        embeddings_config, embeddings_extra_parameters = cls.extract_attributes(
            embeddings_config
        )

        return LLMJudgeConfig(
            llm_config=BaseModelConfig(
                extra_parameters=llm_extra_parameters, **llm_config
            ),
            embeddings=BaseModelConfig(
                extra_parameters=embeddings_extra_parameters, **embeddings_config
            )
            if embeddings_config
            else None,
        )

    @classmethod
    def extract_attributes(
        cls, config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract the expected fields from the configuration."""
        required_config = {}

        expected_fields = [
            PROVIDER_CONFIG_KEY,
            MODEL_CONFIG_KEY,
        ]

        if PROVIDER_CONFIG_KEY in config:
            required_config = {
                expected_field: config.pop(expected_field, None)
                for expected_field in expected_fields
            }

        elif MODELS_CONFIG_KEY in config:
            config = config.pop(MODELS_CONFIG_KEY)[0]

            required_config = {
                expected_field: config.pop(expected_field, None)
                for expected_field in expected_fields
            }

        clean_config = clean_up_config(required_config)
        return clean_config, config

    @property
    def llm_config_as_dict(self) -> Dict[str, Any]:
        return extract_config(self.llm_config)

    @property
    def embeddings_config_as_dict(self) -> Dict[str, Any]:
        if self.embeddings is None:
            return {}

        return extract_config(self.embeddings)


def clean_up_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from the configuration."""
    return {key: value for key, value in config_data.items() if value}


def extract_config(config: BaseModelConfig) -> Dict[str, Any]:
    clean_config = clean_up_config(dict(config))
    extra_parameters = clean_config.pop(KEY_EXTRA_PARAMETERS, {})
    return {**clean_config, **extra_parameters}


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
        return {
            PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
            MODEL_CONFIG_KEY: DEFAULT_E2E_TESTING_MODEL,
        }

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

    llm_judge_config_data = config_data.get(KEY_LLM_JUDGE, {})
    if not llm_judge_config_data:
        structlogger.debug("e2e_config.create_llm_judge_config.no_llm_judge_key")

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
