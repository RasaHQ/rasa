from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import structlog

from rasa.core.constants import (
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
    SEARCH_POLICY_PRIORITY,
)
from rasa.shared.constants import (
    DEFAULT_INCLUDE_DATE_TIME,
    DEFAULT_TIMEZONE,
    EMBEDDINGS_CONFIG_KEY,
    INCLUDE_DATE_TIME_CONFIG_KEY,
    LLM_CONFIG_KEY,
    MAX_COMPLETION_TOKENS_CONFIG_KEY,
    MAX_RETRIES_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    OPENAI_PROVIDER,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    TEMPERATURE_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
    TIMEZONE_CONFIG_KEY,
)
from rasa.shared.utils.configs import (
    raise_deprecation_warnings,
    resolve_aliases,
    validate_forbidden_keys,
    validate_required_keys,
)
from rasa.shared.utils.datetime_utils import validate_datetime_configuration
from rasa.shared.utils.llm import (
    DEFAULT_ENTERPRISE_SEARCH_POLICY_MODEL_NAME,
    DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
    resolve_model_client_config,
)

structlogger = structlog.get_logger()


SOURCE_PROPERTY = "source"
VECTOR_STORE_TYPE_PROPERTY = "type"
VECTOR_STORE_PROPERTY = "vector_store"
VECTOR_STORE_THRESHOLD_PROPERTY = "threshold"
TRACE_TOKENS_PROPERTY = "trace_prompt_tokens"
CITATION_ENABLED_PROPERTY = "citation_enabled"
USE_LLM_PROPERTY = "use_generative_llm"
CHECK_RELEVANCY_PROPERTY = "check_relevancy"
MAX_MESSAGES_IN_QUERY_KEY = "max_messages_in_query"

DEFAULT_VECTOR_STORE_TYPE = "faiss"
DEFAULT_VECTOR_STORE_THRESHOLD = 0.0
DEFAULT_VECTOR_STORE = {
    VECTOR_STORE_TYPE_PROPERTY: DEFAULT_VECTOR_STORE_TYPE,
    SOURCE_PROPERTY: "./docs",
    VECTOR_STORE_THRESHOLD_PROPERTY: DEFAULT_VECTOR_STORE_THRESHOLD,
}

DEFAULT_CHECK_RELEVANCY_PROPERTY = False
DEFAULT_USE_LLM_PROPERTY = True
DEFAULT_CITATION_ENABLED_PROPERTY = False
DEFAULT_TRACE_PROMPT_TOKEN_PROPERTY = False

DEFAULT_MAX_MESSAGES_IN_QUERY = 2

DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_ENTERPRISE_SEARCH_POLICY_MODEL_NAME,
    TIMEOUT_CONFIG_KEY: 10,
    TEMPERATURE_CONFIG_KEY: 0.0,
    MAX_COMPLETION_TOKENS_CONFIG_KEY: 256,
    MAX_RETRIES_CONFIG_KEY: 1,
}

DEFAULT_EMBEDDINGS_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
}

DEFAULT_ENTERPRISE_SEARCH_CONFIG = {
    POLICY_PRIORITY: SEARCH_POLICY_PRIORITY,
    VECTOR_STORE_PROPERTY: DEFAULT_VECTOR_STORE,
}

REQUIRED_KEYS: List[str] = []

FORBIDDEN_KEYS: List[str] = []

DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING = {
    PROMPT_CONFIG_KEY: PROMPT_TEMPLATE_CONFIG_KEY
}


@dataclass
class EnterpriseSearchPolicyConfig:
    """Parses configuration for Enterprise Search Policy."""

    # TODO: llm_config, embeddings_config, and vector_store_config should also be parsed
    #   as "Config" objects. Likely part of a broader Rasa 4.0 rewrite where all
    #   components rely on configuration parser. So, for example, llm_config and
    #   embeddings_config should be parsed as ClientConfig objects, and
    #   vector_store_config parsed as VectorStoreConfig object.
    llm_config: dict
    embeddings_config: dict
    vector_store_config: dict

    prompt_template: str

    use_generative_llm: bool = DEFAULT_USE_LLM_PROPERTY
    enable_citation: bool = DEFAULT_CITATION_ENABLED_PROPERTY
    check_relevancy: bool = DEFAULT_CHECK_RELEVANCY_PROPERTY

    max_history: Optional[int] = None
    max_messages_in_query: int = DEFAULT_MAX_MESSAGES_IN_QUERY
    trace_prompt_tokens: bool = DEFAULT_TRACE_PROMPT_TOKEN_PROPERTY
    include_date_time: bool = DEFAULT_INCLUDE_DATE_TIME
    timezone: str = DEFAULT_TIMEZONE

    @property
    def vector_store_type(self) -> str:
        # TODO: In the future this should ideally be part of the Vector config
        #       and not the property of the EnterpriseSearch config
        return (
            self.vector_store_config.get(VECTOR_STORE_TYPE_PROPERTY)
            or DEFAULT_VECTOR_STORE_TYPE
        )

    @property
    def vector_store_threshold(self) -> float:
        # TODO: In the future this should ideally be part of the Vector config
        #       and not the property of the EnterpriseSearch config
        return (
            self.vector_store_config.get(VECTOR_STORE_THRESHOLD_PROPERTY)
            or DEFAULT_VECTOR_STORE_THRESHOLD
        )

    @property
    def vector_store_source(self) -> Optional[str]:
        # TODO: In the future this should ideally be part of the Vector config
        #       and not the property of the EnterpriseSearch config
        return self.vector_store_config.get(SOURCE_PROPERTY)

    def __post_init__(self) -> None:
        if self.check_relevancy and not self.use_generative_llm:
            structlogger.warning(
                "enterprise_search_policy"
                ".relevancy_check_enabled_with_disabled_generative_search",
                event_info=(
                    f"The config parameter '{CHECK_RELEVANCY_PROPERTY}' is set to"
                    f"'True', but the generative search is disabled (config"
                    f"parameter '{USE_LLM_PROPERTY}' is set to 'False'). As a result, "
                    "the relevancy check for the generative search will be disabled. "
                    f"To use this check, set the config parameter '{USE_LLM_PROPERTY}' "
                    f"to `True`."
                ),
            )
        if self.enable_citation and not self.use_generative_llm:
            structlogger.warning(
                "enterprise_search_policy"
                ".citation_enabled_with_disabled_generative_search",
                event_info=(
                    f"The config parameter '{CITATION_ENABLED_PROPERTY}' is set to"
                    f"'True', but the generative search is disabled (config"
                    f"parameter '{USE_LLM_PROPERTY}' is set to 'False'). As a result, "
                    "the citation for the generative search will be disabled. "
                    f"To use this check, set the config parameter '{USE_LLM_PROPERTY}' "
                    f"to `True`."
                ),
            )

    @classmethod
    def from_dict(cls, config: dict) -> EnterpriseSearchPolicyConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            AzureOpenAIClientConfig
        """
        # Resolve LLM config
        llm_config = (
            resolve_model_client_config(
                config.get(LLM_CONFIG_KEY), EnterpriseSearchPolicyConfig.__name__
            )
            or DEFAULT_LLM_CONFIG
        )

        # Resolve embeddings config
        embeddings_config = (
            resolve_model_client_config(
                config.get(EMBEDDINGS_CONFIG_KEY), EnterpriseSearchPolicyConfig.__name__
            )
            or DEFAULT_EMBEDDINGS_CONFIG
        )

        # Vector store config
        vector_store_config = config.get(VECTOR_STORE_PROPERTY, DEFAULT_VECTOR_STORE)

        # Check for deprecated keys
        raise_deprecation_warnings(
            config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING, "EnterpriseSearchPolicy"
        )
        # Resolve any potential aliases (e.g. 'prompt_template' vs 'prompt')
        config = cls.resolve_config_aliases(config)

        # Validate that the required keys are present
        validate_required_keys(config, REQUIRED_KEYS)
        # Validate that the forbidden keys are not present
        validate_forbidden_keys(config, FORBIDDEN_KEYS)

        # Validate datetime configuration
        _include_date_time = config.get(
            INCLUDE_DATE_TIME_CONFIG_KEY, DEFAULT_INCLUDE_DATE_TIME
        )
        _timezone = config.get(TIMEZONE_CONFIG_KEY, DEFAULT_TIMEZONE)
        validate_datetime_configuration(
            include_date_time=_include_date_time,
            timezone=_timezone,
            is_custom_timezone_provided=TIMEZONE_CONFIG_KEY in config,
            component_name="EnterpriseSearchPolicy",
        )

        this = EnterpriseSearchPolicyConfig(
            llm_config=llm_config,
            embeddings_config=embeddings_config,
            vector_store_config=vector_store_config,
            prompt_template=config.get(PROMPT_TEMPLATE_CONFIG_KEY),
            use_generative_llm=config.get(USE_LLM_PROPERTY, DEFAULT_USE_LLM_PROPERTY),
            enable_citation=config.get(
                CITATION_ENABLED_PROPERTY, DEFAULT_CITATION_ENABLED_PROPERTY
            ),
            check_relevancy=config.get(
                CHECK_RELEVANCY_PROPERTY, DEFAULT_CHECK_RELEVANCY_PROPERTY
            ),
            max_history=config.get(POLICY_MAX_HISTORY),
            max_messages_in_query=config.get(
                MAX_MESSAGES_IN_QUERY_KEY, DEFAULT_MAX_MESSAGES_IN_QUERY
            ),
            trace_prompt_tokens=config.get(
                TRACE_TOKENS_PROPERTY, DEFAULT_TRACE_PROMPT_TOKEN_PROPERTY
            ),
            include_date_time=_include_date_time,
            timezone=_timezone,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        return asdict(self)

    @staticmethod
    def resolve_config_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_aliases(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
