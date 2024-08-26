from dataclasses import asdict, dataclass, field
from typing import Optional

import structlog

from rasa.shared.constants import (
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    OPENAI_API_TYPE_CONFIG_KEY,
    API_TYPE_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
    HUGGINGFACE_API_TYPE,
    HUGGINGFACE_MULTIPROCESS_CONFIG_KEY,
    HUGGINGFACE_CACHE_FOLDER_CONFIG_KEY,
    HUGGINGFACE_SHOW_PROGRESS_CONFIG_KEY,
    HUGGINGFACE_MODEL_KWARGS_CONFIG_KEY,
    HUGGINGFACE_ENCODE_KWARGS_CONFIG_KEY,
    HUGGINGFACE_LOCAL_API_TYPE,
    HUGGINGFACE_LOCAL_EMBEDDING_CACHING_FOLDER,
)
from rasa.shared.providers._configs.utils import (
    resolve_aliases,
    raise_deprecation_warnings,
    validate_required_keys,
)

structlogger = structlog.get_logger()

DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING = {
    # API type aliases
    OPENAI_API_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    # Model name aliases
    MODEL_NAME_CONFIG_KEY: MODEL_CONFIG_KEY,
}

REQUIRED_KEYS = [
    API_TYPE_CONFIG_KEY,
    MODEL_CONFIG_KEY,
]


@dataclass
class HuggingFaceLocalEmbeddingClientConfig:
    """Parses configuration for HuggingFace local embeddings client, resolves
    aliases and raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
            - If `api_type` has a value different from `huggingface_local` or
              `huggingface` (deprecated).
    """

    model: str
    # API Type is not actually used by sentence-transformers, but we define
    # it here because it's used as a switch denominator for HuggingFace
    # local embedding client.
    api_type: str

    multi_process: Optional[bool]
    cache_folder: Optional[str]
    show_progress: Optional[bool]

    model_kwargs: dict = field(default_factory=dict)
    encode_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.api_type not in [HUGGINGFACE_LOCAL_API_TYPE, HUGGINGFACE_API_TYPE]:
            message = f"API type must be set to '{HUGGINGFACE_LOCAL_API_TYPE}'."
            structlogger.error(
                "huggingface_local_embeddings_client_config.validation_error",
                message=message,
                api_type=self.api_type,
            )
            raise ValueError(message)
        if self.model is None:
            message = "Model cannot be set to None."
            structlogger.error(
                "huggingface_local_embeddings_client_config.validation_error",
                message=message,
                model=self.model,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> "HuggingFaceLocalEmbeddingClientConfig":
        """
        Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            DefaultLiteLLMClientConfig
        """
        # Check for deprecated keys
        raise_deprecation_warnings(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Resolve any potential aliases
        config = resolve_aliases(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Validate that required keys are set
        validate_required_keys(config, REQUIRED_KEYS)
        this = HuggingFaceLocalEmbeddingClientConfig(
            # Required parameters
            model=config.pop(MODEL_CONFIG_KEY),
            api_type=config.pop(API_TYPE_CONFIG_KEY),
            # Optional
            multi_process=config.pop(HUGGINGFACE_MULTIPROCESS_CONFIG_KEY, False),
            cache_folder=config.pop(
                HUGGINGFACE_CACHE_FOLDER_CONFIG_KEY,
                str(HUGGINGFACE_LOCAL_EMBEDDING_CACHING_FOLDER),
            ),
            show_progress=config.pop(HUGGINGFACE_SHOW_PROGRESS_CONFIG_KEY, False),
            model_kwargs=config.pop(HUGGINGFACE_MODEL_KWARGS_CONFIG_KEY, {}),
            encode_kwargs=config.pop(HUGGINGFACE_ENCODE_KWARGS_CONFIG_KEY, {}),
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        return asdict(self)


def is_huggingface_local_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure
    an Azure OpenAI client.
    """
    config = resolve_aliases(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)

    # Case: Configuration contains `api_type: huggingface`
    # or `api_type: huggingface_local`.
    if config.get(API_TYPE_CONFIG_KEY) in [
        HUGGINGFACE_LOCAL_API_TYPE,
        HUGGINGFACE_API_TYPE,
    ]:
        return True

    return False
