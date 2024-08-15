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
from rasa.shared.utils.io import raise_deprecation_warning

structlogger = structlog.get_logger()


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
        _raise_deprecation_warnings(config)
        # Resolve any potential aliases
        config = _resolve_aliases(config)
        # Validate that required keys are set
        cls._validate_required_keys(config)
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

    @staticmethod
    def _validate_required_keys(config: dict) -> None:
        """Validates that the passed config is containing
        all the required keys.

        Raises:
            ValueError: The config does not contain required key.
        """
        required_keys = [
            API_TYPE_CONFIG_KEY,
            MODEL_CONFIG_KEY,
        ]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            message = (
                f"Missing required keys '{missing_keys}' for HuggingFace "
                f"local embeddings client configuration."
            )
            structlogger.error(
                "huggingface_local_embeddings_client_config.validate_required_keys",
                message=message,
                missing_keys=missing_keys,
            )
            raise ValueError(message)


def _resolve_aliases(config: dict) -> dict:
    """
    Resolve aliases in the Azure OpenAI configuration to standard keys for
    HuggingFace local embeddings client.

    This function ensures that all configuration keys are standardized by
    replacing any aliases with their corresponding primary keys. It helps in
    maintaining backward compatibility and avoids modifying the original
    dictionary to ensure consistency across multiple usages.

    It does not add new keys if the keys were not previously defined.

    Args:
        config: Dictionary containing the configuration.
    Returns:
        New dictionary containing the processed configuration.

    """
    # Create a new or copied dictionary to avoid modifying the original
    # config, as it's used in multiple places (e.g. command generators).
    config = config.copy()

    # Use `model` and if there are any aliases replace them
    model = config.get(MODEL_NAME_CONFIG_KEY) or config.get(MODEL_CONFIG_KEY)
    if model is not None:
        config[MODEL_CONFIG_KEY] = model

    # Use `api_type` and if there are any aliases replace them
    # In reality, sentence-transformers is not using this at all
    # It's here for denoting that we want to use local embeddings
    # from HF.
    api_type = (
        config.get(API_TYPE_CONFIG_KEY)
        or config.get(OPENAI_API_TYPE_CONFIG_KEY)
        or config.get(RASA_TYPE_CONFIG_KEY)
        or config.get(LANGCHAIN_TYPE_CONFIG_KEY)
    )
    if api_type is not None:
        config[API_TYPE_CONFIG_KEY] = api_type

    # Pop all aliases from the config
    for key in [
        OPENAI_API_TYPE_CONFIG_KEY,
        MODEL_NAME_CONFIG_KEY,
        RASA_TYPE_CONFIG_KEY,
        LANGCHAIN_TYPE_CONFIG_KEY,
    ]:
        config.pop(key, None)

    return config


def _raise_deprecation_warnings(config: dict) -> None:
    # Check for `model` and `api_type` aliases and
    # raise deprecation warnings.
    _mapper_deprecated_keys_to_new_keys = {
        MODEL_NAME_CONFIG_KEY: MODEL_CONFIG_KEY,
        OPENAI_API_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
        RASA_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
        LANGCHAIN_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    }
    for deprecated_key, new_key in _mapper_deprecated_keys_to_new_keys.items():
        if deprecated_key in config:
            raise_deprecation_warning(
                message=(
                    f"'{deprecated_key}' is deprecated and will be removed in "
                    f"version 4.0.0. Use '{new_key}' instead."
                )
            )


def is_huggingface_local_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure
    an Azure OpenAI client.
    """
    config = _resolve_aliases(config)

    # Case: Configuration contains `api_type: huggingface`
    # or `api_type: huggingface_local`.
    if config.get(API_TYPE_CONFIG_KEY) in [
        HUGGINGFACE_LOCAL_API_TYPE,
        HUGGINGFACE_API_TYPE,
    ]:
        return True

    return False
