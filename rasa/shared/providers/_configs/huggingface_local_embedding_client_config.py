from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import structlog

from rasa.shared.constants import (
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
    HUGGINGFACE_MULTIPROCESS_CONFIG_KEY,
    HUGGINGFACE_CACHE_FOLDER_CONFIG_KEY,
    HUGGINGFACE_SHOW_PROGRESS_CONFIG_KEY,
    HUGGINGFACE_MODEL_KWARGS_CONFIG_KEY,
    HUGGINGFACE_ENCODE_KWARGS_CONFIG_KEY,
    HUGGINGFACE_LOCAL_EMBEDDING_CACHING_FOLDER,
    PROVIDER_CONFIG_KEY,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
    TIMEOUT_CONFIG_KEY,
    REQUEST_TIMEOUT_CONFIG_KEY,
)
from rasa.shared.providers._configs.utils import (
    resolve_aliases,
    raise_deprecation_warnings,
    validate_required_keys,
)
from rasa.shared.utils.io import raise_deprecation_warning

structlogger = structlog.get_logger()

DEPRECATED_HUGGINGFACE_TYPE = "huggingface"

DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING = {
    # Provider aliases
    RASA_TYPE_CONFIG_KEY: PROVIDER_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY: PROVIDER_CONFIG_KEY,
    # Model name aliases
    MODEL_NAME_CONFIG_KEY: MODEL_CONFIG_KEY,
    # Timeout aliases
    REQUEST_TIMEOUT_CONFIG_KEY: TIMEOUT_CONFIG_KEY,
}

REQUIRED_KEYS = [MODEL_CONFIG_KEY, PROVIDER_CONFIG_KEY]


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

    multi_process: Optional[bool]
    cache_folder: Optional[str]
    show_progress: Optional[bool]

    # Provider is not actually used by sentence-transformers, but we define
    # it here because it's used as a switch denominator for HuggingFace
    # local embedding client.
    provider: str = HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER

    model_kwargs: dict = field(default_factory=dict)
    encode_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider != HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER:
            message = (
                f"API type must be set to '{HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER}'."
            )
            structlogger.error(
                "huggingface_local_embeddings_client_config.validation_error",
                message=message,
                provider=self.provider,
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
        # Check for usage of deprecated switching key and value:
        # 1. type: huggingface
        # 2. _type: huggingface
        _raise_deprecation_warning_for_huggingface_deprecated_switch_value(config)
        # Check for other deprecated keys
        raise_deprecation_warnings(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Resolve any potential aliases
        config = cls.resolve_config_aliases(config)
        # Validate that required keys are set
        validate_required_keys(config, REQUIRED_KEYS)
        this = HuggingFaceLocalEmbeddingClientConfig(
            # Required parameters
            model=config.pop(MODEL_CONFIG_KEY),
            provider=config.pop(PROVIDER_CONFIG_KEY),
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
    def resolve_config_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
        config = _resolve_huggingface_deprecated_switch_value(config)
        return resolve_aliases(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)


def is_huggingface_local_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure
    a local HuggingFace embedding client.
    """
    # Hugging face special deprecated cases:
    # 1. type: huggingface
    # 2. _type: huggingface
    # If the deprecated setting is detected resolve both alias key and key
    # value. This would mean that the configurations above will be
    # transformed to:
    # provider: huggingface_local
    config = HuggingFaceLocalEmbeddingClientConfig.resolve_config_aliases(config)

    # Case: Configuration contains `provider: huggingface_local`
    if config.get(PROVIDER_CONFIG_KEY) in [
        HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
    ]:
        return True

    return False


def _raise_deprecation_warning_for_huggingface_deprecated_switch_value(
    config: dict,
) -> None:
    deprecated_switch_keys = [RASA_TYPE_CONFIG_KEY, LANGCHAIN_TYPE_CONFIG_KEY]
    deprecation_message = (
        f"Configuration "
        f"`{{deprecated_switch_key}}: {DEPRECATED_HUGGINGFACE_TYPE}` "
        f"is deprecated and will be removed in 4.0.0. "
        f"Please use "
        f"`{PROVIDER_CONFIG_KEY}: {HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER}` "
        f"instead."
    )
    for deprecated_switch_key in deprecated_switch_keys:
        if (
            deprecated_switch_key in config
            and config[deprecated_switch_key] == DEPRECATED_HUGGINGFACE_TYPE
        ):
            raise_deprecation_warning(
                message=deprecation_message.format(
                    deprecated_switch_key=deprecated_switch_key
                )
            )


def _resolve_huggingface_deprecated_switch_value(config: dict) -> dict:
    """
    Resolve use of deprecated switching mechanism for HuggingFace local
    embedding client.

    The following settings (key + value) are deprecated:
    1. `type: huggingface`
    2. `_type: huggingface`
    in favor of `provider: huggingface_local`.


    Args:
        config: given config

    Returns:
        New config with resolved switch mechanism

    """
    config = config.copy()

    deprecated_switch_keys = [RASA_TYPE_CONFIG_KEY, LANGCHAIN_TYPE_CONFIG_KEY]
    debug_message = (
        f"Switching "
        f"`{{deprecated_switch_key}}: {DEPRECATED_HUGGINGFACE_TYPE}` "
        f"to `{PROVIDER_CONFIG_KEY}: {HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER}`."
    )

    for deprecated_switch_key in deprecated_switch_keys:
        if (
            deprecated_switch_key in config
            and config[deprecated_switch_key] == DEPRECATED_HUGGINGFACE_TYPE
        ):
            # Update configuration with new switch mechanism
            config[PROVIDER_CONFIG_KEY] = HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER
            # Pop the deprecated key used
            config.pop(deprecated_switch_key, None)

            structlogger.debug(
                "HuggingFaceLocalEmbeddingClientConfig"
                "._resolve_huggingface_deprecated_switch_value",
                message=debug_message.format(
                    deprecated_switch_key=deprecated_switch_key
                ),
                new_config=config,
            )

    return config
