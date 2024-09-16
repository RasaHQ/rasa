from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Text,
    Type,
    TypeVar,
    TYPE_CHECKING,
    Union,
    cast,
)
import json
import structlog

import rasa.shared.utils.io
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
    PROVIDER_CONFIG_KEY,
)
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.slots import Slot, BooleanSlot, CategoricalSlot
from rasa.shared.engine.caching import (
    get_local_cache_location,
)
from rasa.shared.exceptions import (
    FileIOException,
    FileNotFoundException,
    ProviderClientValidationError,
)
from rasa.shared.providers._configs.azure_openai_client_config import (
    is_azure_openai_config,
)
from rasa.shared.providers._configs.huggingface_local_embedding_client_config import (
    is_huggingface_local_config,
)
from rasa.shared.providers._configs.openai_client_config import is_openai_config
from rasa.shared.providers._configs.self_hosted_llm_client_config import (
    is_self_hosted_config,
)
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.mappings import (
    get_llm_client_from_provider,
    AZURE_OPENAI_PROVIDER,
    OPENAI_PROVIDER,
    SELF_HOSTED_PROVIDER,
    get_embedding_client_from_provider,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
    get_client_config_class_from_provider,
)
from rasa.shared.utils.cli import print_error_and_exit

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()

USER = "USER"

AI = "AI"

DEFAULT_OPENAI_GENERATE_MODEL_NAME = "gpt-3.5-turbo"

DEFAULT_OPENAI_CHAT_MODEL_NAME = "gpt-3.5-turbo"

DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED = "gpt-4"

DEFAULT_OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

DEFAULT_OPENAI_TEMPERATURE = 0.7

DEFAULT_OPENAI_MAX_GENERATED_TOKENS = 256

DEFAULT_MAX_USER_INPUT_CHARACTERS = 420

# Placeholder messages used in the transcript for
# instances where user input results in an error
ERROR_PLACEHOLDER = {
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG: "[User sent really long message]",
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY: "",
    "default": "[User input triggered an error]",
}

_Factory_F = TypeVar(
    "_Factory_F",
    bound=Callable[[Dict[str, Any], Dict[str, Any]], Union[EmbeddingClient, LLMClient]],
)
_CombineConfigs_F = TypeVar(
    "_CombineConfigs_F",
    bound=Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
)


def _compute_hash_for_cache_from_configs(
    config_x: Dict[str, Any], config_y: Dict[str, Any]
) -> int:
    """Get a unique hash of the default and custom configs."""
    return hash(
        json.dumps(config_x, sort_keys=True) + json.dumps(config_y, sort_keys=True)
    )


def _retrieve_from_cache(
    cache: Dict[int, Any], unique_hash: int, function: Callable, function_kwargs: dict
) -> Any:
    """Retrieve the value from the cache if it exists. If it does not exist, cache it"""
    if unique_hash in cache:
        return cache[unique_hash]
    else:
        return_value = function(**function_kwargs)
        cache[unique_hash] = return_value
        return return_value


def _cache_factory(function: _Factory_F) -> _Factory_F:
    """Memoize the factory methods based on the arguments."""
    cache: Dict[int, Union[EmbeddingClient, LLMClient]] = {}

    @wraps(function)
    def factory_method_wrapper(
        config_x: Dict[str, Any], config_y: Dict[str, Any]
    ) -> Union[EmbeddingClient, LLMClient]:
        # Get a unique hash of the default and custom configs.
        unique_hash = _compute_hash_for_cache_from_configs(config_x, config_y)
        return _retrieve_from_cache(
            cache=cache,
            unique_hash=unique_hash,
            function=function,
            function_kwargs={"custom_config": config_x, "default_config": config_y},
        )

    def clear_cache() -> None:
        cache.clear()
        structlogger.debug(
            "Cleared cache for factory method",
            function_name=function.__name__,
        )

    setattr(factory_method_wrapper, "clear_cache", clear_cache)
    return cast(_Factory_F, factory_method_wrapper)


def _cache_combine_custom_and_default_configs(
    function: _CombineConfigs_F,
) -> _CombineConfigs_F:
    """Memoize the combine_custom_and_default_config method based on the arguments."""
    cache: Dict[int, dict] = {}

    @wraps(function)
    def combine_configs_wrapper(
        config_x: Dict[str, Any], config_y: Dict[str, Any]
    ) -> dict:
        # Get a unique hash of the default and custom configs.
        unique_hash = _compute_hash_for_cache_from_configs(config_x, config_y)
        return _retrieve_from_cache(
            cache=cache,
            unique_hash=unique_hash,
            function=function,
            function_kwargs={"custom_config": config_x, "default_config": config_y},
        )

    def clear_cache() -> None:
        cache.clear()
        structlogger.debug(
            "Cleared cache for combine_custom_and_default_config method",
            function_name=function.__name__,
        )

    setattr(combine_configs_wrapper, "clear_cache", clear_cache)
    return cast(_CombineConfigs_F, combine_configs_wrapper)


def tracker_as_readable_transcript(
    tracker: "DialogueStateTracker",
    human_prefix: str = USER,
    ai_prefix: str = AI,
    max_turns: Optional[int] = 20,
) -> str:
    """Creates a readable dialogue from a tracker.

    Args:
        tracker: the tracker to convert
        human_prefix: the prefix to use for human utterances
        ai_prefix: the prefix to use for ai utterances
        max_turns: the maximum number of turns to include in the transcript

    Example:
        >>> tracker = Tracker(
        ...     sender_id="test",
        ...     slots=[],
        ...     events=[
        ...         UserUttered("hello"),
        ...         BotUttered("hi"),
        ...     ],
        ... )
        >>> tracker_as_readable_transcript(tracker)
        USER: hello
        AI: hi

    Returns:
    A string representing the transcript of the tracker
    """
    transcript = []

    # using `applied_events` rather than `events` means that only events after the
    # most recent `Restart` or `SessionStarted` are included in the transcript
    for event in tracker.applied_events():
        if isinstance(event, UserUttered):
            if event.has_triggered_error:
                first_error = event.error_commands[0]
                error_type = first_error.get("error_type")
                message = ERROR_PLACEHOLDER.get(
                    error_type, ERROR_PLACEHOLDER["default"]
                )
            else:
                message = sanitize_message_for_prompt(event.text)
            transcript.append(f"{human_prefix}: {message}")

        elif isinstance(event, BotUttered):
            transcript.append(f"{ai_prefix}: {sanitize_message_for_prompt(event.text)}")

    if max_turns:
        transcript = transcript[-max_turns:]

    return "\n".join(transcript)


def sanitize_message_for_prompt(text: Optional[str]) -> str:
    """Removes new lines from a string.

    Args:
        text: the text to sanitize

    Returns:
    A string with new lines removed.
    """
    return text.replace("\n", " ") if text else ""


@_cache_combine_custom_and_default_configs
def combine_custom_and_default_config(
    custom_config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> Dict[Text, Any]:
    """Merges the given llm config with the default config.

    This method guarantees that the provider is set and all the deprecated keys are
    resolved. Hence, produces only a valid client config.

    Only uses the default configuration arguments, if the type set in the
    custom config matches the type in the default config. Otherwise, only
    the custom config is used.

    Args:
        custom_config: The custom config containing values to overwrite defaults
        default_config: The default config.

    Returns:
        The merged config.
    """
    if custom_config is None:
        return default_config.copy()

    # Get the provider from the custom config.
    custom_config_provider = get_provider_from_config(custom_config)
    # We expect the provider to be set in the default configs of all Rasa components.
    default_config_provider = default_config[PROVIDER_CONFIG_KEY]

    if (
        custom_config_provider is not None
        and custom_config_provider != default_config_provider
    ):
        # Get the provider-specific config class
        client_config_clazz = get_client_config_class_from_provider(
            custom_config_provider
        )
        # Checks for deprecated keys, resolves aliases and returns a valid config.
        # This is done to ensure that the custom config is valid.
        return client_config_clazz.from_dict(custom_config).to_dict()

    # If the provider is the same in both configs
    # OR provider is not specified in the custom config
    # perform MERGE by overriding the default config keys and values
    # with custom config keys and values.
    merged_config = {**default_config.copy(), **custom_config.copy()}
    # Check for deprecated keys, resolve aliases and return a valid config.
    # This is done to ensure that the merged config is valid.
    default_config_clazz = get_client_config_class_from_provider(
        default_config_provider
    )
    return default_config_clazz.from_dict(merged_config).to_dict()


def get_provider_from_config(config: dict) -> Optional[str]:
    """Try to get the provider from the passed llm/embeddings configuration.
    If no provider can be found, return None.
    """
    if not config:
        return None
    if is_self_hosted_config(config):
        return SELF_HOSTED_PROVIDER
    elif is_azure_openai_config(config):
        return AZURE_OPENAI_PROVIDER
    elif is_openai_config(config):
        return OPENAI_PROVIDER
    elif is_huggingface_local_config(config):
        return HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER
    else:
        return config.get(PROVIDER_CONFIG_KEY)


def ensure_cache() -> None:
    """Ensures that the cache is initialized."""
    import litellm

    # Ensure the cache directory exists
    cache_location = get_local_cache_location() / "rasa-llm-cache"
    cache_location.mkdir(parents=True, exist_ok=True)

    # Set diskcache as a caching option
    litellm.cache = litellm.Cache(type="disk", disk_cache_dir=cache_location)


@_cache_factory
def llm_factory(
    custom_config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> LLMClient:
    """Creates an LLM from the given config.

    Args:
        custom_config: The custom config  containing values to overwrite defaults
        default_config: The default config.

    Returns:
        Instantiated LLM based on the configuration.
    """
    config = combine_custom_and_default_config(custom_config, default_config)

    ensure_cache()

    client_clazz: Type[LLMClient] = get_llm_client_from_provider(
        config[PROVIDER_CONFIG_KEY]
    )
    client = client_clazz.from_config(config)
    return client


@_cache_factory
def embedder_factory(
    custom_config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> EmbeddingClient:
    """Creates an Embedder from the given config.

    Args:
        custom_config: The custom config containing values to overwrite defaults
        default_config: The default config.


    Returns:
        Instantiated Embedder based on the configuration.
    """
    config = combine_custom_and_default_config(custom_config, default_config)

    ensure_cache()

    client_clazz: Type[EmbeddingClient] = get_embedding_client_from_provider(
        config[PROVIDER_CONFIG_KEY]
    )
    client = client_clazz.from_config(config)
    return client


def get_prompt_template(
    jinja_file_path: Optional[Text], default_prompt_template: Text
) -> Text:
    """Returns the jinja template.

    Args:
        jinja_file_path: the path to the jinja file
        default_prompt_template: the default prompt template

    Returns:
        The prompt template.
    """
    try:
        if jinja_file_path is not None:
            return rasa.shared.utils.io.read_file(jinja_file_path)
    except (FileIOException, FileNotFoundException):
        structlogger.warning(
            "Failed to read custom prompt template. Using default template instead.",
            jinja_file_path=jinja_file_path,
        )
    return default_prompt_template


def allowed_values_for_slot(slot: Slot) -> Union[str, None]:
    """Get the allowed values for a slot."""
    if isinstance(slot, BooleanSlot):
        return str([True, False])
    if isinstance(slot, CategoricalSlot):
        return str([v for v in slot.values if v != "__other__"])
    else:
        return None


def try_instantiate_llm_client(
    custom_llm_config: Optional[Dict],
    default_llm_config: Optional[Dict],
    log_source_function: str,
    log_source_component: str,
) -> None:
    """Validate llm configuration."""
    try:
        llm_factory(custom_llm_config, default_llm_config)
    except (ProviderClientValidationError, ValueError) as e:
        structlogger.error(
            f"{log_source_function}.llm_instantiation_failed",
            message="Unable to instantiate LLM client.",
            error=e,
        )
        print_error_and_exit(
            f"Unable to create the LLM client for component - {log_source_component}. "
            f"Please make sure you specified the required environment variables. "
            f"Error: {e}"
        )
