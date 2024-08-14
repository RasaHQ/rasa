from typing import Any, Dict, Optional, Text, Type, TYPE_CHECKING, Union

import rasa.shared.utils.io
import structlog
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
    MODEL_KEY,
)
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.slots import Slot, BooleanSlot, CategoricalSlot
from rasa.shared.engine.caching import (
    get_local_cache_location,
)
from rasa.shared.exceptions import (
    FileIOException,
    FileNotFoundException,
)
from rasa.shared.providers._configs.azure_openai_client_config import (
    is_azure_openai_config,
)
from rasa.shared.providers._configs.huggingface_local_embedding_client_config import (
    is_huggingface_local_config,
)
from rasa.shared.providers._configs.openai_client_config import is_openai_config
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.mappings import (
    get_llm_client_from_provider,
    AZURE_OPENAI_PROVIDER,
    OPENAI_PROVIDER,
    get_embedding_client_from_provider,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
)

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


def combine_custom_and_default_config(
    custom_config: Optional[Dict[Text, Any]], default_config: Dict[Text, Any]
) -> Dict[Text, Any]:
    """Merges the given llm config with the default config.

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

    # If provider of the custom config is not the same as the provider used in
    # the default config, don't merge
    custom_config_provider = get_provider_from_config(custom_config)
    default_config_provider = get_provider_from_config(default_config)

    if (
        custom_config_provider is not None
        and custom_config_provider != default_config_provider
    ):
        return custom_config.copy()

    # Otherwise, override default settings with custom settings
    return {**default_config.copy(), **custom_config.copy()}


def get_provider_from_config(config: dict) -> Optional[str]:
    """Try to get the provider from the passed llm/embeddings configuration.
    If no provider can be found, return None.
    """
    if not config:
        return None
    if is_azure_openai_config(config):
        return AZURE_OPENAI_PROVIDER
    elif is_openai_config(config):
        return OPENAI_PROVIDER
    elif is_huggingface_local_config(config):
        return HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER
    else:
        # `get_llm_provider` works for both embedding models and LLMs
        from litellm.utils import get_llm_provider

        try:
            # Try to get the provider from `model` key.
            _, provider, _, _ = get_llm_provider(model=config.get(MODEL_KEY, ""))
            return provider
        except Exception:
            # If provider is not found, return None
            return None


def get_llm_type_after_combining_custom_and_default_config(
    custom_config: Optional[Dict[Text, Any]], default_config: Dict[Text, Any]
) -> Optional[str]:
    """Get the LLM type from the combined config."""
    custom_config_provider = get_provider_from_config(custom_config)
    default_config_provider = get_provider_from_config(default_config)
    return custom_config_provider or default_config_provider


def ensure_cache() -> None:
    """Ensures that the cache is initialized."""
    import litellm

    # Ensure the cache directory exists
    cache_location = get_local_cache_location() / "rasa-llm-cache"
    cache_location.mkdir(parents=True, exist_ok=True)

    # Set diskcache as a caching option
    litellm.cache = litellm.Cache(type="disk", disk_cache_dir=cache_location)


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
    provider = get_provider_from_config(config)
    ensure_cache()
    client_clazz: Type[LLMClient] = get_llm_client_from_provider(provider)
    client = client_clazz.from_config(config)
    return client


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
    provider = get_provider_from_config(config)
    ensure_cache()
    client_clazz: Type[EmbeddingClient] = get_embedding_client_from_provider(provider)
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
