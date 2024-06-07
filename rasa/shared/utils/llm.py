import os
import warnings
from typing import Any, Dict, Optional, Text, Type, TYPE_CHECKING, Union

import structlog

import rasa.shared.utils.io
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
    OPENAI_API_TYPE_ENV_VAR,
    OPENAI_API_VERSION_ENV_VAR,
    OPENAI_API_BASE_ENV_VAR,
    REQUESTS_CA_BUNDLE_ENV_VAR,
    OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
    OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_TYPE_CONFIG_KEY,
    OPENAI_API_BASE_CONFIG_KEY,
    OPENAI_DEPLOYMENT_NAME_CONFIG_KEY,
    OPENAI_DEPLOYMENT_CONFIG_KEY,
    OPENAI_ENGINE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
)
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.slots import Slot, BooleanSlot, CategoricalSlot
from rasa.shared.engine.caching import get_local_cache_location
from rasa.shared.exceptions import (
    FileIOException,
    FileNotFoundException,
)

if TYPE_CHECKING:
    from langchain.chat_models import AzureChatOpenAI
    from langchain.schema.embeddings import Embeddings
    from langchain.llms.base import BaseLLM
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.providers.openai.clients import (
        AioHTTPSessionAzureChatOpenAI,
        AioHTTPSessionOpenAIChat,
    )

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
        return default_config

    if RASA_TYPE_CONFIG_KEY in custom_config:
        # rename type to _type as "type" is the convention we use
        # across the different components in config files.
        # langchain expects "_type" as the key though
        custom_config[LANGCHAIN_TYPE_CONFIG_KEY] = custom_config.pop(
            RASA_TYPE_CONFIG_KEY
        )

    if LANGCHAIN_TYPE_CONFIG_KEY in custom_config and custom_config[
        LANGCHAIN_TYPE_CONFIG_KEY
    ] != default_config.get(LANGCHAIN_TYPE_CONFIG_KEY):
        return custom_config
    return {**default_config, **custom_config}


def ensure_cache() -> None:
    """Ensures that the cache is initialized."""
    import langchain
    from langchain.cache import SQLiteCache

    # ensure the cache directory exists
    cache_location = get_local_cache_location()
    cache_location.mkdir(parents=True, exist_ok=True)

    db_location = cache_location / "rasa-llm-cache.db"
    langchain.llm_cache = SQLiteCache(database_path=str(db_location))


def preprocess_config_for_azure(config: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocesses the config for Azure deployments.

    This function is used to preprocess the config for Azure deployments.
    AzureChatOpenAI does not expect the _type key, as it is not a defined parameter
    in the class. So we need to remove it before passing the config to the class.
    AzureChatOpenAI expects the openai_api_type key to be set instead.

    Args:
        config: The config to preprocess.

    Returns:
        The preprocessed config.
    """
    config["deployment_name"] = (
        config.get(OPENAI_DEPLOYMENT_NAME_CONFIG_KEY)
        or config.get(OPENAI_DEPLOYMENT_CONFIG_KEY)
        or config.get(OPENAI_ENGINE_CONFIG_KEY)
    )
    config["openai_api_base"] = (
        config.get(OPENAI_API_BASE_CONFIG_KEY)
        or config.get(OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY)
        or os.environ.get(OPENAI_API_BASE_ENV_VAR)
    )
    config["openai_api_type"] = (
        config.get(OPENAI_API_TYPE_CONFIG_KEY)
        or config.get(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY)
        or os.environ.get(OPENAI_API_TYPE_ENV_VAR)
    )
    config["openai_api_version"] = (
        config.get(OPENAI_API_VERSION_CONFIG_KEY)
        or config.get(OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY)
        or os.environ.get(OPENAI_API_VERSION_ENV_VAR)
    )
    for keys in [
        OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
        OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY,
        OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
        OPENAI_DEPLOYMENT_CONFIG_KEY,
        OPENAI_ENGINE_CONFIG_KEY,
        LANGCHAIN_TYPE_CONFIG_KEY,
    ]:
        config.pop(keys, None)

    return config


def process_config_for_aiohttp_chat_openai(config: Dict[str, Any]) -> Dict[str, Any]:
    config = config.copy()
    config.pop(LANGCHAIN_TYPE_CONFIG_KEY)
    return config


def llm_factory(
    custom_config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> Union[
    "BaseLLM",
    "AzureChatOpenAI",
    "AioHTTPSessionAzureChatOpenAI",
    "AioHTTPSessionOpenAIChat",
]:
    """Creates an LLM from the given config.

    Args:
        custom_config: The custom config  containing values to overwrite defaults
        default_config: The default config.


    Returns:
    Instantiated LLM based on the configuration.
    """
    from langchain.llms.loading import load_llm_from_config

    ensure_cache()

    config = combine_custom_and_default_config(custom_config, default_config)

    # need to create a copy as the langchain function modifies the
    # config in place...
    structlogger.debug("llmfactory.create.llm", config=config)
    # langchain issues a user warning when using chat models. at the same time
    # it doesn't provide a way to instantiate a chat model directly using the
    # config. so for now, we need to suppress the warning here. Original
    # warning:
    #   packages/langchain/llms/openai.py:189: UserWarning: You are trying to
    #   use a chat model. This way of initializing it is no longer supported.
    #   Instead, please use: `from langchain.chat_models import ChatOpenAI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if is_azure_config(config):
            # Azure deployments are treated differently. This is done as the
            # GPT-3.5 Turbo newer versions 0613 and 1106 only support the
            # Chat Completions API.
            from langchain.chat_models import AzureChatOpenAI
            from rasa.shared.providers.openai.clients import (
                AioHTTPSessionAzureChatOpenAI,
            )

            transformed_config = preprocess_config_for_azure(config.copy())
            if os.environ.get(REQUESTS_CA_BUNDLE_ENV_VAR) is None:
                return AzureChatOpenAI(**transformed_config)
            else:
                return AioHTTPSessionAzureChatOpenAI(**transformed_config)

        if (
            os.environ.get(REQUESTS_CA_BUNDLE_ENV_VAR) is not None
            and config.get(LANGCHAIN_TYPE_CONFIG_KEY) == "openai"
        ):
            from rasa.shared.providers.openai.clients import AioHTTPSessionOpenAIChat

            config = process_config_for_aiohttp_chat_openai(config)
            return AioHTTPSessionOpenAIChat(**config.copy())

        return load_llm_from_config(config.copy())


def embedder_factory(
    custom_config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> "Embeddings":
    """Creates an Embedder from the given config.

    Args:
        custom_config: The custom config containing values to overwrite defaults
        default_config: The default config.


    Returns:
    Instantiated Embedder based on the configuration.
    """
    from langchain.schema.embeddings import Embeddings
    from langchain.embeddings import (
        CohereEmbeddings,
        HuggingFaceHubEmbeddings,
        HuggingFaceInstructEmbeddings,
        HuggingFaceEmbeddings,
        HuggingFaceBgeEmbeddings,
        LlamaCppEmbeddings,
        OpenAIEmbeddings,
        SpacyEmbeddings,
        VertexAIEmbeddings,
    )
    from rasa.shared.providers.openai.clients import AioHTTPSessionOpenAIEmbeddings

    type_to_embedding_cls_dict: Dict[str, Type[Embeddings]] = {
        "azure": OpenAIEmbeddings,
        "openai": OpenAIEmbeddings,
        "openai-aiohttp-session": AioHTTPSessionOpenAIEmbeddings,
        "cohere": CohereEmbeddings,
        "spacy": SpacyEmbeddings,
        "vertexai": VertexAIEmbeddings,
        "huggingface_instruct": HuggingFaceInstructEmbeddings,
        "huggingface_hub": HuggingFaceHubEmbeddings,
        "huggingface_bge": HuggingFaceBgeEmbeddings,
        "huggingface": HuggingFaceEmbeddings,
        "llamacpp": LlamaCppEmbeddings,
    }

    config = combine_custom_and_default_config(custom_config, default_config)
    embedding_type = config.get(LANGCHAIN_TYPE_CONFIG_KEY)

    if (
        os.environ.get(REQUESTS_CA_BUNDLE_ENV_VAR) is not None
        and embedding_type is not None
    ):
        embedding_type = f"{embedding_type}-aiohttp-session"

    structlogger.debug("llmfactory.create.embedder", config=config)

    if not embedding_type:
        return OpenAIEmbeddings()
    elif embeddings_cls := type_to_embedding_cls_dict.get(embedding_type):
        parameters = config.copy()
        parameters.pop(LANGCHAIN_TYPE_CONFIG_KEY)
        return embeddings_cls(**parameters)
    else:
        raise ValueError(f"Unsupported embeddings type '{embedding_type}'")


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


def is_azure_config(config: Dict) -> bool:
    return (
        config.get(OPENAI_API_TYPE_CONFIG_KEY) == "azure"
        or config.get(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY) == "azure"
        or os.environ.get(OPENAI_API_TYPE_ENV_VAR) == "azure"
    )
