from __future__ import annotations

import importlib.resources
import json
import logging
from copy import deepcopy
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Text,
    Type,
    TypeVar,
    Union,
    cast,
)

import structlog
from jinja2 import Environment, select_autoescape
from pydantic import BaseModel, Field

import rasa.cli.telemetry
import rasa.cli.utils
import rasa.shared.utils.cli
import rasa.shared.utils.io
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.shared.constants import (
    CONFIG_NAME_KEY,
    CONFIG_PIPELINE_KEY,
    CONFIG_POLICIES_KEY,
    DEFAULT_PROMPT_PACKAGE_NAME,
    ENDPOINTS_NLG_KEY,
    LLM_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    MODEL_GROUPS_CONFIG_KEY,
    MODELS_CONFIG_KEY,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    ROUTER_CONFIG_KEY,
)
from rasa.shared.core.events import (
    AgentCancelled,
    AgentCompleted,
    AgentInterrupted,
    AgentResumed,
    AgentStarted,
    BotUttered,
    UserUttered,
)
from rasa.shared.core.slots import BooleanSlot, CategoricalSlot, Slot
from rasa.shared.engine.caching import get_local_cache_location
from rasa.shared.exceptions import (
    FileIOException,
    FileNotFoundException,
    InvalidConfigException,
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
from rasa.shared.providers.llm.llm_response import LLMResponse, measure_llm_latency
from rasa.shared.providers.mappings import (
    AZURE_OPENAI_PROVIDER,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
    OPENAI_PROVIDER,
    SELF_HOSTED_PROVIDER,
    get_client_config_class_from_provider,
    get_embedding_client_from_provider,
    get_llm_client_from_provider,
)
from rasa.shared.utils.common import all_subclasses
from rasa.shared.utils.constants import LOG_COMPONENT_SOURCE_METHOD_INIT

if TYPE_CHECKING:
    from rasa.core.agent import Agent
    from rasa.shared.core.trackers import DialogueStateTracker


structlogger = structlog.get_logger()

USER = "USER"

AI = "AI"

DEFAULT_OPENAI_GENERATE_MODEL_NAME = "gpt-4o-2024-11-20"

DEFAULT_OPENAI_CHAT_MODEL_NAME = "gpt-4o-2024-11-20"

DEFAULT_ENTERPRISE_SEARCH_POLICY_MODEL_NAME = "gpt-4.1-mini-2025-04-14"

DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED = "gpt-4-0613"

DEFAULT_OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-large"

DEFAULT_OPENAI_TEMPERATURE = 0.7

DEFAULT_OPENAI_MAX_GENERATED_TOKENS = 256

DEFAULT_MAX_USER_INPUT_CHARACTERS = 420

DEPLOYMENT_CENTRIC_PROVIDERS = [AZURE_OPENAI_PROVIDER]

# Placeholder messages used in the transcript for
# instances where user input results in an error
ERROR_PLACEHOLDER: Dict[str, str] = {
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


class SystemPrompts(BaseModel):
    command_generator: str = Field(
        ..., description="Prompt used by the LLM command generator."
    )
    enterprise_search: str = Field(
        ..., description="Prompt for standard enterprise search requests."
    )
    contextual_response_rephraser: str = Field(
        ..., description="Prompt used for re-phrasing assistant responses."
    )


class LLMInput(BaseModel):
    prompt: Union[List[dict], List[str], str] = Field(
        ..., description="The prompt to send to the LLM."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="The metadata to send to the LLM."
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


@measure_llm_latency
async def acompletion_with_streaming(
    llm_client: LLMClient,
    messages: Union[List[dict], List[str], str],
    output_channel: Optional[Any] = None,
    recipient_id: Optional[str] = None,
    **kwargs: Any,
) -> "LLMResponse":
    """Execute LLM completion with streaming support and output channel integration.

    This utility function handles streaming LLM responses, sending chunks to an output
    channel in real-time while accumulating the complete response. It provides a
    simple interface for streaming with automatic metadata tracking and latency
    measurement.

    Args:
        llm_client: The LLM client to use for completion.
        messages: The message(s) to send to the LLM. Can be:
            - a list of preformatted messages (dicts with 'content' and 'role' keys)
            - a list of strings (formatted as user messages)
            - a single string (formatted as user message)
        output_channel: Optional output channel to send streaming chunks to.
            If None, streaming chunks won't be sent anywhere.
        recipient_id: Optional recipient ID for the output channel.
            Required if output_channel is provided.
        **kwargs: Additional parameters to pass to the LLM completion call.

    Returns:
        LLMResponse: A complete LLMResponse object containing:
            - id: The response ID from the LLM
            - created: The creation timestamp
            - choices: List containing the complete accumulated text
            - model: The model name used
    """
    accumulated_text = ""
    llm_response_metadata = {
        "id": "",
        "created": 0,
        "model": "",
    }

    if not output_channel or not recipient_id:
        raise ValueError(
            "Output channel and recipient ID must be provided for streaming."
        )

    async for chunk_response in llm_client.acompletion_stream(messages, **kwargs):
        chunk_response = LLMResponse.ensure_llm_response(chunk_response)

        # Store metadata from the first chunk
        if llm_response_metadata.get("id") == "":
            await output_channel.send_response_chunk_start(recipient_id)
            llm_response_metadata = {
                "id": chunk_response.id,
                "created": chunk_response.created,
                "model": chunk_response.model,
            }

        # Extract and accumulate chunk text
        if chunk_response.choices and chunk_response.choices[0]:
            chunk_text = chunk_response.choices[0]
            accumulated_text += chunk_text
            await output_channel.send_response_chunk(
                recipient_id=recipient_id,
                chunk=chunk_text,
            )

    # Send end of stream signal to output channel if provided
    await output_channel.send_response_chunk_end(recipient_id)
    return LLMResponse(
        id=llm_response_metadata.get("id", ""),
        created=llm_response_metadata.get("created", 0),
        choices=[accumulated_text],
        model=llm_response_metadata.get("model", ""),
    )


def tracker_as_readable_transcript(
    tracker: "DialogueStateTracker",
    human_prefix: str = USER,
    ai_prefix: str = AI,
    max_turns: Optional[int] = 20,
    turns_wrapper: Optional[Callable[[List[str]], List[str]]] = None,
    highlight_agent_turns: bool = False,
) -> str:
    """Creates a readable dialogue from a tracker.

    Args:
        tracker: the tracker to convert
        human_prefix: the prefix to use for human utterances
        ai_prefix: the prefix to use for ai utterances
        max_turns: the maximum number of turns to include in the transcript
        turns_wrapper: optional function to wrap the turns in a custom way
        highlight_agent_turns: whether to highlight agent turns in the transcript

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
    transcript: List[str] = []

    current_ai_prefix = ai_prefix

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
            transcript.append(
                f"{current_ai_prefix}: {sanitize_message_for_prompt(event.text)}"
            )

        if highlight_agent_turns:
            if isinstance(event, AgentStarted) or isinstance(event, AgentResumed):
                current_ai_prefix = event.agent_id
            elif (
                isinstance(event, AgentCompleted)
                or isinstance(event, AgentCancelled)
                or isinstance(event, AgentInterrupted)
            ):
                current_ai_prefix = ai_prefix

    # turns_wrapper to count multiple utterances by bot/user as single turn
    if turns_wrapper:
        transcript = turns_wrapper(transcript)
    # otherwise, just take the last `max_turns` lines of the transcript
    transcript = transcript[-max_turns if max_turns is not None else None :]

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
    """Merges the given model configuration with the default configuration.

    This method supports both single model configurations and model group configurations
    (configs that have the `models` key).

    If `custom_config` is a single model configuration, it merges `custom_config` with
    `default_config`, which is also a single model configuration.

    If `custom_config` is a model group configuration (contains the `models` key), it
    applies the merging process to each model configuration within the group
    individually, merging each with the `default_config`.

    Note that `default_config` is always a single model configuration.

    The method ensures that the provider is set and all deprecated keys are resolved,
    resulting in a valid client configuration.

    Args:
        custom_config: The custom configuration containing values to overwrite defaults.
            Can be a single model configuration or a model group configuration with a
            `models` key.
        default_config: The default configuration, which is a single model
            configuration.

    Returns:
        The merged configuration, either a single model configuration or a model group
        configuration with merged models.
    """
    if custom_config and MODELS_CONFIG_KEY in custom_config:
        return _combine_model_groups_configs_with_default_config(
            custom_config, default_config
        )
    else:
        return _combine_single_model_configs(custom_config, default_config)


def _combine_model_groups_configs_with_default_config(
    model_group_config: Dict[str, Any], default_config: Dict[str, Any]
) -> Dict[Text, Any]:
    """Merges each model configuration within a model group with the default
    configuration.

    This method processes model group configurations by applying the merging process to
    each model configuration within the group individually.

    Args:
        model_group_config: The model group configuration containing a list of model
            configurations under the `models` key.
        default_config: The default configuration for a single model.

    Returns:
        The merged model group configuration with each model configuration merged
        with the default configuration.
    """
    model_group_config = deepcopy(model_group_config)
    model_group_config_combined_with_defaults = [
        _combine_single_model_configs(model_config, default_config)
        for model_config in model_group_config[MODELS_CONFIG_KEY]
    ]
    # Update the custom models config with the combined config.
    model_group_config[MODELS_CONFIG_KEY] = model_group_config_combined_with_defaults
    return model_group_config


@_cache_combine_custom_and_default_configs
def _combine_single_model_configs(
    custom_config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> Dict[Text, Any]:
    """Merges the given model config with the default config.

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
        return client_config_clazz.from_dict(deepcopy(custom_config)).to_dict()

    # If the provider is the same in both configs
    # OR provider is not specified in the custom config
    # perform MERGE by overriding the default config keys and values
    # with custom config keys and values.
    merged_config = {**deepcopy(default_config), **deepcopy(custom_config)}
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

    If the config is using the old syntax, e.g. defining the llm client directly in
    config.yaml, then standalone client is initialised (no routing).

    If the config uses the using the new, model group syntax, defined in the
    endpoints.yml, then router client is initialised if there are more than one model
    within the group.

    Examples:
    The config below will result in a standalone client:
    ```
    {
       "provider": "openai",
       "model": "gpt-4",
       "timeout": 10,
       "num_retries": 3,
    }
    ```

    The config below will also result in a standalone client:
    ```
    {
        "id": "model-group-id",
        "models": [
            {"provider": "openai", "model": "gpt-4", "api_key": "test"},
        ],
    }
    ```

    The config below will result in a router client:
    ```
    {
        "id": "test-model-group-id",
        "models": [
            {"provider": "openai", "model": "gpt-4", "api_key": "test"},
            {
                "provider": "azure",
                "deployment": "test-deployment",
                "api_key": "test",
                "api_base": "test-api-base",
            },
        ],
        "router": {"routing_strategy": "test"},
    }
    ```

    Args:
        custom_config: The custom config  containing values to overwrite defaults.
        default_config: The default config.

    Returns:
        Instantiated client based on the configuration.
    """
    if custom_config:
        if ROUTER_CONFIG_KEY in custom_config:
            return llm_router_factory(custom_config, default_config)
        if MODELS_CONFIG_KEY in custom_config:
            return llm_client_factory(
                custom_config[MODELS_CONFIG_KEY][0], default_config
            )
    return llm_client_factory(custom_config, default_config)


def llm_router_factory(
    router_config: Dict[str, Any], default_model_config: Dict[str, Any], **kwargs: Any
) -> LLMClient:
    """Creates an LLM Router using the provided configurations.

    This function initializes an LLM Router based on the given router configuration,
    which includes multiple model configurations. For each model specified in the router
    configuration, any missing parameters are supplemented using the default model
    configuration.

    Args:
        router_config: The full router configuration containing multiple model
            configurations. Each model's configuration can override parameters from the
            default model configuration.
        default_model_config: The default configuration parameters for a single model.
            These defaults are used to fill in any missing parameters in each model's
            configuration within the router.

    Returns:
        An instance that conforms to both `LLMClient` and `RouterClient` protocols
        representing the configured LLM Router.
    """
    from rasa.shared.providers.llm.litellm_router_llm_client import (
        LiteLLMRouterLLMClient,
    )

    combined_config = _combine_model_groups_configs_with_default_config(
        router_config, default_model_config
    )
    return LiteLLMRouterLLMClient.from_config(combined_config)


def llm_client_factory(
    custom_config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> LLMClient:
    """Creates an LLM from the given config.

    Args:
        custom_config: The custom config  containing values to overwrite defaults
        default_config: The default config.

    Returns:
        Instantiated LLM based on the configuration.
    """
    config = combine_custom_and_default_config(deepcopy(custom_config), default_config)

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
    """Creates an embedding client from the given config.

    If the config is using the old syntax, e.g. defining the llm client directly in
    config.yaml, then standalone client is initialised (no routing).

    If the config uses the using the new, model group syntax, defined in the
    endpoints.yml, then router client is initialised if there are more than one model
    within the group and the router is defined.

    Examples:
    The config below will result in a standalone client:
    ```
    {
       "provider": "openai",
       "model": "text-embedding-3-large",
       "timeout": 10,
       "num_retries": 3,
    }
    ```

    The config below will also result in a standalone client:
    ```
    {
        "id": "model-group-id",
        "models": [
            {
                "provider": "openai",
                "model": "test-embedding-3-large",
                "api_key": "test"
            },
        ],
    }
    ```

    The config below will result in a router client:
    ```
    {
        "id": "test-model-group-id",
        "models": [
            {"provider": "openai", "model": "gpt-4", "api_key": "test"},
            {
                "provider": "azure",
                "deployment": "test-deployment",
                "api_key": "test",
                "api_base": "test-api-base",
            },
        ],
        "router": {"routing_strategy": "test"},
    }
    ```

    Args:
        custom_config: The custom config  containing values to overwrite defaults.
        default_config: The default config.

    Returns:
        Instantiated client based on the configuration.
    """
    if custom_config:
        if ROUTER_CONFIG_KEY in custom_config:
            return embedder_router_factory(custom_config, default_config)
        if MODELS_CONFIG_KEY in custom_config:
            return embedder_client_factory(
                custom_config[MODELS_CONFIG_KEY][0], default_config
            )
    return embedder_client_factory(custom_config, default_config)


def embedder_router_factory(
    router_config: Dict[str, Any], default_model_config: Dict[str, Any], **kwargs: Any
) -> EmbeddingClient:
    """Creates an Embedder Router using the provided configurations.

    This function initializes an Embedder Router based on the given router
    configuration, which includes multiple model configurations. For each model
    specified in the router configuration, any missing parameters are supplemented using
    the default model configuration.

    Args:
        router_config: The full router configuration containing multiple model
            configurations. Each model's configuration can override parameters from the
            default model configuration.
        default_model_config: The default configuration parameters for a single model.
            These defaults are used to fill in any missing parameters in each model's
            configuration within the router.

    Returns:
        An instance that conforms to both `EmbeddingClient` and `RouterClient` protocols
        representing the configured Embedding Router.
    """
    from rasa.shared.providers.embedding.litellm_router_embedding_client import (
        LiteLLMRouterEmbeddingClient,
    )

    combined_config = _combine_model_groups_configs_with_default_config(
        router_config, default_model_config
    )

    return LiteLLMRouterEmbeddingClient.from_config(combined_config)


def embedder_client_factory(
    custom_config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> EmbeddingClient:
    """Creates an Embedder from the given config.

    Args:
        custom_config: The custom config containing values to overwrite defaults
        default_config: The default config.


    Returns:
        Instantiated Embedder based on the configuration.
    """
    config = combine_custom_and_default_config(deepcopy(custom_config), default_config)

    ensure_cache()

    client_clazz: Type[EmbeddingClient] = get_embedding_client_from_provider(
        config[PROVIDER_CONFIG_KEY]
    )
    client = client_clazz.from_config(config)
    return client


def validate_jinja2_template(template_content: Text) -> None:
    """Validate that a template string has valid Jinja2 syntax.

    Args:
        template_content: The template content to validate

    Raises:
        jinja2.exceptions.TemplateSyntaxError: If the template has invalid syntax
        Exception: If there's an error during validation
    """
    # Create environment with custom filters
    env = Environment(
        autoescape=select_autoescape(
            disabled_extensions=["jinja2"],
            default_for_string=False,
            default=True,
        )
    )
    # Register filters - lazy import to avoid circular dependencies
    from rasa.dialogue_understanding.generator._jinja_filters import (
        to_json_escaped_string,
    )
    from rasa.dialogue_understanding.generator.constants import (
        TO_JSON_ESCAPED_STRING_JINJA_FILTER,
    )

    env.filters[TO_JSON_ESCAPED_STRING_JINJA_FILTER] = to_json_escaped_string

    # Validate Jinja2 syntax
    env.from_string(template_content)


def get_prompt_template(
    jinja_file_path: Optional[Text],
    default_prompt_template: Text,
    *,
    log_source_component: Optional[Text] = None,
    log_source_method: Optional[Literal["init", "fingerprint_addon"]] = None,
) -> Text:
    """Returns the jinja template.

    Args:
        jinja_file_path: The path to the jinja template file. If not provided, the
            default template will be used.
        default_prompt_template: The fallback prompt template to use if no file is
            found or specified.
        log_source_component: The name of the component emitting the log, used to
            identify the source in structured logging.
        log_source_method: The name of the method or function emitting the log for
            better traceability.

    Returns:
        The prompt template.
    """
    try:
        if jinja_file_path is not None:
            prompt_template = rasa.shared.utils.io.read_file(jinja_file_path)

            log_level = (
                logging.INFO
                if log_source_method == LOG_COMPONENT_SOURCE_METHOD_INIT
                else logging.DEBUG
            )

            structlogger.log(
                log_level,
                "utils.llm.get_prompt_template"
                ".custom_prompt_template_read_successfully",
                event_info=(
                    f"Custom prompt template read successfully from "
                    f"`{jinja_file_path}`."
                ),
                prompt_file_path=jinja_file_path,
                log_source_component=log_source_component,
                log_source_method=log_source_method,
            )
            return prompt_template
    except (FileIOException, FileNotFoundException) as e:
        structlogger.warning(
            "utils.llm.get_prompt_template" ".failed_to_read_custom_prompt_template",
            event_info=(
                "Failed to read custom prompt template. Using default template instead."
            ),
            log_source_component=log_source_component,
            log_source_method=log_source_method,
            error=str(e),
        )
    return default_prompt_template


def get_default_prompt_template_based_on_model(
    llm_config: Dict[str, Any],
    model_prompt_mapping: Dict[str, Any],
    default_prompt_path: str,
    fallback_prompt_path: str,
    *,
    log_source_component: Optional[Text] = None,
    log_source_method: Optional[Literal["init", "fingerprint_addon"]] = None,
) -> Text:
    """Returns the default prompt template based on the model name.

    Args:
        llm_config: The model config.
        model_prompt_mapping: The model name -> prompt template mapping.
        default_prompt_path: The path to the default prompt template for the component.
        fallback_prompt_path: The fallback prompt path for all other models that do not
            have a mapping in the model_prompt_mapping.
        log_source_component: The name of the component emitting the log, used to
            identify the source in structured logging.
        log_source_method: The name of the method or function emitting the log for
            better traceability.

    Returns:
        The default prompt template.
    """
    # Extract the provider and model name information from the configuration
    _llm_config = deepcopy(llm_config)
    if MODELS_CONFIG_KEY in _llm_config:
        _llm_config = _llm_config[MODELS_CONFIG_KEY][0]
    provider = _llm_config.get(PROVIDER_CONFIG_KEY)
    model = _llm_config.get(MODEL_CONFIG_KEY)

    # If the model is not defined, we default to the default prompt template.
    if not model:
        structlogger.debug(
            "utils.llm.get_default_prompt_template_based_on_model"
            ".using_default_prompt_template",
            event_info=(
                f"Model not defined in the config. Default prompt template read from"
                f" - `{default_prompt_path}`."
            ),
            default_prompt_path=default_prompt_path,
            log_source_component=log_source_component,
            log_source_method=log_source_method,
        )
        return importlib.resources.read_text(
            DEFAULT_PROMPT_PACKAGE_NAME, default_prompt_path
        )

    full_model_name = model if provider and provider in model else f"{provider}/{model}"

    # If the model is found in the mapping, we use the model-specific prompt
    # template.
    if prompt_file_path := model_prompt_mapping.get(full_model_name):
        structlogger.debug(
            "utils.llm.get_default_prompt_template_based_on_model"
            ".using_model_specific_prompt_template",
            event_info=(
                f"Using model-specific default prompt template. Default prompt "
                f"template read from - `{prompt_file_path}`."
            ),
            default_prompt_path=prompt_file_path,
            model_name=full_model_name,
            log_source_component=log_source_component,
            log_source_method=log_source_method,
        )
        return importlib.resources.read_text(
            DEFAULT_PROMPT_PACKAGE_NAME, prompt_file_path
        )

    # If the model is not found in the mapping, we default to the fallback prompt
    # template.
    structlogger.debug(
        "utils.llm.get_default_prompt_template_based_on_model"
        ".using_fallback_prompt_template",
        event_info=(
            f"Model not found in the model prompt mapping. Fallback prompt template "
            f"read from - `{fallback_prompt_path}`."
        ),
        fallback_prompt_path=fallback_prompt_path,
        model_name=full_model_name,
        log_source_component=log_source_component,
        log_source_method=log_source_method,
    )
    return importlib.resources.read_text(
        DEFAULT_PROMPT_PACKAGE_NAME, fallback_prompt_path
    )


def allowed_values_for_slot(slot: Slot) -> Union[str, None]:
    """Get the allowed values for a slot."""
    if isinstance(slot, BooleanSlot):
        return str([True, False])
    if isinstance(slot, CategoricalSlot):
        return str([v for v in slot.values if v != "__other__"])
    else:
        return None


def resolve_model_client_config(
    model_config: Optional[Dict[str, Any]],
    component_name: Optional[str] = None,
    model_groups: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve the model group in the model config.

    1. If the config is pointing to a model group, the corresponding model group
    of the endpoints.yml is returned.
    2. If the config is using the old syntax, e.g. defining the llm
    directly in config.yml, the config is returned as is.
    3. If the config is already resolved, return it as is.

    Args:
        model_config: The model config to be resolved.
        component_name: The name of the component.
        component_name: The method of the component.
        model_groups: Model groups from endpoints.yml.

    Returns:
        The resolved llm config.
    """

    def _raise_invalid_config_exception(reason: str) -> None:
        """Helper function to raise InvalidConfigException with a formatted message."""
        if component_name:
            message = (
                f"Could not resolve model group '{model_group_id}'"
                f" for component '{component_name}'."
            )
        else:
            message = f"Could not resolve model group '{model_group_id}'."
        message += f" {reason}"
        raise InvalidConfigException(message)

    if model_config is None:
        return None

    # Config is already resolved or defines a client without model groups
    if MODEL_GROUP_CONFIG_KEY not in model_config:
        return model_config

    model_group_id = model_config.get(MODEL_GROUP_CONFIG_KEY)

    # If `model_groups` is provided, use it to initialise `AvailableEndpoints`,
    # since `get_instance()` reads from the local endpoints file instead.
    if model_groups:
        endpoints = AvailableEndpoints(model_groups=model_groups)
    else:
        endpoints = Configuration.get_instance().endpoints
    if endpoints.model_groups is None:
        _raise_invalid_config_exception(
            reason=(
                "No model group with that id found in endpoints.yml. "
                "Please make sure to define the model group."
            )
        )

    copy_model_groups = deepcopy(endpoints.model_groups)
    model_group = [
        model_group
        for model_group in copy_model_groups  # type: ignore[union-attr]
        if model_group.get(MODEL_GROUP_ID_CONFIG_KEY) == model_group_id
    ]

    if len(model_group) == 0:
        _raise_invalid_config_exception(
            reason=(
                "No model group with that id found in endpoints.yml. "
                "Please make sure to define the model group."
            )
        )
    if len(model_group) > 1:
        _raise_invalid_config_exception(
            reason=(
                "Multiple model groups with that id found in endpoints.yml. "
                "Please make sure to define the model group just once."
            )
        )

    return model_group[0]


def generate_sender_id(test_case_name: str) -> str:
    # add timestamp suffix to ensure sender_id is unique
    return f"{test_case_name}_{datetime.now()}"


async def create_tracker_for_user_step(
    step_sender_id: str,
    agent: "Agent",
    test_case_tracker: "DialogueStateTracker",
    index_user_uttered_event: int,
) -> None:
    """Creates a tracker for the user step."""
    tracker = test_case_tracker.copy()
    # modify the sender id so that the original tracker is not overwritten
    tracker.sender_id = step_sender_id

    if tracker.events:
        # get the timestamp of the event just before the user uttered event
        timestamp = tracker.events[index_user_uttered_event - 1].timestamp
        # revert the tracker to the event just before the user uttered event
        tracker = tracker.travel_back_in_time(timestamp)

    # store the tracker with the unique sender id
    await agent.tracker_store.save(tracker)


def check_prompt_config_keys_and_warn_if_deprecated(
    config: dict, component_source: str
) -> None:
    """Checks and warns about deprecated config parameters."""
    if PROMPT_CONFIG_KEY in config and PROMPT_TEMPLATE_CONFIG_KEY in config:
        structlogger.warning(
            f"{component_source}.init"
            ".both_deprecated_and_non_deprecated_config_keys_used_at_the_same_time",
            event_info=(
                f"Both '{PROMPT_CONFIG_KEY}' and '{PROMPT_TEMPLATE_CONFIG_KEY}' "
                f"are present in the config. '{PROMPT_CONFIG_KEY}' will be ignored "
                f"in favor of {PROMPT_TEMPLATE_CONFIG_KEY}."
            ),
        )

    # 'prompt' config key is deprecated in favor of 'prompt_template'
    if PROMPT_CONFIG_KEY in config:
        structlogger.warning(
            f"{component_source}.init.deprecated_config_key",
            event_info=(
                f"The config parameter '{PROMPT_CONFIG_KEY}' is deprecated "
                "and will be removed in Rasa 4.0.0. "
                f"Please use the config parameter '{PROMPT_TEMPLATE_CONFIG_KEY}'"
                f" instead. "
            ),
        )


def _get_llm_command_generator_config(
    config: Dict[Text, Any],
) -> Optional[Dict[Text, Any]]:
    """Get the llm command generator config from config.yml.

    Args:
        config: The config.yml file data.

    Returns:
        The llm command generator config.
    """
    from rasa.dialogue_understanding.generator import LLMBasedCommandGenerator

    # Collect all LLM based Command Generator class names.
    command_generator_subclasses = all_subclasses(LLMBasedCommandGenerator)
    command_generator_class_names = [
        command_generator.__name__ for command_generator in command_generator_subclasses
    ]

    # Read the LLM config of the Command Generator from the config.yml file.
    pipelines = config.get(CONFIG_PIPELINE_KEY, [])
    for pipeline in pipelines:
        if pipeline.get(CONFIG_NAME_KEY) in command_generator_class_names:
            return pipeline.get(LLM_CONFIG_KEY)

    return None


def _get_compact_llm_command_generator_prompt(
    config: Dict[Text, Any], model_groups: Dict[Text, Any]
) -> Text:
    """Get the command generator prompt based on the config."""
    from rasa.dialogue_understanding.generator.single_step.compact_llm_command_generator import (  # noqa: E501
        CompactLLMCommandGenerator,
    )

    model_config = _get_llm_command_generator_config(config)
    llm_config = resolve_model_client_config(
        model_config=model_config,
        model_groups=model_groups,
    )
    return get_default_prompt_template_based_on_model(
        llm_config=llm_config or {},
        model_prompt_mapping=CompactLLMCommandGenerator.get_model_prompt_mapper(),
        default_prompt_path=CompactLLMCommandGenerator.get_default_prompt_template_file_name(),
        fallback_prompt_path=CompactLLMCommandGenerator.get_fallback_prompt_template_file_name(),
    )


def _get_enterprise_search_prompt(config: Dict[Text, Any]) -> Text:
    """Get the enterprise search prompt based on the config."""
    from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy

    def get_enterprise_search_config() -> Dict[Text, Any]:
        policies = config.get(CONFIG_POLICIES_KEY, [])
        for policy in policies:
            if policy.get(CONFIG_NAME_KEY) == EnterpriseSearchPolicy.__name__:
                return policy

        return {}

    enterprise_search_config = get_enterprise_search_config()
    return EnterpriseSearchPolicy.get_system_default_prompt_based_on_config(
        enterprise_search_config
    )


def get_system_default_prompts(
    config: Dict[Text, Any], endpoints: Dict[Text, Any]
) -> SystemPrompts:
    """Returns the system default prompts for the component.

    Args:
        config: The config.yml file data.
        endpoints: The endpoints configuration dictionary.

    Returns:
        SystemPrompts: A Pydantic model containing all default prompts.
    """
    from rasa.core.nlg.contextual_response_rephraser import (
        DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE,
    )

    # The Model Manager / Model API service receives the endpoints configuration
    # as raw YAML text rather than as a file path. However, both
    # Configuration.initialise_endpoints() and AvailableEndpoints.read_endpoints()
    # currently only accept a Path input and do not support loading from in-memory
    # YAML content.

    # Since these classes only support file-based initialization today, we need
    # to bootstrap the Configuration with an empty AvailableEndpoints instance for now.

    # IMPORTANT: This configuration must be properly initialized with valid endpoints
    # and available agents once Studio introduces full support for Agent configurations
    # (A2A and MCP).
    Configuration.initialise_empty()

    model_groups = endpoints.get(MODEL_GROUPS_CONFIG_KEY)
    return SystemPrompts(
        command_generator=_get_compact_llm_command_generator_prompt(
            config, model_groups
        ),
        enterprise_search=_get_enterprise_search_prompt(config),
        contextual_response_rephraser=DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE,
    )


def collect_custom_prompts(
    config: Dict[Text, Any],
    endpoints: Dict[Text, Any],
    project_root: Optional[Path] = None,
) -> Dict[Text, Text]:
    """Collects custom prompts from the project configuration and endpoints.

    Args:
        config: The configuration dictionary of the project.
        endpoints: The endpoints configuration dictionary.
        project_root: The root directory of the project.

    Returns:
        A dictionary containing custom prompts.
        The keys are:
            - 'contextual_response_rephraser'
            - 'command_generator'
            - 'enterprise_search'
    """
    from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
    from rasa.dialogue_understanding.generator.llm_based_command_generator import (
        LLMBasedCommandGenerator,
    )
    from rasa.studio.prompts import (
        COMMAND_GENERATOR_NAME,
        CONTEXTUAL_RESPONSE_REPHRASER_NAME,
        ENTERPRISE_SEARCH_NAME,
    )

    prompts: Dict[Text, Text] = {}
    project_root = project_root or Path(".").resolve()

    def _read_prompt(root: Path, path_in_yaml: Text) -> Optional[Text]:
        if not path_in_yaml:
            return None

        prompt_path = (
            (root / path_in_yaml).resolve()
            if not Path(path_in_yaml).is_absolute()
            else Path(path_in_yaml)
        )
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        structlogger.warning(
            "utils.llm.collect_custom_prompts.prompt_not_found",
            event_info=(f"Prompt file '{prompt_path}' not found. "),
            prompt_path=prompt_path,
            project_root=root,
        )
        return None

    # contextual_response_rephraser
    nlg_conf = endpoints.get(ENDPOINTS_NLG_KEY) or {}
    if prompt_text := _read_prompt(project_root, nlg_conf.get(PROMPT_CONFIG_KEY)):
        prompts[CONTEXTUAL_RESPONSE_REPHRASER_NAME] = prompt_text

    # command_generator
    command_generator_classes = {
        cls.__name__ for cls in all_subclasses(LLMBasedCommandGenerator)
    }
    for component in config.get(CONFIG_PIPELINE_KEY, []):
        if component.get(CONFIG_NAME_KEY) in command_generator_classes:
            if prompt_text := _read_prompt(
                project_root, component.get(PROMPT_TEMPLATE_CONFIG_KEY)
            ):
                prompts[COMMAND_GENERATOR_NAME] = prompt_text
                break

    # enterprise_search
    for policy in config.get(CONFIG_POLICIES_KEY, []):
        if policy.get(CONFIG_NAME_KEY) == EnterpriseSearchPolicy.__name__:
            if prompt_text := _read_prompt(project_root, policy.get(PROMPT_CONFIG_KEY)):
                prompts[ENTERPRISE_SEARCH_NAME] = prompt_text
            break

    return prompts
