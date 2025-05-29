import importlib.resources
import json
import logging
from copy import deepcopy
from datetime import datetime
from functools import wraps
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

import rasa.shared.utils.io
from rasa.core.utils import AvailableEndpoints
from rasa.shared.constants import (
    DEFAULT_PROMPT_PACKAGE_NAME,
    MODEL_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    MODELS_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    ROUTER_CONFIG_KEY,
)
from rasa.shared.core.events import BotUttered, UserUttered
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
from rasa.shared.providers.mappings import (
    AZURE_OPENAI_PROVIDER,
    HUGGINGFACE_LOCAL_EMBEDDING_PROVIDER,
    OPENAI_PROVIDER,
    SELF_HOSTED_PROVIDER,
    get_client_config_class_from_provider,
    get_embedding_client_from_provider,
    get_llm_client_from_provider,
)
from rasa.shared.utils.constants import LOG_COMPONENT_SOURCE_METHOD_INIT

if TYPE_CHECKING:
    from rasa.core.agent import Agent
    from rasa.shared.core.trackers import DialogueStateTracker


structlogger = structlog.get_logger()

USER = "USER"

AI = "AI"

DEFAULT_OPENAI_GENERATE_MODEL_NAME = "gpt-3.5-turbo"

DEFAULT_OPENAI_CHAT_MODEL_NAME = "gpt-3.5-turbo"

DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED = "gpt-4-0613"

DEFAULT_OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

DEFAULT_OPENAI_TEMPERATURE = 0.7

DEFAULT_OPENAI_MAX_GENERATED_TOKENS = 256

DEFAULT_MAX_USER_INPUT_CHARACTERS = 420

DEPLOYMENT_CENTRIC_PROVIDERS = [AZURE_OPENAI_PROVIDER]

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
    turns_wrapper: Optional[Callable[[List[str]], List[str]]] = None,
) -> str:
    """Creates a readable dialogue from a tracker.

    Args:
        tracker: the tracker to convert
        human_prefix: the prefix to use for human utterances
        ai_prefix: the prefix to use for ai utterances
        max_turns: the maximum number of turns to include in the transcript
        turns_wrapper: optional function to wrap the turns in a custom way

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
       "model": "text-embedding-3-small",
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
                "model": "test-embedding-3-small",
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
    except (FileIOException, FileNotFoundException):
        structlogger.warning(
            "utils.llm.get_prompt_template" ".failed_to_read_custom_prompt_template",
            event_info=(
                "Failed to read custom prompt template. Using default template instead."
            ),
            log_source_component=log_source_component,
            log_source_method=log_source_method,
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
    model_config: Optional[Dict[str, Any]], component_name: Optional[str] = None
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

    endpoints = AvailableEndpoints.get_instance()
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
