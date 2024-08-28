from typing import Text, Any, Dict, Optional

import pytest
from pathlib import Path
from pytest import MonkeyPatch
from unittest import mock
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered, SessionStarted, Restarted
from rasa.shared.core.slots import (
    FloatSlot,
    TextSlot,
    BooleanSlot,
    CategoricalSlot,
    Slot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.engine.caching import CACHE_LOCATION_ENV
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.embedding.azure_openai_embedding_client import (
    AzureOpenAIEmbeddingClient,
)
from rasa.shared.providers.embedding.default_litellm_embedding_client import (
    DefaultLiteLLMEmbeddingClient,
)
from rasa.shared.providers.embedding.openai_embedding_client import (
    OpenAIEmbeddingClient,
)
from rasa.shared.providers.llm.azure_openai_llm_client import AzureOpenAILLMClient
from rasa.shared.providers.llm.default_litellm_llm_client import DefaultLiteLLMClient
from rasa.shared.providers.llm.openai_llm_client import OpenAILLMClient
from rasa.shared.utils.llm import (
    get_prompt_template,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
    embedder_factory,
    llm_factory,
    ERROR_PLACEHOLDER,
    allowed_values_for_slot,
    get_provider_from_config,
    ensure_cache,
)


def test_tracker_as_readable_transcript_handles_empty_tracker():
    tracker = DialogueStateTracker(sender_id="test", slots=[])
    assert tracker_as_readable_transcript(tracker) == ""


def test_tracker_as_readable_transcript_handles_tracker_with_events(domain: Domain):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
        ],
    )
    assert tracker_as_readable_transcript(tracker) == ("""USER: hello\nAI: hi""")


def test_tracker_as_readable_transcript_handles_session_restart(domain: Domain):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
            # this should clear the prior conversation from the transcript
            SessionStarted(),
            UserUttered("howdy"),
        ],
    )
    assert tracker_as_readable_transcript(tracker) == ("""USER: howdy""")


def test_tracker_as_readable_transcript_handles_restart(domain: Domain):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
            # this should clear the prior conversation from the transcript
            Restarted(),
            UserUttered("howdy"),
        ],
    )
    assert tracker_as_readable_transcript(tracker) == ("""USER: howdy""")


def test_tracker_as_readable_transcript_handles_tracker_with_events_and_prefixes(
    domain: Domain,
):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
        ],
        domain,
    )
    assert tracker_as_readable_transcript(
        tracker, human_prefix="FOO", ai_prefix="BAR"
    ) == ("""FOO: hello\nBAR: hi""")


def test_tracker_as_readable_transcript_handles_tracker_with_events_and_max_turns(
    domain: Domain,
):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
        ],
        domain,
    )
    assert tracker_as_readable_transcript(tracker, max_turns=1) == ("""AI: hi""")


def test_tracker_as_readable_transcript_and_discard_excess_turns_with_default_max_turns(
    domain: Domain,
):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("A0"),
            BotUttered("B1"),
            UserUttered("C2"),
            BotUttered("D3"),
            UserUttered("E4"),
            BotUttered("F5"),
            UserUttered("G6"),
            BotUttered("H7"),
            UserUttered("I8"),
            BotUttered("J9"),
            UserUttered("K10"),
            BotUttered("L11"),
            UserUttered("M12"),
            BotUttered("N13"),
            UserUttered("O14"),
            BotUttered("P15"),
            UserUttered("Q16"),
            BotUttered("R17"),
            UserUttered("S18"),
            BotUttered("T19"),
            UserUttered("U20"),
            BotUttered("V21"),
            UserUttered("W22"),
            BotUttered("X23"),
            UserUttered("Y24"),
        ],
        domain,
    )
    response = tracker_as_readable_transcript(tracker)
    assert response == (
        """AI: F5\nUSER: G6\nAI: H7\nUSER: I8\nAI: J9\nUSER: K10\nAI: L11\n"""
        """USER: M12\nAI: N13\nUSER: O14\nAI: P15\nUSER: Q16\nAI: R17\nUSER: S18\n"""
        """AI: T19\nUSER: U20\nAI: V21\nUSER: W22\nAI: X23\nUSER: Y24"""
    )
    assert response.count("\n") == 19


@pytest.mark.parametrize(
    "message, command, expected_response",
    [
        (
            "Very long message",
            {
                "command": "error",
                "error_type": RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
            },
            ERROR_PLACEHOLDER[RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG],
        ),
        (
            "",
            {
                "command": "error",
                "error_type": RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
            },
            ERROR_PLACEHOLDER[RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY],
        ),
    ],
)
def test_tracker_as_readable_transcript_with_messages_that_triggered_error(
    message: Text,
    command: Dict[Text, Any],
    expected_response: Text,
    domain: Domain,
):
    # Given
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("Hi"),
            BotUttered("Hi, how can I help you"),
            UserUttered(text=message, parse_data={"commands": [command]}),
            BotUttered("Error response"),
        ]
    )
    # When
    response = tracker_as_readable_transcript(tracker)
    # Then
    assert response == (
        f"USER: Hi\n"
        f"AI: Hi, how can I help you\n"
        f"USER: {expected_response}\n"
        f"AI: Error response"
    )
    assert response.count("\n") == 3


def test_sanitize_message_for_prompt_handles_none():
    assert sanitize_message_for_prompt(None) == ""


def test_sanitize_message_for_prompt_handles_empty_string():
    assert sanitize_message_for_prompt("") == ""


def test_sanitize_message_for_prompt_handles_string_with_newlines():
    assert sanitize_message_for_prompt("hello\nworld") == "hello world"


@pytest.mark.parametrize(
    "config, expected_provider",
    (
        # LiteLLM naming convention
        ({"model": "openai/test-gpt"}, "openai"),
        ({"model": "azure/my-test-gpt-deployment"}, "azure"),
        ({"model": "cohere/command"}, "cohere"),
        ({"model": "bedrock/test-model-on-bedrock"}, "bedrock"),
        # Relying on api_type
        ({"api_type": "openai"}, "openai"),
        ({"api_type": "azure"}, "azure"),
        # Relying on deprecated api_type aliases
        ({"type": "openai"}, "openai"),
        ({"type": "azure"}, "azure"),
        ({"_type": "openai"}, "openai"),
        ({"_type": "azure"}, "azure"),
        ({"openai_api_type": "openai"}, "openai"),
        ({"openai_api_type": "azure"}, "azure"),
        # Relying on azure openai specific config
        ({"deployment": "my-test-deployment-on-azure"}, "azure"),
        # Relying on LiteLLM's list of known models
        ({"model": "gpt-4"}, "openai"),
        ({"model": "text-embedding-3-small"}, "openai"),
        # Unkown provider
        ({"model": "unknown-model"}, None),
        ({"model": "unknown-provider/unknown-model"}, None),
        # Supporting deprecated model_name
        ({"model_name": "openai/test-gpt"}, "openai"),
        ({"model_name": "azure/my-test-gpt-deployment"}, "azure"),
    ),
)
def test_get_provider_from_config(config: dict, expected_provider: Optional[str]):
    # When
    provider = get_provider_from_config(config)
    assert provider == expected_provider


@pytest.mark.parametrize(
    "config",
    (
        {"model_name": "cohere/command"},
        {"model_name": "bedrock/test-model-on-bedrock"},
        {"model_name": "mistral/mistral-medium-latest"},
    ),
)
def test_get_provider_from_config_throws_validation_error(config: dict):
    with pytest.raises(KeyError):
        get_provider_from_config(config)


def test_llm_factory(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(None, {"model": "openai/test-gpt", "api_type": "openai"})
    assert isinstance(llm, OpenAILLMClient)


@pytest.mark.parametrize(
    "config,"
    "expected_model,"
    "expected_api_type,"
    "expected_api_base,"
    "expected_api_version",
    (
        (
            {"model": "openai/test-gpt", "api_type": "openai"},
            "openai/test-gpt",
            "openai",
            None,
            None,
        ),
        # No LiteLLM prefix, but a known model
        ({"model": "gpt-4", "api_type": "openai"}, "gpt-4", "openai", None, None),
        # Deprecated 'model_name'
        (
            {"model_name": "openai/test-gpt", "api_type": "openai"},
            "openai/test-gpt",
            "openai",
            None,
            None,
        ),
        ({"model": "test-gpt", "api_type": "openai"}, "test-gpt", "openai", None, None),
        (
            {"model": "test-gpt", "openai_api_type": "openai"},
            "test-gpt",
            "openai",
            None,
            None,
        ),
        # With `api_type` deprecated aliases
        ({"model": "test-gpt", "type": "openai"}, "test-gpt", "openai", None, None),
        ({"model": "test-gpt", "_type": "openai"}, "test-gpt", "openai", None, None),
        # With api_base and deprecated aliases
        (
            {
                "model": "gpt-4",
                "api_base": "https://my-test-base",
                "api_type": "openai",
            },
            "gpt-4",
            "openai",
            "https://my-test-base",
            None,
        ),
        (
            {
                "model": "gpt-4",
                "openai_api_base": "https://my-test-base",
                "api_type": "openai",
            },
            "gpt-4",
            "openai",
            "https://my-test-base",
            None,
        ),
        # With api_version and deprecated aliases
        (
            {"model": "gpt-4", "api_version": "v1", "api_type": "openai"},
            "gpt-4",
            "openai",
            None,
            "v1",
        ),
        (
            {"model": "gpt-4", "openai_api_version": "v2", "api_type": "openai"},
            "gpt-4",
            "openai",
            None,
            "v2",
        ),
    ),
)
def test_llm_factory_returns_openai_llm_client(
    config: dict,
    expected_model: str,
    expected_api_type: str,
    expected_api_base: str,
    expected_api_version: str,
    monkeypatch: MonkeyPatch,
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # When
    client = llm_factory(config, {})

    # Then
    assert isinstance(client, OpenAILLMClient)
    assert client.model == expected_model
    assert client.api_type == expected_api_type
    assert client.api_base == expected_api_base
    assert client.api_version == expected_api_version


def test_llm_factory_raises_exception_when_openai_client_setup_is_invalid(
    monkeypatch: MonkeyPatch,
):
    """OpenAI client requires the OPENAI_API_KEY environment variable
    to be set.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderClientValidationError):
        llm_factory({"model": "openai/gpt-4", "api_type": "openai"}, {})


@pytest.mark.parametrize(
    "config,"
    "expected_deployment,"
    "expected_api_type,"
    "expected_api_base,"
    "expected_api_version",
    (
        (
            {
                "deployment": "azure/my-test-gpt-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            "azure/my-test-gpt-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        # Deprecated aliases
        (
            {
                "deployment_name": "azure/my-test-gpt-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
            },
            "azure/my-test-gpt-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        (
            {
                "engine": "azure/my-test-gpt-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            "azure/my-test-gpt-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        # With `api_type` deprecated aliases
        (
            {
                "deployment": "azure/my-test-gpt-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "api_type": "azure",
            },
            "azure/my-test-gpt-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        (
            {
                "deployment": "azure/my-test-gpt-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "type": "azure",
            },
            "azure/my-test-gpt-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        (
            {
                "deployment": "azure/my-test-gpt-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "_type": "azure",
            },
            "azure/my-test-gpt-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
    ),
)
def test_llm_factory_returns_azure_openai_llm_client(
    config: dict,
    expected_deployment: str,
    expected_api_type: str,
    expected_api_base: str,
    expected_api_version: str,
    monkeypatch: MonkeyPatch,
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv("AZURE_API_KEY", "test")

    # When
    client = llm_factory(config, {})

    # Then
    assert isinstance(client, AzureOpenAILLMClient)
    assert client.deployment == expected_deployment
    assert client.api_type == expected_api_type
    assert client.api_base == expected_api_base
    assert client.api_version == expected_api_version


def test_llm_factory_returns_azure_openai_llm_client_with_env_vars_settings(
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setenv("AZURE_API_KEY", "test")
    monkeypatch.setenv("AZURE_API_BASE", "https://my-test-base")
    monkeypatch.setenv("AZURE_API_VERSION", "v1")
    client = llm_factory(
        {"deployment": "azure/my-test-gpt-deployment-on-azure", "api_type": "azure"}, {}
    )
    assert isinstance(client, AzureOpenAILLMClient)
    assert client.deployment == "azure/my-test-gpt-deployment-on-azure"
    assert client.api_type == "azure"
    assert client.api_base == "https://my-test-base"
    assert client.api_version == "v1"


def test_llm_factory_raises_exception_when_azure_openai_client_setup_is_invalid(
    monkeypatch: MonkeyPatch,
):
    """OpenAI client requires the following environment variables
    to be set:
    - AZURE_API_KEY
    - AZURE_API_BASE
    - AZURE_API_VERSION
    """

    required_env_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    for env_var in required_env_vars:
        monkeypatch.setenv(env_var, "test")
        with pytest.raises(ProviderClientValidationError):
            llm_factory.clear_cache()
            llm_factory(
                {
                    "deployment": "azure/my-test-gpt-deployment-on-azure",
                    "api_type": "azure",
                },
                {},
            )
        monkeypatch.delenv(env_var, raising=False)


@pytest.mark.parametrize(
    "config, api_key_env",
    (
        ({"model": "cohere/command"}, "COHERE_API_KEY"),
        ({"model": "anthropic/claude"}, "ANTHROPIC_API_KEY"),
    ),
)
def test_llm_factory_returns_default_litellm_client(
    config: dict, api_key_env: str, monkeypatch: MonkeyPatch
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv(api_key_env, "test")
    # When
    client = llm_factory(config, {})
    # Then
    assert isinstance(client, DefaultLiteLLMClient)


def test_llm_factory_raises_exception_when_default_client_setup_is_invalid():
    # Given
    # config not containing `model` key
    config = {"some_random_key": "cohere/command"}
    # When / Then
    with pytest.raises(ValueError):
        llm_factory(config, {})


def test_llm_factory_uses_custom_type(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(
        {"type": "openai", "model": "test-gpt"}, {"_type": "foobar", "model": "foo"}
    )
    assert isinstance(llm, OpenAILLMClient)


def test_llm_factory_ignores_irrelevant_default_args(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    # since the types of the custom config and the default are different
    # all default arguments should be removed.
    llm = llm_factory(
        {"type": "openai", "model": "test-gpt"}, {"_type": "foobar", "temperature": -1}
    )
    assert isinstance(llm, OpenAILLMClient)
    # since the default argument should be removed, this should be the default -
    # which is not -1
    assert llm._extra_parameters.get("temperature") != -1


def test_llm_factory_uses_additional_args_from_custom(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory({"temperature": -1}, {"api_type": "openai", "model": "test-gpt"})
    assert isinstance(llm, OpenAILLMClient)
    assert llm._extra_parameters.get("temperature") == -1


def test_embedder_factory(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    embedder = embedder_factory(None, {"api_type": "openai", "model": "test-embedding"})
    assert isinstance(embedder, OpenAIEmbeddingClient)


@pytest.mark.parametrize(
    "config,"
    "expected_model,"
    "expected_api_type,"
    "expected_api_base,"
    "expected_api_version",
    (
        (
            {"model": "openai/test-embeddings", "api_type": "openai"},
            "openai/test-embeddings",
            "openai",
            None,
            None,
        ),
        # Deprecated 'model_name'
        (
            {"model_name": "test-embeddings", "api_type": "openai"},
            "test-embeddings",
            "openai",
            None,
            None,
        ),
        # With `api_type` deprecated aliases
        (
            {"model": "test-embeddings", "type": "openai"},
            "test-embeddings",
            "openai",
            None,
            None,
        ),
        (
            {"model": "test-embeddings", "_type": "openai"},
            "test-embeddings",
            "openai",
            None,
            None,
        ),
        # With api_base and deprecated aliases
        (
            {
                "model": "test-embeddings",
                "api_base": "https://my-test-base",
                "api_type": "openai",
            },
            "test-embeddings",
            "openai",
            "https://my-test-base",
            None,
        ),
        (
            {
                "model": "test-embeddings",
                "openai_api_base": "https://my-test-base",
                "api_type": "openai",
            },
            "test-embeddings",
            "openai",
            "https://my-test-base",
            None,
        ),
        # With api_version and deprecated aliases
        (
            {"model": "test-embeddings", "api_version": "v1", "api_type": "openai"},
            "test-embeddings",
            "openai",
            None,
            "v1",
        ),
        (
            {
                "model": "test-embeddings",
                "openai_api_version": "v2",
                "api_type": "openai",
            },
            "test-embeddings",
            "openai",
            None,
            "v2",
        ),
    ),
)
def test_embedder_factory_returns_openai_embedding_client(
    config: dict,
    expected_model: str,
    expected_api_type: str,
    expected_api_base: str,
    expected_api_version: str,
    monkeypatch: MonkeyPatch,
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # When
    client = embedder_factory(config, {})

    # Then
    assert isinstance(client, OpenAIEmbeddingClient)
    assert client.model == expected_model
    assert client.api_type == expected_api_type
    assert client.api_base == expected_api_base
    assert client.api_version == expected_api_version


def test_embedder_factory_raises_exception_when_openai_client_setup_is_invalid(
    monkeypatch: MonkeyPatch,
):
    """OpenAI client requires the OPENAI_API_KEY environment variable
    to be set.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderClientValidationError):
        embedder_factory({"model": "openai/gpt-4", "api_type": "openai"}, {})


@pytest.mark.parametrize(
    "config,"
    "expected_deployment,"
    "expected_api_type,"
    "expected_api_base,"
    "expected_api_version",
    (
        (
            {
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            "my-test-embedding-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        # Deprecated aliases
        (
            {
                "deployment_name": "my-test-embedding-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
            },
            "my-test-embedding-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        (
            {
                "engine": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            "my-test-embedding-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        # With `api_type` deprecated aliases
        (
            {
                "deployment": "azure/my-test-embedding-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "api_type": "azure",
            },
            "azure/my-test-embedding-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        (
            {
                "deployment": "azure/my-test-embedding-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "type": "azure",
            },
            "azure/my-test-embedding-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        (
            {
                "deployment": "azure/my-test-embedding-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "_type": "azure",
            },
            "azure/my-test-embedding-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
    ),
)
def test_embedder_factory_returns_azure_openai_embedding_client(
    config: dict,
    expected_deployment: str,
    expected_api_type: str,
    expected_api_base: str,
    expected_api_version: str,
    monkeypatch: MonkeyPatch,
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv("AZURE_API_KEY", "test")

    # When
    client = embedder_factory(config, {})

    # Then
    assert isinstance(client, AzureOpenAIEmbeddingClient)
    assert client.deployment == expected_deployment
    assert client.api_type == expected_api_type
    assert client.api_base == expected_api_base
    assert client.api_version == expected_api_version


def test_embedder_factory_raises_exception_when_azure_openai_client_setup_is_invalid(
    monkeypatch: MonkeyPatch,
):
    """Azure OpenAI client requires the following environment variables
    to be set:
    - AZURE_API_KEY
    - AZURE_API_BASE
    - AZURE_API_VERSION
    """

    required_env_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]

    for env_var in required_env_vars:
        monkeypatch.setenv(env_var, "test")
        with pytest.raises(ProviderClientValidationError):
            embedder_factory(
                {
                    "deployment": "my-test-embedding-deployment-on-azure",
                    "api_type": "azure",
                },
                {},
            )
        monkeypatch.delenv(env_var, raising=False)


@pytest.mark.parametrize(
    "config, api_key_env",
    (
        ({"model": "cohere/embed-english-v3.0"}, "COHERE_API_KEY"),
        ({"model": "huggingface/microsoft/codebert-base"}, "HUGGINGFACE_API_KEY"),
    ),
)
def test_embedder_factory_returns_default_litellm_client(
    config: dict, api_key_env: str, monkeypatch: MonkeyPatch
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv(api_key_env, "test")
    # When
    client = embedder_factory(config, {})
    # Then
    assert isinstance(client, DefaultLiteLLMEmbeddingClient)


def test_embedder_factory_raises_exception_when_default_client_setup_is_invalid():
    # Given
    # config not containing `model` key
    config = {"some_random_key": "cohere/command"}
    # When / Then
    with pytest.raises(ValueError):
        embedder_factory(config, {})


def test_embedder_factory_uses_custom_type(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    embedder = embedder_factory(
        {"type": "openai", "model": "test-embedding"},
        {"_type": "foobar", "model": "foo"},
    )
    assert isinstance(embedder, OpenAIEmbeddingClient)


@pytest.mark.parametrize(
    "input_slot, expected_slot_values",
    [
        (FloatSlot("test_slot", []), None),
        (TextSlot("test_slot", []), None),
        (BooleanSlot("test_slot", []), "[True, False]"),
        (
            CategoricalSlot("test_slot", [], values=["Value1", "Value2"]),
            "['Value1', 'Value2']",
        ),
    ],
)
def test_allowed_values_for_slot(
    input_slot: Slot,
    expected_slot_values: Optional[str],
):
    """Test that allowed_values_for_slot returns the correct values."""
    # When
    allowed_values = allowed_values_for_slot(input_slot)
    # Then
    assert allowed_values == expected_slot_values


def test_get_prompt_template_returns_default_prompt() -> None:
    default_prompt_template = "default prompt template"
    response = get_prompt_template(None, default_prompt_template)
    assert response == default_prompt_template


def test_get_prompt_template_returns_custom_prompt(tmp_path: Path) -> None:
    prompt_template = "This is a custom prompt template"
    custom_prompt_file = tmp_path / "custom_prompt.jinja2"
    custom_prompt_file.write_text(prompt_template)
    response = get_prompt_template(custom_prompt_file, "default prompt")
    assert response == prompt_template


def test_get_prompt_template_returns_default_on_error() -> None:
    default_prompt_template = "default prompt template"
    response = get_prompt_template("non_existent_file.jinja2", default_prompt_template)
    assert response == default_prompt_template


def test_ensure_cache_creates_creates_diskcache_sqlite_db(
    tmpdir, monkeypatch: MonkeyPatch
):
    cache_dir = tmpdir / "test_ensure_cache"
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(cache_dir))
    ensure_cache()

    assert cache_dir.exists()
    assert cache_dir.isdir()
    # cache.db is the database name that is
    # created in the given directory
    assert (cache_dir / "rasa-llm-cache" / "cache.db").exists()


def test_llm_cache_factory() -> None:
    with mock.patch(
        "rasa.shared.utils.llm.get_llm_client_from_provider"
    ) as mock_get_llm_client_from_provider:
        # Reset the cache as the cache is shared across tests.
        llm_factory.clear_cache()

        mock_get_llm_client_from_provider.reset_mock()
        # Call llm_factory with the first set of configs.
        llm_factory(
            {"api_type": "openai", "model": "test-gpt"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_llm_client_from_provider.reset_mock()

        # Call llm_factory with the second set of configs.
        llm_factory(
            {"api_type": "openai", "model": "test-gpt-1000"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_llm_client_from_provider.reset_mock()

        # Call llm_factory with the third set of configs.
        llm_factory(
            {"api_type": "openai", "model": "test-gpt"},
            {"_type": "buzz", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_llm_client_from_provider.reset_mock()

        # Call llm_factory with the first set of configs again
        llm_factory(
            {"api_type": "openai", "model": "test-gpt"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_llm_client_from_provider.assert_not_called()

        # Call llm_factory with the second set of configs again
        llm_factory(
            {"api_type": "openai", "model": "test-gpt-1000"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_llm_client_from_provider.assert_not_called()


def test_cache_factory_ensures_no_mixup_between_llm_and_embedder_factory() -> None:
    with mock.patch(
        "rasa.shared.utils.llm.get_llm_client_from_provider"
    ) as mock_get_llm_client_from_provider, mock.patch(
        "rasa.shared.utils.llm.get_embedding_client_from_provider"
    ) as mock_get_embedding_client_from_provider:
        # Reset the cache as the cache is shared across tests.
        llm_factory.clear_cache()
        embedder_factory.clear_cache()

        # Call llm_factory with the first set of configs.
        llm_factory(
            {"api_type": "openai", "model": "test-gpt"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()
        # Ensure that the embedder factory is not called.
        mock_get_embedding_client_from_provider.assert_not_called()

        # Reset the mocks to track the next calls
        mock_get_llm_client_from_provider.reset_mock()

        # Call embedder_factory with the same configs.
        embedder_factory(
            {"api_type": "openai", "model": "test-gpt"},
            {"_type": "foobar", "model": "foo"},
        )
        # Ensure that the llm factory is not called.
        mock_get_llm_client_from_provider.assert_not_called()
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()

        # Reset the mocks to track the next calls
        mock_get_embedding_client_from_provider.reset_mock()

        # Call llm_factory with the same configs again.
        llm_factory(
            {"api_type": "openai", "model": "test-gpt"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_llm_client_from_provider.assert_not_called()

        # Call embedder_factory with the same configs again.
        embedder_factory(
            {"api_type": "openai", "model": "test-gpt"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_embedding_client_from_provider.assert_not_called()


def test_llm_cache_factory_for_config_keys_in_different_order() -> None:
    with mock.patch(
        "rasa.shared.utils.llm.get_llm_client_from_provider"
    ) as mock_get_llm_client_from_provider:
        # Reset the cache as the cache is shared across tests.
        llm_factory.clear_cache()

        # Call llm_factory with the 1st set of configs
        llm_factory(
            {"api_type": "openai", "model": "test-gpt"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()

        # Reset the mock to track the second call
        mock_get_llm_client_from_provider.reset_mock()

        # Call llm_factory with the 2nd set of configs (same keys, different order)
        llm_factory(
            {"model": "test-gpt", "api_type": "openai"},
            {"model": "foo", "_type": "foobar"},
        )
        # Cache hit!
        mock_get_llm_client_from_provider.assert_not_called()


def test_embedder_cache_factory() -> None:
    with mock.patch(
        "rasa.shared.utils.llm.get_embedding_client_from_provider"
    ) as mock_get_embedding_client_from_provider:
        # Reset the cache as the cache is shared across tests.
        embedder_factory.clear_cache()

        # Call embedder_factory with the 1st set of configs
        embedder_factory(
            {"api_type": "openai", "model": "test-embedding"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_embedding_client_from_provider.reset_mock()

        # Call embedder_factory with the 2nd set of configs
        embedder_factory(
            {"api_type": "openai", "model": "test-embedding-1000"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_embedding_client_from_provider.reset_mock()

        # Call embedder_factory with the 3rd set of configs
        embedder_factory(
            {"api_type": "openai", "model": "test-embedding"},
            {"_type": "buzz", "model": "foo"},
        )
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_embedding_client_from_provider.reset_mock()

        # Call embedder_factory with the 1st set of configs again
        embedder_factory(
            {"api_type": "openai", "model": "test-embedding"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_embedding_client_from_provider.assert_not_called()

        # Call embedder_factory with the 3rd set of configs again
        embedder_factory(
            {"api_type": "openai", "model": "test-embedding"},
            {"_type": "buzz", "model": "foo"},
        )
        # Cache hit!
        mock_get_embedding_client_from_provider.assert_not_called()


def test_embedder_cache_factory_for_config_keys_in_different_order() -> None:
    with mock.patch(
        "rasa.shared.utils.llm.get_embedding_client_from_provider"
    ) as mock_get_embedding_client_from_provider:
        # Reset the cache as the cache is shared across tests.
        embedder_factory.clear_cache()

        # Call embedder_factory with the 1st set of configs
        embedder_factory(
            {"api_type": "openai", "model": "test-embedding"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()

        # Reset the mock to track the second call
        mock_get_embedding_client_from_provider.reset_mock()

        # Call embedder_factory with the 2nd set of configs (same keys, different order)
        embedder_factory(
            {"model": "test-embedding", "api_type": "openai"},
            {"model": "foo", "_type": "foobar"},
        )
        # Cache hit!
        mock_get_embedding_client_from_provider.assert_not_called()


def test_to_show_that_cache_is_persisted_across_different_calls() -> None:
    with mock.patch(
        "rasa.shared.utils.llm.get_embedding_client_from_provider"
    ) as mock_get_embedding_client_from_provider:
        # Cache is not reset, hence the cache is shared across tests.
        # Call embedder_factory with the config used in the previous test -
        # test_embedder_cache_factory_for_config_keys_in_different_order.
        embedder_factory(
            {"api_type": "openai", "model": "test-embedding"},
            {"_type": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_embedding_client_from_provider.assert_not_called()
