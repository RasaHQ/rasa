from typing import Text, Any, Dict, Optional
from unittest.mock import patch

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
from rasa.shared.providers.embedding.huggingface_local_embedding_client import (
    HuggingFaceLocalEmbeddingClient,
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
    combine_custom_and_default_config,
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
        # Relying on provider
        ({"provider": "openai"}, "openai"),
        ({"provider": "azure"}, "azure"),
        # Using deprecated provider aliases for openai and azure
        ({"_type": "openai"}, "openai"),
        ({"type": "openai"}, "openai"),
        ({"type": "azure"}, "azure"),
        ({"_type": "azure"}, "azure"),
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
        ({"model_name": "gpt-4"}, "openai"),
        # Support self-hosted LLM client
        ({"provider": "self-hosted"}, "self-hosted"),
    ),
)
def test_get_provider_from_config(config: dict, expected_provider: Optional[str]):
    # When
    provider = get_provider_from_config(config)
    assert provider == expected_provider


@pytest.mark.parametrize(
    "config",
    (  # Using unsupported model_name for default client configs.
        {"model_name": "cohere/command"},
        {"model_name": "bedrock/test-model-on-bedrock"},
        {"model_name": "mistral/mistral-medium-latest"},
    ),
)
def test_get_provider_from_config_throws_error(config: dict):
    # When
    with pytest.raises(SystemExit):
        get_provider_from_config(config)


def test_llm_factory(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(None, {"model": "openai/test-gpt", "provider": "openai"})
    assert isinstance(llm, OpenAILLMClient)


@pytest.mark.parametrize(
    "config,"
    "expected_model,"
    "expected_api_type,"
    "expected_api_base,"
    "expected_api_version",
    (
        (
            {"model": "openai/test-gpt", "provider": "openai"},
            "openai/test-gpt",
            "openai",
            None,
            None,
        ),
        # Use deprecated provider aliases
        (
            {"model": "openai/test-gpt", "type": "openai"},
            "openai/test-gpt",
            "openai",
            None,
            None,
        ),
        (
            {"model": "openai/test-gpt", "_type": "openai"},
            "openai/test-gpt",
            "openai",
            None,
            None,
        ),
        # No LiteLLM prefix, but a known model
        ({"model": "gpt-4", "provider": "openai"}, "gpt-4", "openai", None, None),
        # Deprecated 'model_name'
        (
            {"model_name": "openai/test-gpt", "provider": "openai"},
            "openai/test-gpt",
            "openai",
            None,
            None,
        ),
        # With api_base and deprecated aliases
        (
            {
                "provider": "openai",
                "model": "gpt-4",
                "api_base": "https://my-test-base",
            },
            "gpt-4",
            "openai",
            "https://my-test-base",
            None,
        ),
        (
            {
                "provider": "openai",
                "model": "gpt-4",
                "openai_api_base": "https://my-test-base",
            },
            "gpt-4",
            "openai",
            "https://my-test-base",
            None,
        ),
        # With api_version and deprecated aliases
        (
            {"model": "gpt-4", "api_version": "v1", "provider": "openai"},
            "gpt-4",
            "openai",
            None,
            "v1",
        ),
        (
            {"model": "gpt-4", "openai_api_version": "v2", "provider": "openai"},
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
    client = llm_factory(config, {"provider": "openai"})

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
        llm_factory(
            {"model": "openai/gpt-4", "provider": "openai"}, {"provider": "openai"}
        )


@pytest.mark.parametrize(
    "config,"
    "expected_deployment,"
    "expected_api_type,"
    "expected_api_base,"
    "expected_api_version",
    (
        (
            {
                "provider": "azure",
                "deployment": "azure/my-test-gpt-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            "azure/my-test-gpt-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        # Use deprecated provider aliases
        (
            {
                "type": "azure",
                "deployment": "azure/my-test-gpt-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            "azure/my-test-gpt-deployment-on-azure",
            "azure",
            "https://my-test-base",
            "v1",
        ),
        (
            {
                "_type": "azure",
                "deployment": "azure/my-test-gpt-deployment-on-azure",
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
                "provider": "azure",
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
                "provider": "azure",
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
    client = llm_factory(config, {"provider": "xyz"})

    # Then
    assert isinstance(client, AzureOpenAILLMClient)
    assert client.deployment == expected_deployment
    assert client.api_type == expected_api_type
    assert client.api_base == expected_api_base
    assert client.api_version == expected_api_version


def test_llm_factory_returns_azure_openai_llm_client_without_specified_provider_key(
    monkeypatch: MonkeyPatch,
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv("AZURE_API_KEY", "test")

    # Do not specify provider key. This is tolerated by llm_factory for now,
    # because of backward compatibility
    config = {
        "deployment": "azure/my-test-gpt-deployment-on-azure",
        "api_base": "https://my-test-base",
        "api_version": "v1",
        "api_type": "azure",
    }

    # When
    client = llm_factory(config, {"provider": "openai"})

    # Then
    assert isinstance(client, AzureOpenAILLMClient)
    assert client.deployment == config["deployment"]
    assert client.api_type == config["api_type"]
    assert client.api_base == config["api_base"]
    assert client.api_version == config["api_version"]


def test_llm_factory_returns_azure_openai_llm_client_with_env_vars_settings(
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setenv("AZURE_API_KEY", "test")
    monkeypatch.setenv("AZURE_API_BASE", "https://my-test-base")
    monkeypatch.setenv("AZURE_API_VERSION", "v1")
    client = llm_factory(
        {"deployment": "azure/my-test-gpt-deployment-on-azure", "provider": "azure"},
        {"provider": "openai"},
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
                {"provider": "openai"},
            )
        monkeypatch.delenv(env_var, raising=False)


@pytest.mark.parametrize(
    "config, api_key_env",
    (
        ({"model": "cohere/command", "provider": "cohere"}, "COHERE_API_KEY"),
        ({"model": "command", "provider": "cohere"}, "COHERE_API_KEY"),
        ({"model": "anthropic/claude", "provider": "anthropic"}, "ANTHROPIC_API_KEY"),
        ({"model": "claude", "provider": "anthropic"}, "ANTHROPIC_API_KEY"),
        ({"model": "some-random-model", "provider": "buzz-ai"}, "BUZZ_AI_API_KEY"),
    ),
)
def test_llm_factory_returns_default_litellm_client(
    config: dict, api_key_env: str, monkeypatch: MonkeyPatch
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv(api_key_env, "test")
    # When
    client = llm_factory(config, {"provider": "openai"})
    # Then
    assert isinstance(client, DefaultLiteLLMClient)
    assert client.model == config["model"]
    assert client.provider == config["provider"]


def test_llm_factory_raises_exception_when_default_client_setup_is_invalid():
    # Given
    # config not containing `model` key
    config = {"some_random_key": "cohere/command"}
    # When / Then
    with pytest.raises(ValueError):
        llm_factory(config, {"provider": "openai"})


def test_llm_factory_uses_custom_provider(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(
        {"provider": "openai", "model": "test-gpt"},
        {"provider": "foobar", "model": "foo"},
    )
    assert isinstance(llm, OpenAILLMClient)


def test_llm_factory_ignores_irrelevant_default_args(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    # since the types of the custom config and the default are different
    # all default arguments should be removed.
    llm = llm_factory(
        {"provider": "openai", "model": "test-gpt"},
        {"provider": "foobar", "temperature": -1},
    )
    assert isinstance(llm, OpenAILLMClient)
    # since the default argument should be removed, this should be the default -
    # which is not -1
    assert llm._extra_parameters.get("temperature") != -1


def test_llm_factory_uses_additional_args_from_custom(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory({"temperature": -1}, {"provider": "openai", "model": "test-gpt"})
    assert isinstance(llm, OpenAILLMClient)
    assert llm._extra_parameters.get("temperature") == -1


def test_embedder_factory(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    embedder = embedder_factory(None, {"provider": "openai", "model": "test-embedding"})
    assert isinstance(embedder, OpenAIEmbeddingClient)


@pytest.mark.parametrize(
    "config,"
    "expected_model,"
    "expected_api_type,"
    "expected_api_base,"
    "expected_api_version",
    (
        (
            {"model": "openai/test-embeddings", "provider": "openai"},
            "openai/test-embeddings",
            "openai",
            None,
            None,
        ),
        # Deprecated `provider` aliases
        (
            {"model": "openai/test-embeddings", "_type": "openai"},
            "openai/test-embeddings",
            "openai",
            None,
            None,
        ),
        (
            {"model": "openai/test-embeddings", "type": "openai"},
            "openai/test-embeddings",
            "openai",
            None,
            None,
        ),
        # Deprecated `model_name`
        (
            {"model_name": "test-embeddings", "provider": "openai"},
            "test-embeddings",
            "openai",
            None,
            None,
        ),
        # With `api_type` deprecated aliases
        (
            {
                "model": "test-embeddings",
                "provider": "openai",
                "openai_api_type": "openai",
            },
            "test-embeddings",
            "openai",
            None,
            None,
        ),
        # With `api_base` and deprecated aliases
        (
            {
                "provider": "openai",
                "model": "test-embeddings",
                "api_base": "https://my-test-base",
            },
            "test-embeddings",
            "openai",
            "https://my-test-base",
            None,
        ),
        (
            {
                "provider": "openai",
                "model": "test-embeddings",
                "openai_api_base": "https://my-test-base",
            },
            "test-embeddings",
            "openai",
            "https://my-test-base",
            None,
        ),
        # With `api_version` and deprecated aliases
        (
            {"model": "test-embeddings", "api_version": "v1", "provider": "openai"},
            "test-embeddings",
            "openai",
            None,
            "v1",
        ),
        (
            {
                "provider": "openai",
                "model": "test-embeddings",
                "openai_api_version": "v2",
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
    client = embedder_factory(config, {"provider": "openai"})

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
        embedder_factory(
            {"model": "openai/gpt-4", "provider": "openai"}, {"provider": "openai"}
        )


@pytest.mark.parametrize(
    "config,"
    "expected_deployment,"
    "expected_api_type,"
    "expected_api_base,"
    "expected_api_version",
    (
        (
            {
                "provider": "azure",
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
        # Deprecated `provider` aliases
        (
            {
                "type": "azure",
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
        (
            {
                "_type": "azure",
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
                "provider": "azure",
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
                "provider": "azure",
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
    client = embedder_factory(config, {"provider": "xyz"})

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
                    "provider": "azure",
                    "deployment": "my-test-embedding-deployment-on-azure",
                },
                {"provider": "openai"},
            )
        monkeypatch.delenv(env_var, raising=False)


def test_embedder_factory_returns_azure_openai_embedding_client_without_specified_provider_key(  # noqa: E501
    monkeypatch: MonkeyPatch,
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv("AZURE_API_KEY", "test")

    # Do not specify provider key. This is tolerated by llm_factory for now,
    # because of backward compatibility
    config = {
        "deployment": "azure/my-test-embedding-deployment-on-azure",
        "api_base": "https://my-test-base",
        "api_version": "v1",
        "api_type": "azure",
    }

    # When
    client = embedder_factory(config, {"provider": "openai"})

    # Then
    assert isinstance(client, AzureOpenAIEmbeddingClient)
    assert client.deployment == config["deployment"]
    assert client.api_type == config["api_type"]
    assert client.api_base == config["api_base"]
    assert client.api_version == config["api_version"]


@pytest.mark.parametrize(
    "config, api_key_env",
    (
        (
            {"model": "cohere/embed-english-v3.0", "provider": "cohere"},
            "COHERE_API_KEY",
        ),
        (
            {"model": "huggingface/microsoft/codebert-base", "provider": "huggingface"},
            "HUGGINGFACE_API_KEY",
        ),
    ),
)
def test_embedder_factory_returns_default_litellm_client(
    config: dict, api_key_env: str, monkeypatch: MonkeyPatch
):
    # Given
    # Client cannot be instantiated without the required environment variable
    monkeypatch.setenv(api_key_env, "test")
    # When
    client = embedder_factory(config, {"provider": "openai"})
    # Then
    assert isinstance(client, DefaultLiteLLMEmbeddingClient)


def test_embedder_factory_raises_exception_when_default_client_setup_is_invalid():
    # Given
    # config not containing `model` key
    config = {"some_random_key": "cohere/command"}
    # When / Then
    with pytest.raises(ValueError):
        embedder_factory(config, {"provider": "openai"})


def test_embedder_factory_uses_custom_provider(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    embedder = embedder_factory(
        {"provider": "openai", "model": "test-embedding"},
        {"provider": "foobar", "model": "foo"},
    )
    assert isinstance(embedder, OpenAIEmbeddingClient)


@pytest.mark.parametrize(
    "config," "expected_model",
    [
        (
            {"provider": "huggingface_local", "model": "hf-repo/model_name"},
            "hf-repo/model_name",
        ),
        # Deprecated `provider` aliases
        (
            {"type": "huggingface_local", "model": "hf-repo/model_name"},
            "hf-repo/model_name",
        ),
        (
            {"_type": "huggingface_local", "model": "hf-repo/model_name"},
            "hf-repo/model_name",
        ),
        # Deprecated combination of `type: huggingface`
        (
            {"type": "huggingface", "model": "hf-repo/model_name"},
            "hf-repo/model_name",
        ),
        (
            {"_type": "huggingface", "model": "hf-repo/model_name"},
            "hf-repo/model_name",
        ),
    ],
)
def test_embedder_factory_returns_huggingface_local_embedding_client(
    config: dict,
    expected_model: str,
    monkeypatch: MonkeyPatch,
):
    # When
    with patch(
        "rasa.shared.providers.embedding.huggingface_local_embedding_client"
        ".HuggingFaceLocalEmbeddingClient._init_client"
    ) as mock_init_client, patch(
        "rasa.shared.providers.embedding.huggingface_local_embedding_client"
        ".HuggingFaceLocalEmbeddingClient._validate_if_sentence_transformers_installed"
    ) as mock_validate_if_sentence_transformers_installed:
        mock_init_client.return_value = None
        mock_validate_if_sentence_transformers_installed.return_value = None

        client = embedder_factory(config, {"provider": "xyz"})

    # Then
    assert isinstance(client, HuggingFaceLocalEmbeddingClient)
    assert client.model == expected_model


@pytest.mark.parametrize(
    "config",
    [
        # `model` not provided
        {"provider": "huggingface_local"},
        # `model` not provided, deprecated configs
        {"type": "huggingface_local"},
        {"_type": "huggingface_local"},
    ],
)
def test_embedder_factory_raises_exception_when_huggingface_local_embedding_client_config_is_invalid(  # noqa: E501
    config,
):
    # When / Then
    with pytest.raises(ValueError):
        embedder_factory(config, {"provider": "xyz"})


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
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_llm_client_from_provider.reset_mock()

        # Call llm_factory with the second set of configs.
        llm_factory(
            {"provider": "openai", "model": "test-gpt-1000"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_llm_client_from_provider.reset_mock()

        # Call llm_factory with the third set of configs.
        llm_factory(
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "buzz", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_llm_client_from_provider.reset_mock()

        # Call llm_factory with the first set of configs again
        llm_factory(
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_llm_client_from_provider.assert_not_called()

        # Call llm_factory with the second set of configs again
        llm_factory(
            {"provider": "openai", "model": "test-gpt-1000"},
            {"provider": "foobar", "model": "foo"},
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
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()
        # Ensure that the embedder factory is not called.
        mock_get_embedding_client_from_provider.assert_not_called()

        # Reset the mocks to track the next calls
        mock_get_llm_client_from_provider.reset_mock()

        # Call embedder_factory with the same configs.
        embedder_factory(
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "model": "foo"},
        )
        # Ensure that the llm factory is not called.
        mock_get_llm_client_from_provider.assert_not_called()
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()

        # Reset the mocks to track the next calls
        mock_get_embedding_client_from_provider.reset_mock()

        # Call llm_factory with the same configs again.
        llm_factory(
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_llm_client_from_provider.assert_not_called()

        # Call embedder_factory with the same configs again.
        embedder_factory(
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "model": "foo"},
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
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_llm_client_from_provider.assert_called_once()

        # Reset the mock to track the second call
        mock_get_llm_client_from_provider.reset_mock()

        # Call llm_factory with the 2nd set of configs (same keys, different order)
        llm_factory(
            {"model": "test-gpt", "provider": "openai"},
            {"model": "foo", "provider": "foobar"},
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
            {"provider": "openai", "model": "test-embedding"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_embedding_client_from_provider.reset_mock()

        # Call embedder_factory with the 2nd set of configs
        embedder_factory(
            {"provider": "openai", "model": "test-embedding-1000"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_embedding_client_from_provider.reset_mock()

        # Call embedder_factory with the 3rd set of configs
        embedder_factory(
            {"provider": "openai", "model": "test-embedding"},
            {"provider": "buzz", "model": "foo"},
        )
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()
        # Reset the mock to track the next call
        mock_get_embedding_client_from_provider.reset_mock()

        # Call embedder_factory with the 1st set of configs again
        embedder_factory(
            {"provider": "openai", "model": "test-embedding"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_embedding_client_from_provider.assert_not_called()

        # Call embedder_factory with the 3rd set of configs again
        embedder_factory(
            {"provider": "openai", "model": "test-embedding"},
            {"provider": "buzz", "model": "foo"},
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
            {"provider": "openai", "model": "test-embedding"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache miss!
        mock_get_embedding_client_from_provider.assert_called_once()

        # Reset the mock to track the second call
        mock_get_embedding_client_from_provider.reset_mock()

        # Call embedder_factory with the 2nd set of configs (same keys, different order)
        embedder_factory(
            {"model": "test-embedding", "provider": "openai"},
            {"model": "foo", "provider": "foobar"},
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
            {"provider": "openai", "model": "test-embedding"},
            {"provider": "foobar", "model": "foo"},
        )
        # Cache hit!
        mock_get_embedding_client_from_provider.assert_not_called()


@pytest.mark.parametrize(
    "custom_config," "expected_combined_config,",
    (  # Test cases for the client - OpenAI.
        (
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        # Deprecated `provider` aliases
        (
            {
                "_type": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        (
            {
                "type": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        # Missing provider, supports backward compatibility
        (
            {
                "api_type": "openai",
                "model": "test-gpt",
            },
            {
                "provider": "openai",
                "api_type": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        (
            {"model": "gpt-4"},
            {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        # Missing provider and uses deprecated aliases
        (
            {
                "type": "openai",
                "model_name": "test-gpt",
            },
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        (
            {
                "api_type": "openai",
                "model_name": "gpt-4",
            },
            {
                "provider": "openai",
                "model": "gpt-4",
                "api_type": "openai",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        (
            {
                "api_type": "openai",
                "model_name": "invalid_model",
            },
            {
                "provider": "openai",
                "model": "invalid_model",
                "api_type": "openai",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        # litellm way of defining model.
        (
            {
                "api_type": "openai",
                "model": "openai/gpt-4",
            },
            {
                "provider": "openai",
                "model": "openai/gpt-4",
                "api_type": "openai",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        (
            {
                "model_name": "openai/gpt-4",
            },
            {
                "provider": "openai",
                "model": "openai/gpt-4",
                "temperature": 0.0,
                "max_tokens": 256,
                "timeout": 7,
            },
        ),
        # ------------------------------------------------------------------------------
        # Test cases for the client - Azure.
        (
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        # Deprecated `provider` aliases
        (
            {
                "type": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        (
            {
                "_type": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        # Missing provider, supports backward compatibility
        (
            {
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        # Missing provider and uses deprecated aliases
        (
            {
                "engine": "my-test-embedding-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        # Missing provider and api_type
        (
            {
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        # Deprecated aliases
        (
            {
                "provider": "azure",
                "deployment_name": "my-test-embedding-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        (
            {
                "provider": "azure",
                "engine": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
            },
        ),
        (
            {
                "provider": "azure",
                "engine": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "request_timeout": 10,
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "timeout": 10,
            },
        ),
        # litellm way of defining model.
        (
            {
                "model": "azure/gpt-4",
            },
            {
                "provider": "azure",
                "model": "azure/gpt-4",
            },
        ),
        (
            {
                "model_name": "azure/gpt-4",
            },
            {
                "provider": "azure",
                "model": "azure/gpt-4",
            },
        ),
        # ------------------------------------------------------------------------------
        # Test cases for the client - Default.
        (
            {
                "provider": "mistral",
                "model": "mistral/mistral-medium",
            },
            {
                "provider": "mistral",
                "model": "mistral/mistral-medium",
            },
        ),
        # Using deprecated request_timeout
        (
            {
                "provider": "mistral",
                "model": "mistral/mistral-medium",
                "request_timeout": 10,
            },
            {
                "provider": "mistral",
                "model": "mistral/mistral-medium",
                "timeout": 10,
            },
        ),
        # Missing provider.
        (
            {
                "model": "mistral/mistral-medium",
            },
            {
                "provider": "mistral",
                "model": "mistral/mistral-medium",
            },
        ),
        (
            {
                "model": "mistral/some-model",
            },
            {
                "provider": "mistral",
                "model": "mistral/some-model",
            },
        ),
        # ------------------------------------------------------------------------------
        # Test cases for the client - self hosted.
        (
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
            },
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
            },
        ),
        # With provider deprecated aliases
        (
            {
                "model": "some_model",
                "api_base": "http://localhost:8000",
                "_type": "self-hosted",
            },
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
            },
        ),
        (
            {
                "model": "some_model",
                "api_base": "http://localhost:8000",
                "type": "self-hosted",
            },
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
            },
        ),
        # with api_base, api_type and api_version deprecated aliases
        (
            {
                "provider": "self-hosted",
                "model": "some_model",
                "openai_api_base": "http://localhost:8000",
                "openai_api_type": "openai",
                "openai_api_version": "v1",
            },
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
                "api_type": "openai",
                "api_version": "v1",
            },
        ),
        # with model deprecated aliases
        (
            {
                "provider": "self-hosted",
                "model_name": "some_model",
                "api_base": "http://localhost:8000",
            },
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
            },
        ),
        # with request_timeout deprecated aliases
        (
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
                "request_timeout": 10,
            },
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
                "timeout": 10,
            },
        ),
    ),
)
def test_combine_custom_and_default_config(
    custom_config: Dict[str, Any], expected_combined_config: Dict[str, Any]
) -> None:
    default_config = {
        "provider": "openai",
        "model": "test-gpt",
        "temperature": 0.0,
        "max_tokens": 256,
        "timeout": 7,
    }
    combined_config = combine_custom_and_default_config(custom_config, default_config)

    assert combined_config == expected_combined_config


@pytest.mark.parametrize(
    "custom_config",
    (  # Test cases for the client - Default.
        {
            "provider": "mistral",
            "model_name": "mistral/some-model",
        },
    ),
)
def test_combine_custom_and_default_config_throw_error(
    custom_config: Dict[str, Any],
) -> None:
    default_config = {
        "provider": "openai",
        "model": "test-gpt",
        "temperature": 0.0,
        "max_tokens": 256,
        "timeout": 7,
    }

    with pytest.raises(SystemExit):
        combine_custom_and_default_config(custom_config, default_config)
