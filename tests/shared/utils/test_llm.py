from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Text
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from pytest import MonkeyPatch

from rasa.core.agent import Agent
from rasa.core.config.configuration import Configuration
from rasa.core.nlg.contextual_response_rephraser import (
    DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE,
)
from rasa.core.policies.enterprise_search_policy import (
    DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE,
    DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_CITATION_TEMPLATE,
    DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_RELEVANCY_CHECK_AND_CITATION_TEMPLATE,
)
from rasa.dialogue_understanding.generator import LLMBasedCommandGenerator
from rasa.dialogue_understanding.generator.single_step.compact_llm_command_generator import (  # noqa: E501
    DEFAULT_COMMAND_PROMPT_TEMPLATE_FILE_NAME,
    FALLBACK_COMMAND_PROMPT_TEMPLATE_FILE_NAME,
    MODEL_PROMPT_MAPPER,
    get_default_prompt_template_based_on_model,
)
from rasa.exceptions import ValidationError
from rasa.shared.constants import (
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_KEY_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    MODEL_GROUP_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    AgentCancelled,
    AgentCompleted,
    AgentInterrupted,
    AgentResumed,
    AgentStarted,
    BotUttered,
    Event,
    Restarted,
    SessionStarted,
    UserUttered,
)
from rasa.shared.core.slots import (
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    Slot,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.engine.caching import CACHE_LOCATION_ENV
from rasa.shared.exceptions import (
    InvalidConfigException,
    InvalidPromptTemplateException,
    ProviderClientValidationError,
)
from rasa.shared.providers.embedding.azure_openai_embedding_client import (
    AzureOpenAIEmbeddingClient,
)
from rasa.shared.providers.embedding.default_litellm_embedding_client import (
    DefaultLiteLLMEmbeddingClient,
)
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.embedding.huggingface_local_embedding_client import (
    HuggingFaceLocalEmbeddingClient,
)
from rasa.shared.providers.embedding.litellm_router_embedding_client import (
    LiteLLMRouterEmbeddingClient,
)
from rasa.shared.providers.embedding.openai_embedding_client import (
    OpenAIEmbeddingClient,
)
from rasa.shared.providers.llm.azure_openai_llm_client import AzureOpenAILLMClient
from rasa.shared.providers.llm.default_litellm_llm_client import DefaultLiteLLMClient
from rasa.shared.providers.llm.litellm_router_llm_client import LiteLLMRouterLLMClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.llm.openai_llm_client import OpenAILLMClient
from rasa.shared.providers.router.router_client import RouterClient
from rasa.shared.utils.common import all_subclasses
from rasa.shared.utils.llm import (
    ERROR_PLACEHOLDER,
    SystemPrompts,
    _get_enterprise_search_prompt,
    _get_llm_command_generator_config,
    allowed_values_for_slot,
    combine_custom_and_default_config,
    create_tracker_for_user_step,
    embedder_client_factory,
    embedder_factory,
    embedder_router_factory,
    ensure_cache,
    generate_sender_id,
    get_prompt_template,
    get_provider_from_config,
    get_system_default_prompts,
    llm_client_factory,
    llm_factory,
    llm_router_factory,
    resolve_model_client_config,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
)
from rasa.shared.utils.yaml import read_yaml


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


@pytest.mark.parametrize(
    "events, expected_response",
    [
        # Agent started and completed
        (
            [
                UserUttered("Hi"),
                BotUttered("Hi, how can I help you"),
                UserUttered("start agent 1"),
                AgentStarted("agent_1", "flow_1"),
                BotUttered("This is an agent message"),
                UserUttered("stop agent 1"),
                AgentCompleted("agent_1", "flow_1"),
                BotUttered("What else can I help you with?"),
            ],
            "USER: Hi\nAI: Hi, how can I help you\nUSER: start agent 1\nagent_1: This is an agent message\nUSER: stop agent 1\nAI: What else can I help you with?",  # noqa: E501
        ),
        # Agent started, interrupted, resumed, and cancelled
        (
            [
                UserUttered("Hi"),
                BotUttered("Hi, how can I help you"),
                UserUttered("start agent 1"),
                AgentStarted("agent_1", "flow_1"),
                BotUttered("This is an agent message"),
                UserUttered("interrupt agent 1"),
                AgentInterrupted("agent_1", "flow_1"),
                UserUttered("some message"),
                BotUttered("some response"),
                UserUttered("resume agent 1"),
                AgentResumed("agent_1", "flow_1"),
                BotUttered("some response"),
                UserUttered("cancel agent 1"),
                AgentCancelled("agent_1", "flow_1"),
            ],
            "USER: Hi\nAI: Hi, how can I help you\nUSER: start agent 1\nagent_1: This is an agent message\nUSER: interrupt agent 1\nUSER: some message\nAI: some response\nUSER: resume agent 1\nagent_1: some response\nUSER: cancel agent 1",  # noqa: E501
        ),
        # One agent interrupted another agent
        (
            [
                UserUttered("Hi"),
                BotUttered("Hi, how can I help you"),
                UserUttered("start agent 1"),
                AgentStarted("agent_1", "flow_1"),
                BotUttered("This is an agent message"),
                UserUttered("start agent 2"),
                AgentStarted("agent_2", "flow_2"),
                BotUttered("This is an agent message"),
                UserUttered("some message"),
                BotUttered("some response"),
                AgentCompleted("agent_2", "flow_2"),
                AgentResumed("agent_1", "flow_1"),
                BotUttered("This is an agent message"),
                UserUttered("finished agent 1"),
                AgentCompleted("agent_1", "flow_1"),
            ],
            "USER: Hi\nAI: Hi, how can I help you\nUSER: start agent 1\nagent_1: This is an agent message\nUSER: start agent 2\nagent_2: This is an agent message\nUSER: some message\nagent_2: some response\nagent_1: This is an agent message\nUSER: finished agent 1",  # noqa: E501
        ),
    ],
)
def test_tracker_as_readable_transcript_highlight_agent_turns(
    events: List[Event],
    expected_response: Text,
    domain: Domain,
):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(events)

    # When
    response = tracker_as_readable_transcript(tracker, highlight_agent_turns=True)

    # Then
    assert response.strip() == expected_response.strip()


def test_sanitize_message_for_prompt_handles_none():
    assert sanitize_message_for_prompt(None) == ""


def test_sanitize_message_for_prompt_handles_empty_string():
    assert sanitize_message_for_prompt("") == ""


def test_sanitize_message_for_prompt_handles_string_with_newlines():
    assert sanitize_message_for_prompt("hello\nworld") == "hello world"


@pytest.mark.parametrize(
    "config, expected_provider",
    (
        # LiteLLM naming convention without specifying the provider key
        # should return None, because inference of provider from 'model'
        # is not allowed for default clients
        ({"model": "cohere/command"}, None),
        ({"model": "bedrock/test-model-on-bedrock"}, None),
        ({"model": "azure/my-test-gpt-deployment"}, None),
        ({"model": "openai/test-gpt"}, None),
        ({"model": "gpt-4"}, None),
        ({"model": "huggingface/some-huggingface-model"}, None),
        # Relying on provider
        ({"provider": "openai"}, "openai"),
        ({"provider": "azure"}, "azure"),
        ({"provider": "huggingface_local"}, "huggingface_local"),
        ({"provider": "self-hosted"}, "self-hosted"),
        ({"model": "cohere/command", "provider": "cohere"}, "cohere"),
        ({"model": "bedrock/test-model-on-bedrock", "provider": "bedrock"}, "bedrock"),
        # Using deprecated provider aliases for openai and azure
        ({"_type": "openai"}, "openai"),
        ({"type": "openai"}, "openai"),
        ({"type": "azure"}, "azure"),
        ({"_type": "azure"}, "azure"),
        # Using deprecated provider alias for hugging face local embeddings
        ({"type": "huggingface"}, "huggingface_local"),
        ({"_type": "huggingface"}, "huggingface_local"),
        # Deprecated provider aliases are not allowed for other providers
        ({"_type": "cohere"}, None),
        # Relying on azure openai specific config
        ({"deployment": "my-test-deployment-on-azure"}, "azure"),
        ({"deployment": "left-over-key", "provider": "ollama"}, "ollama"),
    ),
)
def test_get_provider_from_config(config: dict, expected_provider: Optional[str]):
    # When
    provider = get_provider_from_config(config)
    assert provider == expected_provider


class TestLLMFactory:
    @pytest.fixture
    def default_model_configuration(self, monkeypatch: MonkeyPatch) -> Dict:
        monkeypatch.setenv("COHERE_API_KEY", "dummy_key_cohere")
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "dummy_key_openai")
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "dummy_key_azure")
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "dummy_base_azure")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "dummy_version_azure")

        return {
            "provider": "openai",
            "model": "gpt-4",
            "timeout": 10,
            "num_retries": 5,
        }

    @pytest.mark.parametrize(
        "custom_config, expected_client, api_key",
        (
            (
                {
                    "provider": "cohere",
                    "model": "test-cohere",
                },
                DefaultLiteLLMClient,
                "COHERE_API_KEY",
            ),
            (
                {
                    "provider": "openai",
                    "model": "openai/test-gpt",
                },
                OpenAILLMClient,
                "OPENAI_API_KEY",
            ),
            (
                {
                    "provider": "azure",
                    "deployment": "azure/my-test-gpt-deployment-on-azure",
                    "api_base": "https://my-test-base",
                    "api_version": "v1",
                },
                AzureOpenAILLMClient,
                "AZURE_API_KEY",
            ),
        ),
    )
    def test_correctly_initializes_llm_clients(
        self,
        custom_config,
        expected_client,
        api_key,
        default_model_configuration,
        monkeypatch,
    ):
        monkeypatch.setenv(api_key, "test")
        client = llm_factory(custom_config, default_model_configuration)
        assert isinstance(client, expected_client)

    def test_correctly_initializes_router_clients(self, default_model_configuration):
        router_config = {
            "id": "test-model-group-id",
            "models": [
                {"provider": "cohere", "model": "test-cohere", "api_key": "test"},
                {"provider": "openai", "model": "gpt-4", "api_key": "test"},
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "test-api-base",
                },
                {
                    "provider": "self-hosted",
                    "model": "test-model",
                    "api_base": "test-api-base",
                    "api_key": "test",
                    "api_version": "test-api-version",
                },
            ],
            "router": {
                "routing_strategy": "test",
                "use_chat_completions_endpoint": True,
            },
        }
        client = llm_factory(router_config, default_model_configuration)
        assert isinstance(client, LLMClient)
        assert isinstance(client, RouterClient)
        # Currently, this is one and only implementation of RouterClient, this might
        # change in the future
        assert isinstance(client, LiteLLMRouterLLMClient)

    def test_initializes_llm_client_when_router_is_not_present(
        self, default_model_configuration, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        # This configuration is expected to be returned when llm config is
        # resolved
        router_config = {
            "id": "valid_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
            # router: {...} not present
        }
        client = llm_factory(router_config, default_model_configuration)
        assert isinstance(client, LLMClient)
        assert isinstance(client, OpenAILLMClient)


class TestLLMClientFactory:
    def test_llm_client_factory(self, monkeypatch: MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")

        llm = llm_client_factory(
            None, {"model": "openai/test-gpt", "provider": "openai"}
        )
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
    def test_returns_openai_llm_client(
        self,
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
        client = llm_client_factory(config, {"provider": "openai"})

        # Then
        assert isinstance(client, OpenAILLMClient)
        assert client.model == expected_model
        assert client.api_type == expected_api_type
        assert client.api_base == expected_api_base
        assert client.api_version == expected_api_version

    def test_raises_exception_when_openai_client_setup_is_invalid(
        self,
        monkeypatch: MonkeyPatch,
    ):
        """OpenAI client requires the OPENAI_API_KEY environment variable
        to be set.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderClientValidationError):
            llm_client_factory(
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
    def test_returns_azure_openai_llm_client(
        self,
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
        client = llm_client_factory(config, {"provider": "xyz"})

        # Then
        assert isinstance(client, AzureOpenAILLMClient)
        assert client.deployment == expected_deployment
        assert client.api_type == expected_api_type
        assert client.api_base == expected_api_base
        assert client.api_version == expected_api_version

    def test_returns_azure_openai_llm_client_without_specified_provider_key(
        self,
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
        client = llm_client_factory(config, {"provider": "openai"})

        # Then
        assert isinstance(client, AzureOpenAILLMClient)
        assert client.deployment == config["deployment"]
        assert client.api_type == config["api_type"]
        assert client.api_base == config["api_base"]
        assert client.api_version == config["api_version"]

    def test_returns_azure_openai_llm_client_with_env_vars_settings(
        self,
        monkeypatch: MonkeyPatch,
    ):
        monkeypatch.setenv("AZURE_API_KEY", "test")
        monkeypatch.setenv("AZURE_API_BASE", "https://my-test-base")
        monkeypatch.setenv("AZURE_API_VERSION", "v1")
        client = llm_client_factory(
            {
                "deployment": "azure/my-test-gpt-deployment-on-azure",
                "provider": "azure",
            },
            {"provider": "openai"},
        )
        assert isinstance(client, AzureOpenAILLMClient)
        assert client.deployment == "azure/my-test-gpt-deployment-on-azure"
        assert client.api_type == "azure"
        assert client.api_base == "https://my-test-base"
        assert client.api_version == "v1"

    def test_raises_exception_when_azure_openai_client_setup_is_invalid(
        self,
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
                llm_client_factory(
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
            (
                {"model": "anthropic/claude", "provider": "anthropic"},
                "ANTHROPIC_API_KEY",
            ),
            ({"model": "claude", "provider": "anthropic"}, "ANTHROPIC_API_KEY"),
            ({"model": "some-random-model", "provider": "buzz-ai"}, "BUZZ_AI_API_KEY"),
        ),
    )
    def test_returns_default_litellm_client(
        self, config: dict, api_key_env: str, monkeypatch: MonkeyPatch
    ):
        # Given
        # Client cannot be instantiated without the required environment variable
        monkeypatch.setenv(api_key_env, "test")
        # When
        client = llm_client_factory(config, {"provider": "openai"})
        # Then
        assert isinstance(client, DefaultLiteLLMClient)
        assert client.model == config["model"]
        assert client.provider == config["provider"]

    def test_raises_exception_when_default_client_setup_is_invalid(
        self,
    ):
        # Given
        # config not containing `model` key
        config = {"some_random_key": "cohere/command"}
        # When / Then
        with pytest.raises(ValueError):
            llm_client_factory(config, {"provider": "openai"})

    def test_uses_custom_provider(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")

        llm = llm_client_factory(
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "model": "foo"},
        )
        assert isinstance(llm, OpenAILLMClient)

    def test_ignores_irrelevant_default_args(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        # since the types of the custom config and the default are different
        # all default arguments should be removed.
        llm = llm_client_factory(
            {"provider": "openai", "model": "test-gpt"},
            {"provider": "foobar", "temperature": -1},
        )
        assert isinstance(llm, OpenAILLMClient)
        # since the default argument should be removed, this should be the default -
        # which is not -1
        assert llm._extra_parameters.get("temperature") != -1

    def test_uses_additional_args_from_custom(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")

        llm = llm_client_factory(
            {"temperature": -1}, {"provider": "openai", "model": "test-gpt"}
        )
        assert isinstance(llm, OpenAILLMClient)
        assert llm._extra_parameters.get("temperature") == -1


class TestLLMRouterFactory:
    @pytest.fixture
    def default_model_configuration(self, monkeypatch: MonkeyPatch) -> Dict:
        monkeypatch.setenv("COHERE_API_KEY", "dummy_key_cohere")
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "dummy_key_openai")
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "dummy_key_azure")
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "dummy_base_azure")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "dummy_version_azure")

        return {
            "provider": "openai",
            "model": "gpt-4",
            "timeout": 10,
            "num_retries": 5,
        }

    def test_llm_router_factory(self, default_model_configuration: Dict):
        router_config = {
            "id": "test-model-group-id",
            "models": [
                {"provider": "cohere", "model": "test-cohere", "api_key": "test"},
                {"provider": "openai", "model": "gpt-4", "api_key": "test"},
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "test-api-base",
                },
                {
                    "provider": "self-hosted",
                    "model": "test-model",
                    "api_base": "test-api-base",
                    "api_key": "test",
                    "api_version": "test-api-version",
                },
            ],
            "router": {"routing_strategy": "test"},
        }

        router = llm_router_factory(router_config, default_model_configuration)
        assert isinstance(router, RouterClient)
        assert isinstance(router, LLMClient)
        # This for current and only implementation of Router. Could change in the
        # future.
        assert isinstance(router, LiteLLMRouterLLMClient)

    def test_router_is_initialized_with_correctly_combined_with_default_parameters(
        self, default_model_configuration
    ):
        router_config = {
            "id": "test-model-group-id",
            "models": [
                {"provider": "cohere", "model": "test-cohere", "api_key": "test"},
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "test-api-base",
                },
                {"provider": "openai", "model": "gpt-4", "api_key": "test"},
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": "test",
                    "num_retries": 100,
                    "timeout": 100,
                },
            ],
            "router": {"routing_strategy": "test"},
        }
        expected_litellm_model_configurations = [
            # For providers other than specified in default configuration,
            # everything should be as provided.
            {
                "model": "cohere/test-cohere",
                "api_key": "test",
            },
            {
                "model": "azure/test-deployment",
                "api_key": "test",
                "api_base": "test-api-base",
            },
            # For providers matching the provider specified in default configuration,
            # we expect the default parameters to be present if not provided.
            {
                "model": "openai/gpt-4",
                "api_key": "test",
                "timeout": 10,
                "num_retries": 5,
            },
            # For providers matching the provider specified in default configuration,
            # we expect the default parameters not to override the be present
            # parameters.
            {
                "model": "openai/gpt-4",
                "api_key": "test",
                "timeout": 100,
                "num_retries": 100,
            },
        ]

        router = llm_router_factory(router_config, default_model_configuration)
        assert isinstance(router, RouterClient)
        assert isinstance(router, LLMClient)
        # This for current and only implementation of Router. Could change in the
        # future.
        assert isinstance(router, LiteLLMRouterLLMClient)
        actual_litellm_model_configurations = [
            model_configuration["litellm_params"]
            for model_configuration in router.model_configurations
        ]
        assert (
            actual_litellm_model_configurations == expected_litellm_model_configurations
        )

    @pytest.mark.parametrize(
        "router_config",
        [
            # Use of forbidden 'n' parameter
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "cohere",
                        "model": "test-cohere",
                        "api_key": "test",
                        "n": 10,
                    },
                ],
                "router": {},
            },
            # Use of forbidden 'stream' parameter
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "cohere",
                        "model": "test-cohere",
                        "api_key": "test",
                        "stream": 10,
                    },
                ],
                "router": {},
            },
            # Missing "api_key"
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "azure",
                        "deployment": "test-deployment",
                        # "api_key" missing
                        "api_base": "https://example.azure.com",
                    }
                ],
                "router": {"routing_strategy": "test"},
            },
            # Missing "api_base"
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "azure",
                        "deployment": "test-deployment",
                        "api_key": "test",
                        # "api_base" missing
                    }
                ],
                "router": {"routing_strategy": "test"},
            },
            # Missing "router"
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "azure",
                        "deployment": "test-deployment",
                        "api_key": "test",
                        "api_base": "test-api-base",
                    }
                ],
                # "router": {"routing_strategy": "test"}, missing
            },
        ],
    )
    def test_raises_error_if_configuration_is_invalid(
        self,
        router_config: Dict,
        default_model_configuration: Dict,
        monkeypatch: MonkeyPatch,
    ):
        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, raising=False)
        monkeypatch.delenv(AZURE_API_BASE_ENV_VAR, raising=False)
        monkeypatch.delenv(AZURE_API_VERSION_ENV_VAR, raising=False)

        with pytest.raises((ValueError, ProviderClientValidationError)):
            llm_router_factory(router_config, default_model_configuration)


class TestEmbedderFactory:
    @pytest.fixture
    def default_model_configuration(self, monkeypatch: MonkeyPatch) -> Dict:
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "dummy_openai")
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "dummy_azure")
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "dummy_base")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "dummy_version")

        return {
            "provider": "openai",
            "model": "test-embeddings-3",
            "timeout": 10,
            "num_retries": 5,
        }

    @pytest.mark.parametrize(
        "custom_config, expected_client, api_key",
        (
            (
                {
                    "provider": "huggingface",
                    "model": "test-hf/test-embeddings",
                },
                DefaultLiteLLMEmbeddingClient,
                "HUGGINGFACE_API_KEY",
            ),
            (
                {
                    "provider": "openai",
                    "model": "openai/test-embeddings",
                },
                OpenAIEmbeddingClient,
                "OPENAI_API_KEY",
            ),
            (
                {
                    "provider": "azure",
                    "deployment": "azure/my-test-gpt-deployment-on-azure",
                    "api_base": "https://my-test-base",
                    "api_version": "v1",
                },
                AzureOpenAIEmbeddingClient,
                "AZURE_API_KEY",
            ),
        ),
    )
    def test_correctly_initializes_embedding_clients(
        self,
        custom_config,
        expected_client,
        api_key,
        default_model_configuration,
        monkeypatch,
    ):
        monkeypatch.setenv(api_key, "test")
        client = embedder_factory(custom_config, default_model_configuration)
        assert isinstance(client, expected_client)

    def test_correctly_initializes_router_clients(self, default_model_configuration):
        router_config = {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "huggingface",
                    "model": "test-hf/test-hf-embeddings",
                    "api_key": "test",
                },
                {"provider": "openai", "model": "test-embeddings-3", "api_key": "test"},
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "test-api-base",
                },
            ],
            "router": {"routing_strategy": "test"},
        }
        client = embedder_factory(router_config, default_model_configuration)
        assert isinstance(client, EmbeddingClient)
        assert isinstance(client, RouterClient)
        # Currently, this is one and only implementation of RouterClient, this might
        # change in the future
        assert isinstance(client, LiteLLMRouterEmbeddingClient)

    def test_initializes_embedding_client_when_router_is_not_present(
        self, default_model_configuration, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        # This configuration is expected to be returned when llm config is
        # resolved
        router_config = {
            "id": "valid_id",
            "models": [{"provider": "openai", "model": "test-embeddings-3"}],
            # router: {...} not present
        }
        client = embedder_factory(router_config, default_model_configuration)
        assert isinstance(client, EmbeddingClient)
        assert isinstance(client, OpenAIEmbeddingClient)


class TestEmbedderClientFactory:
    def test_embedder_client_factory(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        embedder = embedder_client_factory(
            None, {"provider": "openai", "model": "test-embedding"}
        )
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
    def test_factory_returns_openai_embedding_client(
        self,
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
        client = embedder_client_factory(config, {"provider": "openai"})

        # Then
        assert isinstance(client, OpenAIEmbeddingClient)
        assert client.model == expected_model
        assert client.api_type == expected_api_type
        assert client.api_base == expected_api_base
        assert client.api_version == expected_api_version

    def test_raises_exception_when_openai_client_setup_is_invalid(
        self,
        monkeypatch: MonkeyPatch,
    ):
        """OpenAI client requires the OPENAI_API_KEY environment variable
        to be set.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderClientValidationError):
            embedder_client_factory(
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
    def test_returns_azure_openai_embedding_client(
        self,
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
        client = embedder_client_factory(config, {"provider": "xyz"})

        # Then
        assert isinstance(client, AzureOpenAIEmbeddingClient)
        assert client.deployment == expected_deployment
        assert client.api_type == expected_api_type
        assert client.api_base == expected_api_base
        assert client.api_version == expected_api_version

    def test_raises_exception_when_azure_openai_client_setup_is_invalid(
        self,
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
                embedder_client_factory(
                    {
                        "provider": "azure",
                        "deployment": "my-test-embedding-deployment-on-azure",
                    },
                    {"provider": "openai"},
                )
            monkeypatch.delenv(env_var, raising=False)

    def test_returns_azure_openai_embedding_client_without_specified_provider_key(
        self,
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
        client = embedder_client_factory(config, {"provider": "openai"})

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
                {
                    "model": "huggingface/microsoft/codebert-base",
                    "provider": "huggingface",
                },
                "HUGGINGFACE_API_KEY",
            ),
        ),
    )
    def test_returns_default_litellm_client(
        self, config: dict, api_key_env: str, monkeypatch: MonkeyPatch
    ):
        # Given
        # Client cannot be instantiated without the required environment variable
        monkeypatch.setenv(api_key_env, "test")
        # When
        client = embedder_client_factory(config, {"provider": "openai"})
        # Then
        assert isinstance(client, DefaultLiteLLMEmbeddingClient)

    def test_raises_exception_when_default_client_setup_is_invalid(self):
        # Given
        # config not containing `model` key
        config = {"some_random_key": "cohere/command"}
        # When / Then
        with pytest.raises(ValueError):
            embedder_client_factory(config, {"provider": "openai"})

    def test_uses_custom_provider(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")

        embedder = embedder_client_factory(
            {"provider": "openai", "model": "test-embedding"},
            {"provider": "foobar", "model": "foo"},
        )
        assert isinstance(embedder, OpenAIEmbeddingClient)

    @pytest.mark.parametrize(
        "config,expected_model",
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
    def test_returns_huggingface_local_embedding_client(
        self,
        config: dict,
        expected_model: str,
        monkeypatch: MonkeyPatch,
    ):
        # When
        with (
            patch(
                "rasa.shared.providers.embedding.huggingface_local_embedding_client"
                ".HuggingFaceLocalEmbeddingClient._init_client"
            ) as mock_init_client,
            patch(
                "rasa.shared.providers.embedding.huggingface_local_embedding_client"
                ".HuggingFaceLocalEmbeddingClient._validate_if_sentence_transformers_installed"
            ) as mock_validate_if_sentence_transformers_installed,
        ):
            mock_init_client.return_value = None
            mock_validate_if_sentence_transformers_installed.return_value = None

            client = embedder_client_factory(config, {"provider": "xyz"})

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
    def test_raises_exception_when_huggingface_local_embedding_client_config_is_invalid(
        self,
        config,
    ):
        # When / Then
        with pytest.raises(ValueError):
            embedder_client_factory(config, {"provider": "xyz"})


class TestEmbedderRouterFactory:
    @pytest.fixture
    def default_model_configuration(self, monkeypatch: MonkeyPatch) -> Dict:
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "dummy_openai")
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "dummy_azure")
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "dummy_base")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "dummy_version")

        return {
            "provider": "openai",
            "model": "test-text-embedding",
            "timeout": 10,
            "num_retries": 5,
        }

    def test_llm_router_factory(self, default_model_configuration: Dict):
        router_config = {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "openai",
                    "model": "some-other-test-embeddings",
                    "api_key": "test",
                },
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "test-api-base",
                },
            ],
            "router": {"routing_strategy": "test"},
        }

        router = embedder_router_factory(router_config, default_model_configuration)
        assert isinstance(router, RouterClient)
        assert isinstance(router, EmbeddingClient)
        # This for current and only implementation of Router. Could change in the
        # future.
        assert isinstance(router, LiteLLMRouterEmbeddingClient)

    def test_router_is_initialized_with_correctly_combined_with_default_parameters(
        self, default_model_configuration
    ):
        router_config = {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "test-api-base",
                },
                {
                    "provider": "openai",
                    "model": "some-other-test-embeddings",
                    "api_key": "test",
                },
                {
                    "provider": "openai",
                    "model": "test-text-embedding",
                    "api_key": "test",
                    "num_retries": 100,
                    "timeout": 100,
                },
            ],
            "router": {"routing_strategy": "test"},
        }
        expected_litellm_model_configurations = [
            # For providers other than specified in default configuration,
            # everything should be as provided.
            {
                "model": "azure/test-deployment",
                "api_key": "test",
                "api_base": "test-api-base",
            },
            # For providers matching the provider specified in default configuration,
            # we expect the default parameters to be present if not provided.
            {
                "model": "openai/some-other-test-embeddings",
                "api_key": "test",
                "timeout": 10,
                "num_retries": 5,
            },
            # For providers matching the provider specified in default configuration,
            # we expect the default parameters not to override the be present
            # parameters.
            {
                "model": "openai/test-text-embedding",
                "api_key": "test",
                "timeout": 100,
                "num_retries": 100,
            },
        ]

        router = embedder_router_factory(router_config, default_model_configuration)
        assert isinstance(router, RouterClient)
        assert isinstance(router, EmbeddingClient)
        # This for current and only implementation of Router. Could change in the
        # future.
        assert isinstance(router, LiteLLMRouterEmbeddingClient)
        actual_litellm_model_configurations = [
            model_configuration["litellm_params"]
            for model_configuration in router.model_configurations
        ]
        assert (
            actual_litellm_model_configurations == expected_litellm_model_configurations
        )

    @pytest.mark.parametrize(
        "router_config",
        [
            # Missing "api_key"
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "azure",
                        "deployment": "test-deployment",
                        # "api_key" missing
                        "api_base": "https://example.azure.com",
                    }
                ],
                "router": {"routing_strategy": "test"},
            },
            # Missing "api_base"
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "azure",
                        "deployment": "test-deployment",
                        "api_key": "test",
                        # "api_base" missing
                    }
                ],
                "router": {"routing_strategy": "test"},
            },
            # Missing "router"
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "azure",
                        "deployment": "test-deployment",
                        "api_key": "test",
                        "api_base": "test-api-base",
                    }
                ],
                # "router": {"routing_strategy": "test"}, missing
            },
        ],
    )
    def test_raises_error_if_configuration_is_invalid(
        self,
        router_config: Dict,
        default_model_configuration: Dict,
        monkeypatch: MonkeyPatch,
    ):
        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, raising=False)
        monkeypatch.delenv(AZURE_API_BASE_ENV_VAR, raising=False)
        monkeypatch.delenv(AZURE_API_VERSION_ENV_VAR, raising=False)

        with pytest.raises((ValueError, ProviderClientValidationError)):
            llm_router_factory(router_config, default_model_configuration)


class TestFactoryCaching:
    def test_llm_cache_factory(self) -> None:
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

    def test_cache_factory_ensures_no_mixup_between_llm_and_embedder_factory(
        self,
    ) -> None:
        with (
            mock.patch(
                "rasa.shared.utils.llm.get_llm_client_from_provider"
            ) as mock_get_llm_client_from_provider,
            mock.patch(
                "rasa.shared.utils.llm.get_embedding_client_from_provider"
            ) as mock_get_embedding_client_from_provider,
        ):
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

    def test_llm_cache_factory_for_config_keys_in_different_order(self) -> None:
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

    def test_embedder_cache_factory(self) -> None:
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

    def test_embedder_cache_factory_for_config_keys_in_different_order(self) -> None:
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

            # Call embedder_factory with the 2nd set of configs
            # (same keys, different order)
            embedder_factory(
                {"model": "test-embedding", "provider": "openai"},
                {"model": "foo", "provider": "foobar"},
            )
            # Cache hit!
            mock_get_embedding_client_from_provider.assert_not_called()

    def test_to_show_that_cache_is_persisted_across_different_calls(self) -> None:
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


def test_get_prompt_template_raises_error_on_file_not_found() -> None:
    """Test that an exception is raised and error is logged when file is not found."""
    default_prompt_template = "default prompt template"

    with patch("rasa.shared.utils.llm.structlogger.error") as mock_error:
        with pytest.raises(InvalidPromptTemplateException) as exc_info:
            get_prompt_template("non_existent_file.jinja2", default_prompt_template)

        # Should raise exception with the file path info
        assert exc_info.value.file_path == "non_existent_file.jinja2"
        assert exc_info.value.resolved_path is not None

        # Should log an error with the file path
        mock_error.assert_called_once()
        call_kwargs = mock_error.call_args[1]
        assert call_kwargs["prompt_file_path"] == "non_existent_file.jinja2"
        assert "resolved_path" in call_kwargs


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


@pytest.mark.parametrize(
    "custom_config,expected_combined_config,",
    (  # Test cases for the client - OpenAI.
        # case: 0
        (
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
            },
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_type": "openai",
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 1
        # Deprecated `provider` aliases
        (
            {
                "_type": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
            },
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_type": "openai",
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 2
        (
            {
                "type": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
            },
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_type": "openai",
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 3
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
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 4
        (
            {"model": "gpt-4"},
            {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_type": "openai",
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 5
        # Missing provider and uses deprecated aliases
        (
            {
                "type": "openai",
                "model_name": "test-gpt",
                "max_tokens": 256,
            },
            {
                "provider": "openai",
                "model": "test-gpt",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_type": "openai",
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 6
        (
            {"api_type": "openai", "model_name": "gpt-4", "max_tokens": 256},
            {
                "provider": "openai",
                "model": "gpt-4",
                "api_type": "openai",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 7
        (
            {
                "api_type": "openai",
                "model_name": "invalid_model",
                "max_tokens": 256,
            },
            {
                "provider": "openai",
                "model": "invalid_model",
                "api_type": "openai",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 8
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
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 9
        (
            {
                "model_name": "openai/gpt-4",
            },
            {
                "provider": "openai",
                "model": "openai/gpt-4",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_type": "openai",
                "api_base": None,
                "api_version": None,
            },
        ),
        # ------------------------------------------------------------------------------
        # Test cases for the client - Azure.
        # case: 10
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
                "model": None,
            },
        ),
        # case: 11
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
                "model": None,
            },
        ),
        # case: 12
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
                "model": None,
            },
        ),
        # case: 13
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
                "model": None,
            },
        ),
        # case: 14
        # Missing provider and uses deprecated aliases
        (
            {
                "engine": "my-test-embedding-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
                "max_tokens": 256,
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "max_completion_tokens": 256,
                "model": None,
            },
        ),
        # case: 15
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
                "api_type": "azure",
                "model": None,
            },
        ),
        # case: 16
        # Deprecated aliases
        (
            {
                "provider": "azure",
                "deployment_name": "my-test-embedding-deployment-on-azure",
                "openai_api_type": "azure",
                "openai_api_base": "https://my-test-base",
                "openai_api_version": "v1",
                "max_tokens": 256,
            },
            {
                "provider": "azure",
                "deployment": "my-test-embedding-deployment-on-azure",
                "api_type": "azure",
                "api_base": "https://my-test-base",
                "api_version": "v1",
                "max_completion_tokens": 256,
                "model": None,
            },
        ),
        # case: 17
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
                "model": None,
            },
        ),
        # case: 18
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
                "model": None,
            },
        ),
        # case: 19
        # litellm way of defining model.
        (
            {
                "deployment": "azure/gpt-4",
            },
            {
                "provider": "azure",
                "deployment": "azure/gpt-4",
                "model": None,
                "api_type": "azure",
                "api_base": None,
                "api_version": None,
            },
        ),
        # case: 20
        (
            {
                "engine": "azure/gpt-4",
            },
            {
                "provider": "azure",
                "deployment": "azure/gpt-4",
                "model": None,
                "api_type": "azure",
                "api_base": None,
                "api_version": None,
            },
        ),
        # ------------------------------------------------------------------------------
        # case: 21
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
        # case: 22
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
        # case: 23
        # Missing provider <=> overriding the default config
        # TODO: Update the codebase so this test fails. For this we can
        #       leverage LiteLLM's utils.get_llm_provider.
        #       This would be the IDEAL behaviour.
        (
            {
                "model": "mistral/mistral-medium",
            },
            {
                "provider": "openai",
                "model": "mistral/mistral-medium",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_type": "openai",
                "api_version": None,
                "api_base": None,
            },
        ),
        # case: 24
        # Missing provider <=> overriding the default config
        # TODO: Update the codebase so this test fails. For this we can
        #       leverage LiteLLM's utils.get_llm_provider.
        #       This would be the IDEAL behaviour.
        (
            {
                "model": "mistral/some-model",
            },
            {
                "provider": "openai",
                "model": "mistral/some-model",
                "temperature": 0.0,
                "max_completion_tokens": 256,
                "timeout": 7,
                "api_type": "openai",
                "api_version": None,
                "api_base": None,
            },
        ),
        # ------------------------------------------------------------------------------
        # case: 25
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
                "api_type": "openai",
                "api_version": None,
                "use_chat_completions_endpoint": True,
            },
        ),
        # case: 26
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
                "api_type": "openai",
                "api_version": None,
                "use_chat_completions_endpoint": True,
            },
        ),
        # case: 27
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
                "api_type": "openai",
                "api_version": None,
                "use_chat_completions_endpoint": True,
            },
        ),
        # case: 28
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
                "use_chat_completions_endpoint": True,
            },
        ),
        # case: 29
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
                "api_type": "openai",
                "api_version": None,
                "use_chat_completions_endpoint": True,
            },
        ),
        # case: 30
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
                "api_type": "openai",
                "api_version": None,
                "use_chat_completions_endpoint": True,
            },
        ),
        # case: 31
        # with use_chat_completions_endpoint set to False
        (
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
                "use_chat_completions_endpoint": False,
            },
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "http://localhost:8000",
                "api_type": "openai",
                "api_version": None,
                "use_chat_completions_endpoint": False,
            },
        ),
    ),
)
def test_combine_custom_and_default_config_combining_single_model_configurations(
    custom_config: Dict[str, Any], expected_combined_config: Dict[str, Any]
) -> None:
    default_config = {
        "provider": "openai",
        "model": "test-gpt",
        "temperature": 0.0,
        "max_completion_tokens": 256,
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
def test_combine_custom_and_default_config_combining_single_model_configurations_throw_error(  # noqa 501
    custom_config: Dict[str, Any],
) -> None:
    default_config = {
        "provider": "openai",
        "model": "test-gpt",
        "temperature": 0.0,
        "max_completion_tokens": 256,
        "timeout": 7,
    }

    with pytest.raises(ValidationError):
        combine_custom_and_default_config(custom_config, default_config)


def test_combine_custom_and_default_config_combining_model_group_configuration() -> (
    None
):
    # Given
    model_configs = [
        {"provider": "openai", "model": "gpt-4"},
        {"provider": "cohere", "model": "test-cohere", "api_key": "test"},
        {
            "provider": "azure",
            "deployment": "test-deployment",
            "api_key": "test",
            "api_base": "test-api-base",
            "api_version": "v1",
        },
    ]
    expected_combined_configs = [
        {
            "provider": "openai",
            "api_type": "openai",
            "api_base": None,  # automatically set by config parser
            "api_version": None,  # automatically set by config parser
            "model": "gpt-4",
            "temperature": 0.0,
            "max_completion_tokens": 256,
            "timeout": 7,
        },
        {"provider": "cohere", "model": "test-cohere", "api_key": "test"},
        {
            "provider": "azure",
            "deployment": "test-deployment",
            "api_key": "test",
            "api_base": "test-api-base",
            "api_type": "azure",  # automatically set by config parser
            "api_version": "v1",
            "model": None,  # automatically set by config parser
        },
    ]
    default_config = {
        "provider": "openai",
        "model": "test-gpt",
        "temperature": 0.0,
        "max_completion_tokens": 256,
        "timeout": 7,
    }
    model_group_config = {"id": "test-model-group", "models": model_configs}
    expected_model_group_config = {
        "id": "test-model-group",
        "models": expected_combined_configs,
    }

    # When
    combined_config = combine_custom_and_default_config(
        model_group_config, default_config
    )

    assert combined_config == expected_model_group_config


def test_resolve_llm_config_with_invalid_model_group_id(
    mock_available_endpoints: MagicMock,
    mock_configuration: MagicMock,
    monkeypatch: MonkeyPatch,
):
    llm_config = {MODEL_GROUP_CONFIG_KEY: "invalid_id"}
    component_name = "test_component"

    mock_available_endpoints.model_groups = [
        {
            "id": "valid_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }
    ]
    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

    with pytest.raises(InvalidConfigException, match="Could not resolve model group"):
        resolve_model_client_config(llm_config, component_name)


def test_resolve_llm_config_with_duplicate_model_groups_defined(
    mock_available_endpoints: MagicMock,
    mock_configuration: MagicMock,
    monkeypatch: MonkeyPatch,
):
    llm_config = {MODEL_GROUP_CONFIG_KEY: "some_id"}
    component_name = "test_component"

    mock_available_endpoints.model_groups = [
        {
            "id": "valid_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        },
        {
            "id": "valid_id",
            "models": [{"provider": "openai", "model": "gpt-3.5"}],
        },
    ]

    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

    with pytest.raises(InvalidConfigException):
        resolve_model_client_config(llm_config, component_name)


def test_resolve_llm_config_with_model_id(
    mock_available_endpoints: MagicMock,
    mock_configuration: MagicMock,
    monkeypatch: MonkeyPatch,
):
    llm_config = {MODEL_GROUP_CONFIG_KEY: "valid_id"}
    component_name = "test_component"

    mock_available_endpoints.model_groups = [
        {
            "id": "valid_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }
    ]
    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

    result = resolve_model_client_config(llm_config, component_name)
    assert result == {
        "id": "valid_id",
        "models": [{"provider": "openai", "model": "gpt-4"}],
    }


def test_resolve_llm_config_with_no_model_groups_defined(
    mock_available_endpoints: MagicMock,
    mock_configuration: MagicMock,
    monkeypatch: MonkeyPatch,
):
    llm_config = {MODEL_GROUP_CONFIG_KEY: "some_id"}
    component_name = "test_component"

    mock_available_endpoints.model_groups = None

    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

    with pytest.raises(
        InvalidConfigException,
        match="No model group with that id found in endpoints.yml",
    ):
        resolve_model_client_config(llm_config, component_name)


@pytest.mark.parametrize(
    "llm_config",
    ({"provider": "openai", "model": "gpt-4"}, None, {}),
)
def test_resolve_llm_config_return_same_config(llm_config: Optional[Dict[str, Any]]):
    component_name = "test_component"

    result = resolve_model_client_config(llm_config, component_name)
    assert result == llm_config


@pytest.fixture
def patch_structlogger():
    with patch("rasa.shared.utils.llm.structlogger.info") as mock:
        yield mock


def test_generate_sender_id():
    test_case_name = "test_case"
    with patch("rasa.shared.utils.llm.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
        sender_id = generate_sender_id(test_case_name)
        assert sender_id == "test_case_2023-01-01 12:00:00"


@pytest.mark.asyncio
async def test_create_tracker_for_user_step():
    step_sender_id = "test_sender_id"
    agent = Agent()
    tracker = DialogueStateTracker.from_events(
        step_sender_id, evts=[UserUttered(f"test {i}") for i in range(5)]
    )
    agent.tracker_store.save(tracker)

    index_user_uttered_event = 3

    await create_tracker_for_user_step(
        step_sender_id, agent, tracker, index_user_uttered_event
    )

    new_tracker = await agent.tracker_store.retrieve(step_sender_id)
    assert new_tracker.sender_id == step_sender_id
    assert len(new_tracker.events) == 3
    assert new_tracker.latest_message.text == f"test {index_user_uttered_event - 1}"


def test_returns_expected_llm_config():
    subclasses = all_subclasses(LLMBasedCommandGenerator)
    yaml_str = f"""
    pipeline:
      - name: {subclasses.pop().__name__}
        llm:
          model_name: "gpt-4"
          temperature: 0.1
    """
    config = read_yaml(yaml_str)
    cfg = _get_llm_command_generator_config(config)

    assert cfg == {"model_name": "gpt-4", "temperature": 0.1}


def test_returns_none_if_no_matching_component():
    yaml_str = """
    pipeline:
      - name: "SomeOtherComponent"
        random_key: "random_value"
    """
    config = read_yaml(yaml_str)
    cfg = _get_llm_command_generator_config(config)

    assert cfg is None


def test_get_system_default_prompts_returns_expected_values():
    root_path = Path(__file__).parent.parent.parent.parent
    default_template_path = root_path / "rasa" / "cli" / "project_templates" / "default"

    config_yaml = (default_template_path / "config.yml").read_text()
    endpoints_yaml = (default_template_path / "endpoints.yml").read_text()

    # Reset configuration
    Configuration._instance = None

    prompts = get_system_default_prompts(
        config=read_yaml(config_yaml), endpoints=read_yaml(endpoints_yaml)
    )

    assert isinstance(prompts, SystemPrompts)

    # Assert Command Generator prompt
    llm_config = resolve_model_client_config(model_config={})
    expected_cmd_prompt = get_default_prompt_template_based_on_model(
        llm_config=llm_config,
        model_prompt_mapping=MODEL_PROMPT_MAPPER,
        default_prompt_path=DEFAULT_COMMAND_PROMPT_TEMPLATE_FILE_NAME,
        fallback_prompt_path=FALLBACK_COMMAND_PROMPT_TEMPLATE_FILE_NAME,
    )
    assert prompts.command_generator == expected_cmd_prompt

    # Assert Enterprise Search prompt
    enterprise_search_prompt_path = (
        root_path
        / "rasa"
        / "core"
        / "policies"
        / "enterprise_search_prompt_template.jinja2"
    )
    enterprise_search_prompt = enterprise_search_prompt_path.read_text()
    assert prompts.enterprise_search == enterprise_search_prompt

    # Assert Response Rephraser prompt
    assert (
        prompts.contextual_response_rephraser
        == DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE
    )


@pytest.mark.parametrize(
    "config, expected_prompt",
    [
        ({}, DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE),
        (
            {
                "policies": [
                    {"name": "EnterpriseSearchPolicy", "citation_enabled": True}
                ]
            },
            DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_CITATION_TEMPLATE,
        ),
        (
            {"policies": [{"name": "EnterpriseSearchPolicy", "check_relevancy": True}]},
            DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_RELEVANCY_CHECK_AND_CITATION_TEMPLATE,
        ),
    ],
)
def test_get_enterprise_search_prompt_returns_correct_template(config, expected_prompt):
    prompt = _get_enterprise_search_prompt(config)
    assert prompt == expected_prompt


@pytest.mark.parametrize(
    "template_content,should_raise",
    [
        # Valid templates
        ("Valid template: {{ user_message }}", False),
        ("{% if condition %}true{% endif %}", False),
        ("{% for item in items %}{{ item }}{% endfor %}", False),
        # Invalid templates
        ("Invalid: {% if condition %}", True),  # Missing endif
        ("Invalid: {{ unclosed_variable", True),  # Missing closing brace
        ("Invalid: {% for item in items %}{{ item }}", True),  # Missing endfor
        # Complex valid template
        (
            """
            {% if user_message %}
                User said: {{ user_message }}
                {% if tracker.slots %}
                    {% for slot_name, slot_value in tracker.slots.items() %}
                        Slot {{ slot_name }}: {{ slot_value }}
                    {% endfor %}
                {% endif %}
            {% else %}
                No message provided
            {% endif %}
            """,
            False,
        ),
        # Complex invalid template
        (
            """
            {% if user_message %}
                User said: {{ user_message }}
                {% if tracker.slots %}
                    {% for slot_name, slot_value in tracker.slots.items() %}
                        Slot {{ slot_name }}: {{ slot_value }}
                    {% endfor %}
                {% endif %}
            {% else %}
                No message provided
            <!-- Missing endif for outer if -->
            """,
            True,
        ),
    ],
)
def test_validate_jinja2_template(template_content: str, should_raise: bool) -> None:
    """Test validate_jinja2_template function with various templates."""
    from rasa.shared.utils.llm import validate_jinja2_template

    if should_raise:
        with pytest.raises(
            Exception
        ):  # Could be jinja2.exceptions.TemplateSyntaxError or other exceptions
            validate_jinja2_template(template_content)
    else:
        # Should not raise any exception
        validate_jinja2_template(template_content)


def test_validate_jinja2_template_with_custom_filter() -> None:
    """Test validate_jinja2_template function with custom Jinja2 filter."""
    from rasa.shared.utils.llm import validate_jinja2_template

    # Template using the custom filter should be valid
    template_with_filter = "{{ user_message | to_json_escaped_string }}"
    validate_jinja2_template(template_with_filter)


def test_validate_jinja2_template_error_details() -> None:
    """Test that validate_jinja2_template provides detailed error information."""
    import jinja2.exceptions

    from rasa.shared.utils.llm import validate_jinja2_template

    invalid_template = """
        Line 1: {{ user_message }}
        Line 2: {% if condition %}
        Line 3:   Some content
        Line 4: {% endif %}
        Line 5: {% if missing_endif %}
        Line 6:   This will cause error
    """.strip()

    with pytest.raises(jinja2.exceptions.TemplateSyntaxError) as exc_info:
        validate_jinja2_template(invalid_template)

    error = exc_info.value
    assert error.lineno == 5
    assert "unexpected end of template" in str(error).lower()
