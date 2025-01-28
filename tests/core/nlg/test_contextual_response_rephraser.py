from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from rasa.core.actions.action import ActionBotResponse
from rasa.core.nlg.contextual_response_rephraser import (
    ContextualResponseRephraser,
)
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.shared.constants import (
    LLM_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    KEY_COMPONENT_NAME,
    KEY_LLM_RESPONSE_METADATA,
    KEY_PROMPT_NAME,
    KEY_USER_PROMPT,
    PROMPTS,
)
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMUsage
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def domain_with_responses() -> Domain:
    return Domain.from_dict(
        {
            "responses": {
                "utter_allows_rephrasing": [
                    {
                        "text": "Hey there! How can I help you?",
                        "metadata": {"rephrase": True},
                    }
                ],
                "utter_does_not_allow_rephrasing": [
                    {
                        "text": "Hey there! How can I help you?",
                        "metadata": {"rephrase": False},
                    }
                ],
                "utter_no_metadata": [{"text": "Hey there! How can I help you?"}],
                "utter_with_prompt": [
                    {
                        "text": "Hey there! How can I help you?",
                        "metadata": {"rephrase_prompt": "foobar", "rephrase": True},
                    }
                ],
                "utter_allows_rephrasing_no_summary": [
                    {
                        "text": "Hey there! How can I help you?",
                        "metadata": {
                            "rephrase": True,
                        },
                    },
                ],
                "utter_allows_rephrasing_with_summary": [
                    {
                        "text": "Hey there! How can I help you?",
                        "metadata": {
                            "rephrase": True,
                            "summarize_conversation": True,
                        },
                    },
                ],
            }
        }
    )


@pytest.fixture
def greet_tracker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events(
        "test",
        evts=[
            BotUttered("I'm a Rasa bot!"),
            BotUttered("How can I help you today?"),
            UserUttered("Hello", {"name": "greet", "confidence": 1.0}),
        ],
    )


@pytest.fixture(autouse=True)
def set_mock_openai_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in rephraser")


class MockedContextualResponseRephraser(ContextualResponseRephraser):
    async def _create_history(self, tracker: DialogueStateTracker) -> str:
        return "User said hello"

    async def _generate_llm_response(self, prompt: str) -> Optional[LLMResponse]:
        return LLMResponse(
            id="mock-id",
            created=123456,
            choices=["hello foobar"],
            model="test-model",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=7),
        )


def test_does_allow_rephrasing(monkeypatch: MonkeyPatch) -> None:
    domain = Domain.empty()
    endpoint_config = EndpointConfig.from_dict({})
    rephraser = ContextualResponseRephraser(
        endpoint_config=endpoint_config, domain=domain
    )
    assert rephraser.does_response_allow_rephrasing({"metadata": {"rephrase": True}})


def test_does_not_allow_rephrasing(monkeypatch: MonkeyPatch) -> None:
    domain = Domain.empty()
    endpoint_config = EndpointConfig.from_dict({})
    rephraser = ContextualResponseRephraser(
        endpoint_config=endpoint_config, domain=domain
    )
    assert not rephraser.does_response_allow_rephrasing({})


async def test_rephraser_generates_response(
    monkeypatch: MonkeyPatch,
    greet_tracker: DialogueStateTracker,
    domain_with_responses: Domain,
) -> None:
    endpoint_config = EndpointConfig.from_dict({})
    rephraser = MockedContextualResponseRephraser(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    generated = await rephraser.generate(
        "utter_allows_rephrasing",
        greet_tracker,
        output_channel="callback",
    )
    assert generated == {"metadata": {"rephrase": True}, "text": "hello foobar"}


async def test_rephraser_does_not_rephrase(
    monkeypatch: MonkeyPatch,
    greet_tracker: DialogueStateTracker,
    domain_with_responses: Domain,
) -> None:
    endpoint_config = EndpointConfig.from_dict({})
    rephraser = MockedContextualResponseRephraser(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    generated = await rephraser.generate(
        "utter_does_not_allow_rephrasing",
        greet_tracker,
        output_channel="callback",
    )
    assert generated == {
        "metadata": {"rephrase": False},
        "text": "Hey there! How can I help you?",
    }


async def test_rephraser_handles_failure_in_generation(
    monkeypatch: MonkeyPatch,
    greet_tracker: DialogueStateTracker,
    domain_with_responses: Domain,
) -> None:
    async def none_no_op(x: Any) -> None:
        return None

    endpoint_config = EndpointConfig.from_dict({})
    rephraser = MockedContextualResponseRephraser(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    monkeypatch.setattr(rephraser, "_generate_llm_response", none_no_op)

    generated = await rephraser.generate(
        "utter_allows_rephrasing",
        greet_tracker,
        output_channel="callback",
    )
    assert generated == {
        "metadata": {"rephrase": True},
        "text": "Hey there! How can I help you?",
    }


async def test_rephraser_uses_template_from_response(
    monkeypatch: MonkeyPatch,
    greet_tracker: DialogueStateTracker,
    domain_with_responses: Domain,
    llm_response_object: LLMResponse,
) -> None:
    class MockedTemplatedResponseRephraser(ContextualResponseRephraser):
        async def _create_history(self, tracker: DialogueStateTracker) -> str:
            return "User said hello"

        async def _generate_llm_response(self, prompt: str) -> Optional[LLMResponse]:
            llm_response_object.choices = ["hello foobar"]
            return llm_response_object

    endpoint_config = EndpointConfig.from_dict({})
    rephraser = MockedTemplatedResponseRephraser(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    generated = await rephraser.generate(
        "utter_with_prompt",
        greet_tracker,
        output_channel="callback",
    )
    assert generated == {
        "metadata": {"rephrase_prompt": "foobar", "rephrase": True},
        "text": "hello foobar",
    }


async def test_rephraser_default_template(
    monkeypatch: MonkeyPatch,
    greet_tracker: DialogueStateTracker,
    domain_with_responses: Domain,
    llm_response_object: LLMResponse,
) -> None:
    class MockedTemplatedResponseRephraser(ContextualResponseRephraser):
        async def _create_history(self, tracker: DialogueStateTracker) -> str:
            return "User said hello"

        async def _generate_llm_response(self, prompt: str) -> Optional[LLMResponse]:
            assert prompt == (
                "The following is a conversation with\n"
                "an AI assistant. The assistant is helpful, creative, "
                "clever, and very friendly.\n"
                "Rephrase the suggested AI response staying close "
                "to the original message and retaining\n"
                "its meaning. Use simple english.\n\n"
                "Context / previous conversation with the user:\n"
                "User said hello\n\n"
                "USER: Hello\n\n"
                "Suggested "
                "AI Response: Hey there! How can I help you?\n\n"
                "Rephrased AI Response:"
            )
            llm_response_object.choices = ["hello foobar"]
            return llm_response_object

    endpoint_config = EndpointConfig.from_dict({})
    rephraser = MockedTemplatedResponseRephraser(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    generated = await rephraser.generate(
        "utter_allows_rephrasing",
        greet_tracker,
        output_channel="callback",
    )
    assert generated == {"metadata": {"rephrase": True}, "text": "hello foobar"}


@pytest.mark.parametrize(
    "endpoint_config, expected_prompt",
    [
        (
            # default - summarize history
            {},
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple english.\n\n"
            "Context / previous conversation with the user:\n"
            "User said hello\n\n"
            "USER: Hello\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
        (
            # explicity set summarize_history to true
            {
                "summarize_history": True,
            },
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple english.\n\n"
            "Context / previous conversation with the user:\n"
            "User said hello\n\n"
            "USER: Hello\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
        (
            # set summairze history to false and max turns to 0
            {
                "summarize_history": False,
                "max_historical_turns": 0,
            },
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple english.\n\n"
            "Context / previous conversation with the user:\n"
            "USER: Hello\n\n\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
        (
            # set summairze history to false and max turns to 20
            {
                "summarize_history": False,
                "max_historical_turns": 20,
            },
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple english.\n\n"
            "Context / previous conversation with the user:\n"
            "AI: I'm a Rasa bot!\n"
            "AI: How can I help you today?\n"
            "USER: Hello\n\n\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
        (
            # set summairze history to false and max turns to 2
            {
                "summarize_history": False,
                "max_historical_turns": 2,
            },
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple english.\n\n"
            "Context / previous conversation with the user:\n"
            "AI: How can I help you today?\n"
            "USER: Hello\n\n\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
    ],
)
async def test_rephraser_template_summarisation(
    monkeypatch: MonkeyPatch,
    greet_tracker: DialogueStateTracker,
    domain_with_responses: Domain,
    endpoint_config: Dict[str, Any],
    expected_prompt: str,
    llm_response_object: LLMResponse,
) -> None:
    class MockedTemplatedResponseRephraser(ContextualResponseRephraser):
        async def _create_history(self, tracker: DialogueStateTracker) -> str:
            return "User said hello"

        async def _generate_llm_response(self, prompt: str) -> Optional[LLMResponse]:
            assert prompt == expected_prompt
            llm_response_object.choices = ["hello foobar"]
            return llm_response_object

    endpoint_config = EndpointConfig.from_dict(endpoint_config)
    rephraser = MockedTemplatedResponseRephraser(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    generated = await rephraser.generate(
        "utter_allows_rephrasing",
        greet_tracker,
        output_channel="callback",
    )
    assert generated == {"metadata": {"rephrase": True}, "text": "hello foobar"}


async def test_contextual_response_rephraser_prompt_init_custom(
    domain_with_responses: Domain,
) -> None:
    rephraser = ContextualResponseRephraser(
        EndpointConfig.from_dict(
            {"prompt": "data/prompt_templates/test_prompt.jinja2"}
        ),
        domain_with_responses,
    )
    assert rephraser.prompt_template.startswith("Identify the user's message")


async def test_contextual_response_rephraser_prompt_init_default(
    domain_with_responses: Domain,
) -> None:
    rephraser = ContextualResponseRephraser(
        EndpointConfig.from_dict({}), domain_with_responses
    )
    assert rephraser.prompt_template.startswith("The following is a conversation")


@pytest.mark.parametrize(
    "config, expected_llm_config",
    [
        (
            {
                LLM_CONFIG_KEY: {"provider": "openai", "model": "gpt-4"},
            },
            {"provider": "openai", "model": "gpt-4"},
        ),
        (
            {
                LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
            },
            {
                "id": "openai_gpt-4",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            },
        ),
        (
            {},
            None,
        ),
    ],
)
def test_contextual_response_rephraser_init_with_different_llm_configs(
    config: Dict[str, Any],
    expected_llm_config: Optional[Dict[str, Any]],
    monkeypatch,
) -> None:
    class MockAvailableEndpoints:
        @staticmethod
        def get_instance():
            return MockAvailableEndpoints()

        def __init__(self):
            self.model_groups = [
                {
                    "id": "openai_gpt-4",
                    "models": [{"provider": "openai", "model": "gpt-4"}],
                },
            ]

    mock_endpoints = MockAvailableEndpoints()
    monkeypatch.setattr("rasa.shared.utils.llm.AvailableEndpoints", mock_endpoints)

    rephraser = ContextualResponseRephraser(
        EndpointConfig.from_dict(config),
        Domain.empty(),
    )

    assert rephraser.llm_config == expected_llm_config


def test_add_prompt_and_llm_metadata_to_response_with_llm_response(
    llm_response_object: LLMResponse,
):
    response = {}
    prompt_name = "test_prompt"
    user_prompt = "What is the weather like?"
    with set_record_commands_and_prompts():
        result = ContextualResponseRephraser._add_prompt_and_llm_metadata_to_response(
            response, prompt_name, user_prompt, llm_response_object
        )
        assert result[PROMPTS] == [
            {
                KEY_COMPONENT_NAME: ContextualResponseRephraser.__name__,
                KEY_PROMPT_NAME: prompt_name,
                KEY_USER_PROMPT: user_prompt,
                KEY_LLM_RESPONSE_METADATA: llm_response_object.to_dict(),
            },
        ]


def test_add_prompt_and_llm_metadata_to_response_without_llm_response():
    response = {}
    prompt_name = "test_prompt"
    user_prompt = "What is the weather like?"

    with set_record_commands_and_prompts():
        result = ContextualResponseRephraser._add_prompt_and_llm_metadata_to_response(
            response, prompt_name, user_prompt
        )

    assert result["prompts"] == [
        {
            KEY_COMPONENT_NAME: ContextualResponseRephraser.__name__,
            KEY_PROMPT_NAME: prompt_name,
            KEY_USER_PROMPT: user_prompt,
            KEY_LLM_RESPONSE_METADATA: None,
        }
    ]


def test_add_prompt_and_llm_metadata_to_response_existing_prompts():
    response = {
        PROMPTS: [
            {
                KEY_COMPONENT_NAME: ContextualResponseRephraser.__name__,
                KEY_PROMPT_NAME: "existing_prompt",
                KEY_USER_PROMPT: "Existing prompt",
            }
        ]
    }
    prompt_name = "test_prompt"
    user_prompt = "What is the weather like?"
    with set_record_commands_and_prompts():
        result = ContextualResponseRephraser._add_prompt_and_llm_metadata_to_response(
            response, prompt_name, user_prompt
        )

    assert result["prompts"] == [
        {
            KEY_COMPONENT_NAME: ContextualResponseRephraser.__name__,
            KEY_PROMPT_NAME: "existing_prompt",
            KEY_USER_PROMPT: "Existing prompt",
        },
        {
            KEY_COMPONENT_NAME: ContextualResponseRephraser.__name__,
            KEY_PROMPT_NAME: prompt_name,
            KEY_USER_PROMPT: user_prompt,
            KEY_LLM_RESPONSE_METADATA: None,
        },
    ]


async def test_rephraser_prompt_is_stored_in_the_tracker(
    default_channel,
    default_nlg,
    default_tracker,
    domain: Domain,
    llm_response_dict: Dict[str, Any],
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setattr(
        ContextualResponseRephraser,
        "does_response_allow_rephrasing",
        MagicMock(return_value=True),
    )

    endpoint_config = EndpointConfig.from_dict({})
    rephraser = MockedContextualResponseRephraser(
        endpoint_config=endpoint_config, domain=domain
    )
    with set_record_commands_and_prompts():
        events = await ActionBotResponse("utter_channel").run(
            default_channel, rephraser, default_tracker, domain
        )

    prompts = events[0].metadata[PROMPTS]

    assert prompts[0][KEY_COMPONENT_NAME] == MockedContextualResponseRephraser.__name__
    assert prompts[0][KEY_PROMPT_NAME] == "rephrase_prompt"
    assert KEY_USER_PROMPT in prompts[0]
    assert KEY_LLM_RESPONSE_METADATA in prompts[0]

    llm_response_dict["choices"] = ["hello foobar"]
    assert prompts[0][KEY_LLM_RESPONSE_METADATA] == llm_response_dict
