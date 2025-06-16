from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import Template
from pytest import MonkeyPatch

from rasa.core.actions.action import ActionBotResponse
from rasa.core.nlg.contextual_response_rephraser import (
    ContextualResponseRephraser,
)
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.engine.language import Language
from rasa.shared.constants import (
    LLM_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.slots import StrictCategoricalSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    KEY_COMPONENT_NAME,
    KEY_LATENCY,
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


@pytest.fixture
def greet_tracker_consecutive_utts_by_same_speaker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events(
        "test",
        evts=[
            BotUttered("Hello"),  # turn 1
            UserUttered("Hi there"),  # turn 2
            UserUttered("Hello, are you still there?"),  # turn 2
            BotUttered("Hi"),  # turn 3
            BotUttered("How can I help you today?"),  # turn 3
            BotUttered("Do you have any questions?"),  # turn 3
            UserUttered("Yes, I want to know my balance."),  # turn 4
            BotUttered("Okay, please let me know your account number"),  # turn 5
            UserUttered("Sure, my account number is 123456"),  # turn 6
            UserUttered("Do you need any more info?"),  # turn 6
            BotUttered(
                "No, that's fine. Please wait so I can retrieve your current balance."
            ),  # turn 7
        ],
    )


@pytest.fixture
def greet_tracker_last_msg_by_user() -> DialogueStateTracker:
    return DialogueStateTracker.from_events(
        "test",
        evts=[
            BotUttered("Hello, I'm a Rasa bot!"),
            UserUttered("Hi there"),
        ],
    )


@pytest.fixture
def greet_tracker_last_msg_by_bot() -> DialogueStateTracker:
    return DialogueStateTracker.from_events(
        "test",
        evts=[
            BotUttered("Hello, I'm a Rasa bot!"),
            UserUttered("Hi there"),
            BotUttered("How can I help you today?"),
        ],
    )


@pytest.fixture
def greet_tracker_no_msg_by_user() -> DialogueStateTracker:
    return DialogueStateTracker.from_events(
        "test",
        evts=[
            BotUttered("Hello, I'm a Rasa bot!"),
            BotUttered("Hi, are you there?"),
        ],
    )


@pytest.fixture(autouse=True)
def set_mock_openai_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in rephraser")


@pytest.fixture
def tracker_with_language(monkeypatch: MonkeyPatch) -> DialogueStateTracker:
    language = Language.from_language_code("es", is_default=True)
    slots = [
        StrictCategoricalSlot(
            name="language",
            mappings=[{}],
            initial_value=language.code,
            values=[language.code],
        )
    ]
    tracker = DialogueStateTracker("default", slots=slots)
    return tracker


@pytest.fixture
def tracker_without_language(
    monkeypatch: MonkeyPatch, patch_default_language: None
) -> DialogueStateTracker:
    return DialogueStateTracker("default", slots=[])


@pytest.fixture
def empty_rephraser() -> ContextualResponseRephraser:
    domain = Domain.empty()
    endpoint_config = EndpointConfig.from_dict({})
    return ContextualResponseRephraser(endpoint_config=endpoint_config, domain=domain)


@pytest.fixture
def english_language() -> Language:
    return Language.from_language_code("en", is_default=True)


@pytest.fixture
def patch_default_language(
    monkeypatch: MonkeyPatch, english_language: Language
) -> None:
    monkeypatch.setattr(
        DialogueStateTracker,
        "default_language",
        property(lambda self: english_language),
    )


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
    patch_default_language: None,
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
    patch_default_language: None,
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
    patch_default_language: None,
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
    patch_default_language: None,
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
                "its meaning. Use simple English.\n\n"
                "Context / previous conversation with the user:\n"
                "User said hello\n\n"
                "Last user message:\n"
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
            "its meaning. Use simple English.\n\n"
            "Context / previous conversation with the user:\n"
            "User said hello\n\n"
            "Last user message:\n"
            "USER: Hello\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
        (
            # explicitly set summarize_history to true
            {
                "summarize_history": True,
            },
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple English.\n\n"
            "Context / previous conversation with the user:\n"
            "User said hello\n\n"
            "Last user message:\n"
            "USER: Hello\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
        (
            # set summarize history to false and max turns to 0
            {
                "summarize_history": False,
                "max_historical_turns": 0,
                "count_multiple_utterances_as_single_turn": False,
            },
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple English.\n\n"
            "Context / previous conversation with the user:\n"
            "USER: Hello\n\n"
            "Last user message:\n"
            "USER: Hello\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
        (
            # set summarize history to false and max turns to 20
            {
                "summarize_history": False,
                "max_historical_turns": 20,
                "count_multiple_utterances_as_single_turn": False,
            },
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple English.\n\n"
            "Context / previous conversation with the user:\n"
            "AI: I'm a Rasa bot!\n"
            "AI: How can I help you today?\n"
            "USER: Hello\n\n"
            "Last user message:\n"
            "USER: Hello\n\n"
            "Suggested "
            "AI Response: Hey there! How can I help you?\n\n"
            "Rephrased AI Response:",
        ),
        (
            # set summarize history to false and max turns to 2
            {
                "summarize_history": False,
                "max_historical_turns": 2,
                "count_multiple_utterances_as_single_turn": False,
            },
            "The following is a conversation with\n"
            "an AI assistant. The assistant is helpful, creative, "
            "clever, and very friendly.\n"
            "Rephrase the suggested AI response staying close "
            "to the original message and retaining\n"
            "its meaning. Use simple English.\n\n"
            "Context / previous conversation with the user:\n"
            "AI: How can I help you today?\n"
            "USER: Hello\n\n"
            "Last user message:\n"
            "USER: Hello\n\n"
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
    patch_default_language: None,
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
            {PROMPT_TEMPLATE_CONFIG_KEY: "data/prompt_templates/test_prompt.jinja2"}
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


async def test_deprecation_warning_with_prompt(
    domain_with_responses: Domain,
) -> None:
    # When
    with patch("rasa.shared.utils.llm.structlogger.warning") as mock_warning:
        ContextualResponseRephraser(
            EndpointConfig.from_dict(
                {PROMPT_CONFIG_KEY: "data/prompt_templates/test_prompt.jinja2"}
            ),
            domain_with_responses,
        )
    mock_warning.assert_any_call(
        "contextual_response_rephraser.init.deprecated_config_key",
        event_info=(
            "The config parameter 'prompt' is deprecated "
            "and will be removed in Rasa 4.0.0. "
            "Please use the config parameter 'prompt_template' instead. "
        ),
    )


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
    patch_default_language: None,
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
    assert KEY_LATENCY in prompts[0][KEY_LLM_RESPONSE_METADATA]
    del prompts[0][KEY_LLM_RESPONSE_METADATA][KEY_LATENCY]
    assert prompts[0][KEY_LLM_RESPONSE_METADATA] == llm_response_dict


@pytest.mark.parametrize(
    "summarize_history, count_multiple_utterances_as_single_turn, max_historical_turns",
    [(True, True, 5), (True, False, 5), (False, True, 5), (False, False, 5)],
)
async def test_rephraser_prompt_conv_history_amended_by_turn_wrapper(
    default_channel,
    greet_tracker_consecutive_utts_by_same_speaker,
    domain: Domain,
    llm_response_object: LLMResponse,
    patch_default_language: None,
    monkeypatch: MonkeyPatch,
    count_multiple_utterances_as_single_turn,
    summarize_history,
    max_historical_turns,
):
    # MockedContextualResponseRephraser to mock LLM response, but not to set history
    class MockedContextualResponseRephraser(ContextualResponseRephraser):
        async def _generate_llm_response(self, prompt: str) -> Optional[LLMResponse]:
            llm_response_object.choices = ["hello foobar"]
            return llm_response_object

    monkeypatch.setattr(
        ContextualResponseRephraser,
        "does_response_allow_rephrasing",
        MagicMock(return_value=True),
    )

    endpoint_config = EndpointConfig.from_dict({})
    rephraser = MockedContextualResponseRephraser(
        endpoint_config=endpoint_config, domain=domain
    )
    rephraser.summarize_history = summarize_history
    rephraser.count_multiple_utterances_as_single_turn = (
        count_multiple_utterances_as_single_turn
    )
    rephraser.max_historical_turns = max_historical_turns

    with set_record_commands_and_prompts():
        events = await ActionBotResponse("utter_default").run(
            default_channel,
            rephraser,
            greet_tracker_consecutive_utts_by_same_speaker,
            domain,
        )
    prompts = events[0].metadata[PROMPTS]

    assert prompts[0][KEY_COMPONENT_NAME] == MockedContextualResponseRephraser.__name__
    assert prompts[0][KEY_PROMPT_NAME] == "rephrase_prompt"
    assert KEY_USER_PROMPT in prompts[0]

    if (
        summarize_history
        and count_multiple_utterances_as_single_turn
        or not summarize_history
        and count_multiple_utterances_as_single_turn
    ):
        assert (
            "Context / previous conversation with the user:\n"
            "AI: Hi How can I help you today? Do you have any questions?\n"
            "USER: Yes, I want to know my balance.\n"
            "AI: Okay, please let me know your account number\n"
            "USER: Sure, my account number is 123456 Do you need any more info?\n"
            "AI: No, that's fine. Please wait so I can retrieve your current balance."
            in prompts[0][KEY_USER_PROMPT]
        )
    elif (
        summarize_history
        and not count_multiple_utterances_as_single_turn
        or not summarize_history
        and not count_multiple_utterances_as_single_turn
    ):
        assert (
            "Context / previous conversation with the user:\n"
            "USER: Yes, I want to know my balance.\n"
            "AI: Okay, please let me know your account number\n"
            "USER: Sure, my account number is 123456\n"
            "USER: Do you need any more info?\n"
            "AI: No, that's fine. Please wait so I can retrieve your current balance."
            in prompts[0][KEY_USER_PROMPT]
        )


@pytest.mark.parametrize(
    "tracker_fixture, expect_user_msg",
    [
        ("greet_tracker_last_msg_by_user", True),
        ("greet_tracker_last_msg_by_bot", True),
        ("greet_tracker_no_msg_by_user", False),
    ],
)
async def test_rephraser_last_user_utt_in_prompt(
    default_channel,
    tracker_fixture,
    domain: Domain,
    patch_default_language: None,
    monkeypatch: MonkeyPatch,
    request,
    expect_user_msg,
):
    tracker = request.getfixturevalue(tracker_fixture)
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
        events = await ActionBotResponse("utter_default").run(
            default_channel, rephraser, tracker, domain
        )
    prompts = events[0].metadata[PROMPTS]

    assert prompts[0][KEY_COMPONENT_NAME] == MockedContextualResponseRephraser.__name__
    assert prompts[0][KEY_PROMPT_NAME] == "rephrase_prompt"
    assert KEY_USER_PROMPT in prompts[0]
    print("**** KEY_USER_PROMPT", prompts[0][KEY_USER_PROMPT])
    if expect_user_msg:
        assert "Last user message:\nUSER: " in prompts[0][KEY_USER_PROMPT]
    else:
        assert "Last user message:\nUSER: " not in prompts[0][KEY_USER_PROMPT]


def test_get_language_label_with_language(
    empty_rephraser: ContextualResponseRephraser,
    tracker_with_language: DialogueStateTracker,
):
    """Language label should be extracted from the tracker."""
    assert empty_rephraser.get_language_label(tracker_with_language) == "Spanish"


def test_get_language_label_without_language(
    empty_rephraser: ContextualResponseRephraser,
    tracker_without_language: DialogueStateTracker,
):
    """Default language label should be used when no language is set."""
    assert empty_rephraser.get_language_label(tracker_without_language) == "English"


def test_prompt_includes_language_label_with_language(
    empty_rephraser: ContextualResponseRephraser,
    tracker_with_language: DialogueStateTracker,
):
    """Prompt should include the language label when language is set."""
    prompt_template_text = empty_rephraser._template_for_response_rephrasing({})
    prompt = Template(prompt_template_text).render(
        language=empty_rephraser.get_language_label(tracker_with_language),
    )
    assert "simple Spanish" in prompt


def test_prompt_includes_language_label_without_language(
    empty_rephraser: ContextualResponseRephraser,
    tracker_without_language: DialogueStateTracker,
):
    """Prompt should include the default language label when no language is set."""
    prompt_template_text = empty_rephraser._template_for_response_rephrasing({})
    prompt = Template(prompt_template_text).render(
        language=empty_rephraser.get_language_label(tracker_without_language),
    )
    assert "simple English" in prompt
