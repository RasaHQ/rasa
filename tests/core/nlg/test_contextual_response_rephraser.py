from typing import Any, Optional

import pytest
from pytest import MonkeyPatch
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

from rasa.core.nlg.contextual_response_rephraser import (
    ContextualResponseRephraser,
)


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
            }
        }
    )


@pytest.fixture
def greet_tracker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("Hello", {"name": "greet", "confidence": 1.0}),
        ],
    )


@pytest.fixture(autouse=True)
def set_mock_openai_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in rephraser")


class MockedContextualResponseRephraser(ContextualResponseRephraser):
    async def _create_history(self, tracker: DialogueStateTracker) -> str:
        return "User said hello"

    async def _generate_llm_response(self, prompt: str) -> Optional[str]:
        return "hello foobar"


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
) -> None:
    class MockedTemplatedResponseRephraser(ContextualResponseRephraser):
        async def _create_history(self, tracker: DialogueStateTracker) -> str:
            return "User said hello"

        async def _generate_llm_response(self, prompt: str) -> Optional[str]:
            assert prompt == "foobar"
            return "hello foobar"

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
) -> None:
    class MockedTemplatedResponseRephraser(ContextualResponseRephraser):
        async def _create_history(self, tracker: DialogueStateTracker) -> str:
            return "User said hello"

        async def _generate_llm_response(self, prompt: str) -> Optional[str]:
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
            return "hello foobar"

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
