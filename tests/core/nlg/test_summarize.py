from typing import Any
from unittest.mock import patch, AsyncMock

from langchain_community.llms import OpenAI
from openai import OpenAIError
from pytest import MonkeyPatch

from rasa.core.nlg.summarize import summarize_conversation
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.providers.llm.openai_llm_client import OpenAILLMClient


def mocked_openai_complete_response(text: str) -> Any:
    return type(
        "obj",
        (object,),
        {
            "choices": [
                type(
                    "obj",
                    (object,),
                    {"text": text},
                )
            ]
        },
    )


async def test_summarize_conversation_handles_openai_exception(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("Hello", {"name": "greet"}),
        ],
    )
    # use patch to mock openai.Completion.create to raise an exception
    with patch("openai.Completion.create") as mock_completion:
        error = OpenAIError("test")  # type: ignore[no-untyped-call]
        mock_completion.side_effect = error

        # should fallback to transcript
        llm = OpenAI()
        assert await summarize_conversation(tracker, llm) == "USER: Hello"


async def test_summarize_conversation_handles_openai_response() -> None:
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("Hello", {"name": "greet"}),
        ],
    )
    llm = AsyncMock(spec=OpenAILLMClient)
    llm.acompletion.return_value = AsyncMock(
        spec=LLMResponse, choices=["User says hello."]
    )
    # should use response from openai
    assert await summarize_conversation(tracker, llm) == "User says hello."


async def test_summarize_conversation_strips_whitespace() -> None:
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("Hello", {"name": "greet"}),
        ],
    )
    llm = AsyncMock(spec=OpenAILLMClient)
    llm.acompletion.return_value = AsyncMock(
        spec=LLMResponse, choices=["     User says hello. "]
    )

    # should strip whitespace from response
    assert await summarize_conversation(tracker, llm) == "User says hello."
