from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, patch

import pytest
from langchain_community.llms import OpenAI
from openai import OpenAIError
from pytest import MonkeyPatch

from rasa.core.nlg.summarize import get_llm_tracing_metadata, summarize_conversation
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.providers.llm.openai_llm_client import OpenAILLMClient
from rasa.shared.utils.constants import (
    LANGFUSE_METADATA_AGENT_ID,
    LANGFUSE_METADATA_COMPONENT_NAME,
    LANGFUSE_METADATA_CUSTOM_METADATA,
    LANGFUSE_METADATA_MODEL_ID,
    LANGFUSE_METADATA_SESSION_ID,
    LANGFUSE_METADATA_TAGS,
)


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


@pytest.mark.parametrize(
    "sender_id, assistant_id, model_id, expected_metadata",
    [
        (
            "user123",
            "assistant456",
            "model789",
            {
                LANGFUSE_METADATA_SESSION_ID: "user123",
                LANGFUSE_METADATA_TAGS: ["ContextualResponseRephraser Summarizer"],
                LANGFUSE_METADATA_CUSTOM_METADATA: {
                    LANGFUSE_METADATA_AGENT_ID: "assistant456",
                    LANGFUSE_METADATA_MODEL_ID: "model789",
                    LANGFUSE_METADATA_COMPONENT_NAME: (
                        "ContextualResponseRephraser Summarizer"
                    ),
                },
            },
        ),
        (
            "user123",
            None,
            None,
            {
                LANGFUSE_METADATA_SESSION_ID: "user123",
                LANGFUSE_METADATA_TAGS: ["ContextualResponseRephraser Summarizer"],
                LANGFUSE_METADATA_CUSTOM_METADATA: {
                    LANGFUSE_METADATA_AGENT_ID: None,
                    LANGFUSE_METADATA_MODEL_ID: None,
                    LANGFUSE_METADATA_COMPONENT_NAME: (
                        "ContextualResponseRephraser Summarizer"
                    ),
                },
            },
        ),
        (
            "user123",
            "assistant456",
            None,
            {
                LANGFUSE_METADATA_SESSION_ID: "user123",
                LANGFUSE_METADATA_TAGS: ["ContextualResponseRephraser Summarizer"],
                LANGFUSE_METADATA_CUSTOM_METADATA: {
                    LANGFUSE_METADATA_AGENT_ID: "assistant456",
                    LANGFUSE_METADATA_MODEL_ID: None,
                    LANGFUSE_METADATA_COMPONENT_NAME: (
                        "ContextualResponseRephraser Summarizer"
                    ),
                },
            },
        ),
    ],
)
def test_get_llm_tracing_metadata(
    sender_id: str,
    assistant_id: Optional[str],
    model_id: Optional[str],
    expected_metadata: Dict[str, Any],
) -> None:
    """Test that get_llm_tracing_metadata returns correct metadata from tracker."""
    tracker = DialogueStateTracker(sender_id=sender_id, slots=[])
    tracker.assistant_id = assistant_id
    tracker.model_id = model_id

    metadata = get_llm_tracing_metadata(tracker)

    assert metadata == expected_metadata


async def test_summarize_conversation_passes_metadata_to_llm() -> None:
    """Test that summarize_conversation passes metadata to LLM via
    get_llm_tracing_metadata."""
    tracker = DialogueStateTracker.from_events(
        "test_sender",
        [
            UserUttered("Hello", {"name": "greet"}),
        ],
    )
    tracker.assistant_id = "test_assistant"
    tracker.model_id = "test_model"

    llm = AsyncMock(spec=OpenAILLMClient)
    llm.acompletion.return_value = AsyncMock(
        spec=LLMResponse, choices=["User says hello."]
    )

    await summarize_conversation(tracker, llm)

    # Verify that acompletion was called with metadata
    assert llm.acompletion.called
    call_args = llm.acompletion.call_args
    assert call_args is not None

    # The metadata should be passed as a keyword argument
    assert "metadata" in call_args.kwargs
    metadata = call_args.kwargs["metadata"]
    assert metadata is not None
    assert metadata[LANGFUSE_METADATA_SESSION_ID] == "test_sender"
    assert metadata[LANGFUSE_METADATA_TAGS] == [
        "ContextualResponseRephraser Summarizer"
    ]
    assert (
        metadata[LANGFUSE_METADATA_CUSTOM_METADATA][LANGFUSE_METADATA_AGENT_ID]
        == "test_assistant"
    )
    assert (
        metadata[LANGFUSE_METADATA_CUSTOM_METADATA][LANGFUSE_METADATA_MODEL_ID]
        == "test_model"
    )
    assert (
        metadata[LANGFUSE_METADATA_CUSTOM_METADATA][LANGFUSE_METADATA_COMPONENT_NAME]
        == "ContextualResponseRephraser Summarizer"
    )
