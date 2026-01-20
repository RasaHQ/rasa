import asyncio
import re
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import MagicMock

import pytest
import structlog
from agents import RawResponsesStreamEvent, StreamEvent
from openai.types.responses import (
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputText,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from rasa.builder.copilot.models import (
    ControlledPredictionContent,
    CopilotOutput,
    CopilotTextContent,
    CopilotTextEndContent,
    CopilotTextStartContent,
    ExceptionContent,
    GeneratedContent,
    ReferenceSection,
    ResponseCategory,
)
from rasa.builder.copilot.response_handling.agent_copilot_response_handler import (
    AgentCopilotResponseHandler,
)
from rasa.builder.copilot.response_handling.constants import (
    EXCEPTION_RESPONSE,
    KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION,
    LLM_PREFIXES_TO_SUFFIX_REMOVE,
    OUT_OF_SCOPE_PREDICTION,
    PREDICTION_RESPONSES,
    ROLEPLAY_PREDICTION,
)
from rasa.builder.copilot.response_handling.utils import (
    extract_text_content_from_events,
    is_text_content_part_start_event,
)
from rasa.builder.document_retrieval.models import Document


def create_text_delta_event_mock(delta: str) -> StreamEvent:
    """Create a mock ResponseTextDeltaEvent wrapped in RawResponsesStreamEvent."""
    text_delta = MagicMock(spec=ResponseTextDeltaEvent)
    text_delta.delta = delta

    event = MagicMock(spec=RawResponsesStreamEvent)
    event.data = text_delta
    return event


def create_text_content_part_start_event_mock() -> StreamEvent:
    """Create a mock ResponseContentPartAddedEvent for text content part start."""
    text_output = MagicMock(spec=ResponseOutputText)
    part_added = MagicMock(spec=ResponseContentPartAddedEvent)
    part_added.part = text_output

    event = MagicMock(spec=RawResponsesStreamEvent)
    event.data = part_added
    return event


def create_text_output_done_event_mock() -> StreamEvent:
    """Create a mock ResponseTextDoneEvent for text output done."""
    text_done = MagicMock(spec=ResponseTextDoneEvent)

    event = MagicMock(spec=RawResponsesStreamEvent)
    event.data = text_done
    return event


def create_text_content_part_end_event_mock() -> StreamEvent:
    """Create a mock ResponseContentPartDoneEvent for text content part end."""
    text_output = MagicMock(spec=ResponseOutputText)
    part_done = MagicMock(spec=ResponseContentPartDoneEvent)
    part_done.part = text_output

    event = MagicMock(spec=RawResponsesStreamEvent)
    event.data = part_done
    return event


async def mock_response_stream(content: str) -> AsyncGenerator[StreamEvent, None]:
    """Mock response stream that yields StreamEvent objects with text content."""
    # First yield the start event
    yield create_text_content_part_start_event_mock()

    # Split content into words to simulate token-based streaming
    words = content.split()
    for word in words:
        yield create_text_delta_event_mock(word + " ")
        # Small delay to simulate streaming
        await asyncio.sleep(0.01)

    # Yield the text output done event before the content part end
    yield create_text_output_done_event_mock()

    # Yield the end event
    yield create_text_content_part_end_event_mock()


async def create_stream_from_events(
    events: List[StreamEvent],
) -> AsyncGenerator[StreamEvent, None]:
    """Create a stream from a list of stream events."""
    for event in events:
        yield event


class TestAgentCopilotResponseHandler:
    """Test class for AgentCopilotResponseHandler.

    Note: These tests are specifically for CopilotResponseHandler functionality
    from agent_copilot_response_handler.py. Tests use CopilotResponseHandler directly
    when accessing internal state like _llm_stream_buffer. For tests that use the
    public API, CopilotResponseHandler can be used (which may be either implementation
    based on config).
    """

    @pytest.mark.parametrize(
        "input_content,"
        "expected_contains,"
        "check_if_suffix_is_removed,"
        "check_if_prefix_is_removed",
        [
            # Normal response - should pass through unchanged
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("This "),
                    create_text_delta_event_mock("is "),
                    create_text_delta_event_mock("a "),
                    create_text_delta_event_mock("normal "),
                    create_text_delta_event_mock("response "),
                    create_text_delta_event_mock("about "),
                    create_text_delta_event_mock("Rasa "),
                    create_text_delta_event_mock("development."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                ["This is a normal response about Rasa development."],
                False,
                False,
            ),
            # Normal response wrapped in code block
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock("This "),
                    create_text_delta_event_mock("is "),
                    create_text_delta_event_mock("a "),
                    create_text_delta_event_mock("normal "),
                    create_text_delta_event_mock("response "),
                    create_text_delta_event_mock("about "),
                    create_text_delta_event_mock("Rasa "),
                    create_text_delta_event_mock("development."),
                    create_text_delta_event_mock("```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                ["This is a normal response about Rasa development."],
                True,
                True,
            ),
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock('"""'),
                    create_text_delta_event_mock("This "),
                    create_text_delta_event_mock("is "),
                    create_text_delta_event_mock("a "),
                    create_text_delta_event_mock("normal "),
                    create_text_delta_event_mock("response "),
                    create_text_delta_event_mock("about "),
                    create_text_delta_event_mock("Rasa "),
                    create_text_delta_event_mock("development."),
                    create_text_delta_event_mock('"""'),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                ["This is a normal response about Rasa development."],
                True,
                True,
            ),
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```markdown\n"),
                    create_text_delta_event_mock("This "),
                    create_text_delta_event_mock("is "),
                    create_text_delta_event_mock("a "),
                    create_text_delta_event_mock("normal "),
                    create_text_delta_event_mock("response "),
                    create_text_delta_event_mock("about "),
                    create_text_delta_event_mock("Rasa "),
                    create_text_delta_event_mock("development.\n"),
                    create_text_delta_event_mock("```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                ["This is a normal response about Rasa development."],
                True,
                True,
            ),
            # Roleplay detection
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("I "),
                    create_text_delta_event_mock("detected "),
                    create_text_delta_event_mock("a "),
                    create_text_delta_event_mock("roleplay "),
                    create_text_delta_event_mock("request: "),
                    create_text_delta_event_mock(ROLEPLAY_PREDICTION),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0]],
                False,
                False,
            ),
            # We don't check for suffix or prefix removal because it doesn't matter.
            # The new response is generated from the template.
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock(ROLEPLAY_PREDICTION),
                    create_text_delta_event_mock("```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0]],
                False,
                False,
            ),
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("`"),
                    create_text_delta_event_mock(ROLEPLAY_PREDICTION),
                    create_text_delta_event_mock("`"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0]],
                False,
                False,
            ),
            # Out-of-scope detection
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("I "),
                    create_text_delta_event_mock("detected "),
                    create_text_delta_event_mock("an "),
                    create_text_delta_event_mock("out-of-scope "),
                    create_text_delta_event_mock("request: "),
                    create_text_delta_event_mock(OUT_OF_SCOPE_PREDICTION),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0]],
                False,
                False,
            ),
            # We don't check for suffix or prefix removal because it doesn't matter.
            # The new response is generated from the template.
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("I "),
                    create_text_delta_event_mock("deteted "),
                    create_text_delta_event_mock("that "),
                    create_text_delta_event_mock("this "),
                    create_text_delta_event_mock("is "),
                    create_text_delta_event_mock("out "),
                    create_text_delta_event_mock("of "),
                    create_text_delta_event_mock("scope: "),
                    create_text_delta_event_mock(OUT_OF_SCOPE_PREDICTION),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0]],
                False,
                False,
            ),
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock(OUT_OF_SCOPE_PREDICTION),
                    create_text_delta_event_mock("```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0]],
                False,
                False,
            ),
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("`"),
                    create_text_delta_event_mock(OUT_OF_SCOPE_PREDICTION),
                    create_text_delta_event_mock("`"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0]],
                False,
                False,
            ),
            # Knowledge base access requested detection
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("I "),
                    create_text_delta_event_mock("detected "),
                    create_text_delta_event_mock("a "),
                    create_text_delta_event_mock("knowledge "),
                    create_text_delta_event_mock("base "),
                    create_text_delta_event_mock("access "),
                    create_text_delta_event_mock("requested:"),
                    create_text_delta_event_mock(
                        KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION
                    ),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION][0]],
                False,
                False,
            ),
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock(
                        KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION
                    ),
                    create_text_delta_event_mock("```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION][0]],
                False,
                False,
            ),
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("`"),
                    create_text_delta_event_mock(
                        KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION
                    ),
                    create_text_delta_event_mock("`"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION][0]],
                False,
                False,
            ),
            # Controlled prediction detected in normal flow
            # (not CopilotStreamEndedEarly)
            # This requires enough tokens (>20) to avoid the early exception
            (
                [
                    create_text_content_part_start_event_mock(),
                    # Add enough tokens to exceed
                    # max_expected_special_response_tokens (20)
                    create_text_delta_event_mock("token1 "),
                    create_text_delta_event_mock("token2 "),
                    create_text_delta_event_mock("token3 "),
                    create_text_delta_event_mock("token4 "),
                    create_text_delta_event_mock("token5 "),
                    create_text_delta_event_mock("token6 "),
                    create_text_delta_event_mock("token7 "),
                    create_text_delta_event_mock("token8 "),
                    create_text_delta_event_mock("token9 "),
                    create_text_delta_event_mock("token10 "),
                    create_text_delta_event_mock("token11 "),
                    create_text_delta_event_mock("token12 "),
                    create_text_delta_event_mock("token13 "),
                    create_text_delta_event_mock("token14 "),
                    create_text_delta_event_mock("token15 "),
                    create_text_delta_event_mock("token16 "),
                    create_text_delta_event_mock("token17 "),
                    create_text_delta_event_mock("token18 "),
                    create_text_delta_event_mock("token19 "),
                    create_text_delta_event_mock("token20 "),
                    create_text_delta_event_mock("token21 "),
                    # Now add the controlled prediction
                    create_text_delta_event_mock(
                        f"I detected a roleplay request: {ROLEPLAY_PREDICTION}"
                    ),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0]],
                False,
                False,
            ),
            # Empty stream (no delta events)
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [""],
                False,
                False,
            ),
            # Multiple content parts - Normal response split across parts
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("First "),
                    create_text_delta_event_mock("part "),
                    create_text_delta_event_mock("content."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("Second "),
                    create_text_delta_event_mock("part "),
                    create_text_delta_event_mock("content."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                ["First part content.", "Second part content."],
                False,
                False,
            ),
            # Multiple content parts - Normal response wrapped in code block
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock("First "),
                    create_text_delta_event_mock("part"),
                    create_text_delta_event_mock("```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock("Second "),
                    create_text_delta_event_mock("part"),
                    create_text_delta_event_mock("```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                ["First part", "Second part"],
                True,
                True,
            ),
            # Multiple content parts - Roleplay detection in first part
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock(
                        f"I detected a roleplay request: {ROLEPLAY_PREDICTION}"
                    ),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("Second part content."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [
                    PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0],
                    "Second part content.",
                ],
                False,
                False,
            ),
            # Multiple content parts - Roleplay detection in second part
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("First part content."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock(f"```{ROLEPLAY_PREDICTION}```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [
                    "First part content.",
                    PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0],
                ],
                False,
                False,
            ),
            # Multiple content parts - Out-of-scope detection
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("Some initial content."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock(
                        f"I detected an out-of-scope request: "
                        f"{OUT_OF_SCOPE_PREDICTION}"
                    ),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [
                    "Some initial content.",
                    PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0],
                ],
                False,
                False,
            ),
            # Multiple content parts - Knowledge base access requested detection
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("Preliminary content."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("`"),
                    create_text_delta_event_mock(
                        KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION
                    ),
                    create_text_delta_event_mock("`"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [
                    "Preliminary content.",
                    PREDICTION_RESPONSES[KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION][0],
                ],
                False,
                False,
            ),
            # Multiple content parts - Three parts with normal content
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("First "),
                    create_text_delta_event_mock("part."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("Second "),
                    create_text_delta_event_mock("part."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("Third "),
                    create_text_delta_event_mock("part."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                ["First part.", "Second part.", "Third part."],
                False,
                False,
            ),
            # Multiple content parts - Code block spanning multiple parts
            (
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```markdown"),
                    create_text_delta_event_mock("\n"),
                    create_text_delta_event_mock("Take "),
                    create_text_delta_event_mock("a "),
                    create_text_delta_event_mock("look "),
                    create_text_delta_event_mock("at "),
                    create_text_delta_event_mock("this "),
                    create_text_delta_event_mock("yaml "),
                    create_text_delta_event_mock("code: "),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock("yaml "),
                    create_text_delta_event_mock("\n"),
                    create_text_delta_event_mock("flows:"),
                    create_text_delta_event_mock("\n"),
                    create_text_delta_event_mock("- "),
                    create_text_delta_event_mock("action: "),
                    create_text_delta_event_mock("some_action"),
                    create_text_delta_event_mock("\n"),
                    create_text_delta_event_mock("- "),
                    create_text_delta_event_mock("action: "),
                    create_text_delta_event_mock("some_other_action"),
                    create_text_delta_event_mock("\n"),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock("\n"),
                    create_text_delta_event_mock("This "),
                    create_text_delta_event_mock("is "),
                    create_text_delta_event_mock("what "),
                    create_text_delta_event_mock("you "),
                    create_text_delta_event_mock("need "),
                    create_text_delta_event_mock("to "),
                    create_text_delta_event_mock("do"),
                    create_text_delta_event_mock("."),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("```"),
                    create_text_delta_event_mock("Summary: "),
                    create_text_delta_event_mock("This "),
                    create_text_delta_event_mock("is "),
                    create_text_delta_event_mock("what "),
                    create_text_delta_event_mock("you "),
                    create_text_delta_event_mock("need "),
                    create_text_delta_event_mock("to "),
                    create_text_delta_event_mock("do."),
                    create_text_delta_event_mock("```"),
                    create_text_output_done_event_mock(),
                    create_text_content_part_end_event_mock(),
                ],
                [
                    "Take a look at this yaml code: flows: - action: some_action - action: some_other_action This is what you need to do.",  # noqa: E501
                    "Summary: This is what you need to do.",
                ],
                True,
                True,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_stream(
        self,
        input_content: List[StreamEvent],
        expected_contains: List[str],
        check_if_suffix_is_removed: bool,
        check_if_prefix_is_removed: bool,
    ):
        # Given
        input_stream = create_stream_from_events(input_content)
        handler = AgentCopilotResponseHandler(input_stream)
        expected_content_parts = sum(
            1 for event in input_content if is_text_content_part_start_event(event)
        )

        # When
        responses: List[CopilotOutput] = []
        async for response in handler.stream():
            responses.append(response)

        # Then
        response_contents: List[str] = []
        start_count = 0
        end_count = 0

        for response in responses:
            if isinstance(response, CopilotTextStartContent):
                start_count += 1
            elif isinstance(response, CopilotTextEndContent):
                end_count += 1
            elif isinstance(
                response,
                (GeneratedContent, CopilotTextContent, ControlledPredictionContent),
            ):
                response_contents.append(response.content)
        response_content = " ".join(response_contents)

        # Verify that each content part has a start and end
        assert (
            start_count == expected_content_parts
        ), f"Expected {expected_content_parts} content part starts, got {start_count}"
        assert (
            end_count == expected_content_parts
        ), f"Expected {expected_content_parts} content part ends, got {end_count}"
        assert start_count == end_count, (
            f"Expected balanced start/end events, got {start_count} starts "
            f"and {end_count} ends"
        )

        # Check if any of the expected content is in the yielded responses
        assert any(expected in response_content for expected in expected_contains)

        # Check if the suffix and prefix are removed if expected
        for prefix, suffix in LLM_PREFIXES_TO_SUFFIX_REMOVE.items():
            if check_if_suffix_is_removed:
                assert not response_content.endswith(suffix)
            if check_if_prefix_is_removed:
                assert not response_content.startswith(prefix)

    @pytest.mark.asyncio
    async def test_stream_multiple_text_content_parts_structure(self):
        # Given
        async def create_multi_part_stream() -> AsyncGenerator[StreamEvent, None]:
            """Create a stream with multiple text content parts."""
            # First content part
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("First ")
            yield create_text_delta_event_mock("part ")
            yield create_text_delta_event_mock("content.")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

            # Second content part
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("Second ")
            yield create_text_delta_event_mock("part ")
            yield create_text_delta_event_mock("content.")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

            # Third content part
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("Third ")
            yield create_text_delta_event_mock("part ")
            yield create_text_delta_event_mock("content.")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(create_multi_part_stream())

        # When
        responses: List[CopilotOutput] = []
        async for response in handler.stream():
            responses.append(response)

        # Then
        # Extract text content from responses
        text_contents: List[str] = []
        start_count = 0
        end_count = 0

        for response in responses:
            if isinstance(response, CopilotTextStartContent):
                start_count += 1
            elif isinstance(response, CopilotTextEndContent):
                end_count += 1
            elif isinstance(response, CopilotTextContent):
                text_contents.append(response.content)

        # Verify that all content parts are processed independently
        assert (
            start_count == 3
        ), f"Expected 3 text content part starts, got {start_count}"
        assert end_count == 3, f"Expected 3 text content part ends, got {end_count}"

        # Verify all parts' content is present
        full_content = " ".join(text_contents)
        assert "First" in full_content, "Expected first part content to be present"
        assert "Second" in full_content, "Expected second part content to be present"
        assert "Third" in full_content, "Expected third part content to be present"
        assert "part" in full_content, "Expected part content to be present"
        assert "content" in full_content, "Expected content to be present"

        # Verify start and end events are balanced
        assert start_count == end_count, (
            f"Expected balanced start/end events, got {start_count} starts "
            f"and {end_count} ends"
        )

    @pytest.mark.asyncio
    async def test_controlled_predictions_independent_per_content_part(self):
        """Test controlled predictions are evaluated independently per content part."""

        # Given: A stream with two content parts
        # - First part contains a controlled prediction marker (roleplay)
        # - Second part contains only normal content (no markers)
        async def create_stream_with_marker_in_first_part() -> (
            AsyncGenerator[StreamEvent, None]
        ):
            """Create a stream where first part has a marker, second part is normal."""
            # First content part with roleplay detection marker
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock(ROLEPLAY_PREDICTION)
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

            # Second content part with normal content (no markers)
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("This is normal content")
            yield create_text_delta_event_mock(" without any markers.")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(create_stream_with_marker_in_first_part())

        # When: Process the stream
        responses: List[CopilotOutput] = []
        async for response in handler.stream():
            responses.append(response)

        # Then: Verify each content part is evaluated independently
        # Group responses by content part
        content_part_responses: List[List[CopilotOutput]] = []
        current_part: List[CopilotOutput] = []

        for response in responses:
            if isinstance(response, CopilotTextStartContent):
                if current_part:
                    content_part_responses.append(current_part)
                current_part = [response]
            else:
                current_part.append(response)
        if current_part:
            content_part_responses.append(current_part)

        assert (
            len(content_part_responses) == 2
        ), f"Expected 2 content parts, got {len(content_part_responses)}"

        # Extract controlled predictions and text content from each part
        first_part = content_part_responses[0]
        first_controlled_prediction = next(
            (
                response
                for response in first_part
                if isinstance(response, ControlledPredictionContent)
            ),
            None,
        )

        second_part = content_part_responses[1]
        second_controlled_prediction = next(
            (
                response
                for response in second_part
                if isinstance(response, ControlledPredictionContent)
            ),
            None,
        )
        second_text_content = [
            response.content
            for response in second_part
            if isinstance(response, CopilotTextContent)
        ]

        # Verify first part triggers controlled prediction
        assert (
            first_controlled_prediction is not None
        ), "First content part should trigger controlled prediction"
        assert (
            first_controlled_prediction.response_category
            == ResponseCategory.ROLEPLAY_DETECTION
        ), "First part should detect roleplay"

        # Verify second part does NOT trigger controlled prediction
        assert second_controlled_prediction is None, (
            "Second content part should NOT trigger controlled prediction "
            "even though first part had a marker"
        )
        assert (
            len(second_text_content) > 0
        ), "Second content part should yield normal text content"
        assert "normal content" in "".join(
            second_text_content
        ), "Second part should contain the normal content text"

    @pytest.mark.parametrize(
        "buffer_content,documents,expected_references,expected_warnings",
        [
            # Test case 1: Valid numeric references with matching URLs
            (
                "Here are some references: [1](https://docs.rasa.com/guide1)"
                " and [2](https://docs.rasa.com/guide2)",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 2",
                        url="https://docs.rasa.com/guide2",
                        title="Rasa Guide 2",
                        metadata=None,
                    ),
                ],
                [
                    {
                        "index": 1,
                        "title": "Rasa Guide 1",
                        "url": "https://docs.rasa.com/guide1",
                    },
                    {
                        "index": 2,
                        "title": "Rasa Guide 2",
                        "url": "https://docs.rasa.com/guide2",
                    },
                ],
                [],
            ),
            # Test case 2: Non-numeric reference text (should warn)
            (
                "Here is a reference: [invalid](https://docs.rasa.com/guide1)",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                ],
                [],
                ["invalid_reference_number"],
            ),
            # Test case 3: URL not found in documents (should warn and skip)
            (
                "Here is a reference: [1](https://docs.rasa.com/nonexistent)",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                ],
                [],
                ["url_not_found"],
            ),
            # Test case 4: Mixed valid and invalid references
            (
                "References: [1](https://docs.rasa.com/guide1) "
                "[invalid](https://docs.rasa.com/guide2) "
                "[2](https://docs.rasa.com/nonexistent) ",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 2",
                        url="https://docs.rasa.com/guide2",
                        title="Rasa Guide 2",
                        metadata=None,
                    ),
                ],
                [
                    {
                        "index": 1,
                        "title": "Rasa Guide 1",
                        "url": "https://docs.rasa.com/guide1",
                    },
                ],
                ["invalid_reference_number", "url_not_found"],
            ),
            # Test case 5: No references in content
            (
                "This is just regular text without any references.",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                ],
                [],
                [],  # type: ignore
            ),
            # Test case 6: Document without title (should use fallback)
            (
                "Reference: [1](https://docs.rasa.com/guide1)",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title=None,
                        metadata=None,
                    ),
                ],
                [
                    {
                        "index": 1,
                        "title": "Reference 1",
                        "url": "https://docs.rasa.com/guide1",
                    },
                ],
                [],
            ),
            # Test case 7: Duplicate reference text with same URL (should keep first
            # occurrence)
            (
                "First reference: [1](https://docs.rasa.com/guide1) "
                "Second reference: [1](https://docs.rasa.com/guide1) "
                "Third reference: [2](https://docs.rasa.com/guide2)",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 2",
                        url="https://docs.rasa.com/guide2",
                        title="Rasa Guide 2",
                        metadata=None,
                    ),
                ],
                [
                    {
                        "index": 1,
                        "title": "Rasa Guide 1",
                        "url": "https://docs.rasa.com/guide1",
                    },
                    {
                        "index": 2,
                        "title": "Rasa Guide 2",
                        "url": "https://docs.rasa.com/guide2",
                    },
                ],
                [],
            ),
            # Test case 8: Duplicate reference text with different URLs (should warn and
            # keep first)
            (
                "First reference: [1](https://docs.rasa.com/guide1) "
                "Second reference: [1](https://docs.rasa.com/guide2) "
                "Third reference: [2](https://docs.rasa.com/guide3)",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 2",
                        url="https://docs.rasa.com/guide2",
                        title="Rasa Guide 2",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 3",
                        url="https://docs.rasa.com/guide3",
                        title="Rasa Guide 3",
                        metadata=None,
                    ),
                ],
                [
                    {
                        "index": 1,
                        "title": "Rasa Guide 1",
                        "url": "https://docs.rasa.com/guide1",
                    },
                    {
                        "index": 2,
                        "title": "Rasa Guide 3",
                        "url": "https://docs.rasa.com/guide3",
                    },
                ],
                ["duplicate_reference_text"],
            ),
            # Test case 9: References not in document order (should use reference text
            # as index)
            (
                "References: [3](https://docs.rasa.com/guide3) "
                "[1](https://docs.rasa.com/guide1) "
                "[2](https://docs.rasa.com/guide2)",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 2",
                        url="https://docs.rasa.com/guide2",
                        title="Rasa Guide 2",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 3",
                        url="https://docs.rasa.com/guide3",
                        title="Rasa Guide 3",
                        metadata=None,
                    ),
                ],
                [
                    {
                        "index": 1,
                        "title": "Rasa Guide 1",
                        "url": "https://docs.rasa.com/guide1",
                    },
                    {
                        "index": 2,
                        "title": "Rasa Guide 2",
                        "url": "https://docs.rasa.com/guide2",
                    },
                    {
                        "index": 3,
                        "title": "Rasa Guide 3",
                        "url": "https://docs.rasa.com/guide3",
                    },
                ],
                [],
            ),
            # Test case 10: Multiple duplicate references with mixed scenarios
            (
                "Text with [1](https://docs.rasa.com/guide1) "
                "and [1](https://docs.rasa.com/guide2) "  # Duplicate with different URL
                "and [2](https://docs.rasa.com/guide3) "
                "and [2](https://docs.rasa.com/guide3) "  # Duplicate with same URL
                "and [invalid](https://docs.rasa.com/guide4) "  # Invalid reference
                "and [3](https://docs.rasa.com/nonexistent)"  # Non-existent URL
                "and [4](https://docs.rasa.com/guide4) ",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 2",
                        url="https://docs.rasa.com/guide2",
                        title="Rasa Guide 2",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 3",
                        url="https://docs.rasa.com/guide3",
                        title="Rasa Guide 3",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 4",
                        url="https://docs.rasa.com/guide4",
                        title="Rasa Guide 4",
                        metadata=None,
                    ),
                ],
                [
                    {
                        "index": 1,
                        "title": "Rasa Guide 1",
                        "url": "https://docs.rasa.com/guide1",
                    },
                    {
                        "index": 2,
                        "title": "Rasa Guide 3",
                        "url": "https://docs.rasa.com/guide3",
                    },
                    {
                        "index": 4,
                        "title": "Rasa Guide 4",
                        "url": "https://docs.rasa.com/guide4",
                    },
                ],
                [
                    "duplicate_reference_text",
                    "invalid_reference_number",
                    "url_not_found",
                ],
            ),
            # Test case 11: Non-numeric reference text that should warn and be skipped
            (
                "References with letters: [a](https://docs.rasa.com/guide1) "
                "and [b](https://docs.rasa.com/guide2) "
                "and numeric [1](https://docs.rasa.com/guide3)",
                [
                    Document(
                        content="Guide 1",
                        url="https://docs.rasa.com/guide1",
                        title="Rasa Guide 1",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 2",
                        url="https://docs.rasa.com/guide2",
                        title="Rasa Guide 2",
                        metadata=None,
                    ),
                    Document(
                        content="Guide 3",
                        url="https://docs.rasa.com/guide3",
                        title="Rasa Guide 3",
                        metadata=None,
                    ),
                ],
                [
                    {
                        "index": 1,  # Only numeric reference is accepted
                        "title": "Rasa Guide 3",
                        "url": "https://docs.rasa.com/guide3",
                    },
                ],
                [
                    "invalid_reference_number",
                    "invalid_reference_number",
                ],  # Two warnings for [a] and [b]
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_extract_references(
        self,
        buffer_content: str,
        documents: List[Document],
        expected_references: List[Dict[str, Any]],
        expected_warnings: List[str],  # type: ignore
    ):
        """Test the extract_references method with various scenarios.

        Note: Uses CopilotResponseHandler directly as we need to access
        the internal _llm_stream_buffer attribute which is specific to the
        agent implementation.
        """
        # Given
        # Create a mock stream that's already been processed
        handler = AgentCopilotResponseHandler(mock_response_stream(buffer_content))
        # Process the stream to populate the buffer
        # We need to consume the stream first to populate the buffer
        async for _ in handler.stream():
            pass

        # When
        with structlog.testing.capture_logs() as caplog:
            result = handler.extract_references(documents)

        # Then
        assert isinstance(result, ReferenceSection)
        assert len(result.references) == len(expected_references)

        for i, (actual_ref, expected_ref) in enumerate(
            zip(result.references, expected_references)
        ):
            assert actual_ref.index == expected_ref["index"]
            assert actual_ref.title == expected_ref["title"]
            assert actual_ref.url == expected_ref["url"]

        # Verify warnings
        warning_logs = [log for log in caplog if log.get("log_level") == "warning"]
        assert len(warning_logs) == len(expected_warnings)
        for expected_warning in expected_warnings:
            # Check if any warning log contains the expected warning type
            warning_found = False
            for log in warning_logs:
                log_message = str(log)
                if expected_warning in log_message:
                    warning_found = True
                    break
            assert warning_found

    @pytest.mark.asyncio
    async def test_extract_references_multiple_content_parts(self):
        """Test that references are extracted once across multiple content parts."""
        # Given
        documents = [
            Document(
                content="Guide 1",
                url="https://docs.rasa.com/guide1",
                title="Rasa Guide 1",
                metadata=None,
            ),
            Document(
                content="Guide 2",
                url="https://docs.rasa.com/guide2",
                title="Rasa Guide 2",
                metadata=None,
            ),
            Document(
                content="Guide 3",
                url="https://docs.rasa.com/guide3",
                title="Rasa Guide 3",
                metadata=None,
            ),
        ]

        async def create_multi_part_stream_with_references() -> (
            AsyncGenerator[StreamEvent, None]
        ):
            """Create a stream with multiple content parts containing references."""
            # First content part with reference [1]
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("First part with ")
            yield create_text_delta_event_mock("[1](https://docs.rasa.com/guide1)")
            yield create_text_delta_event_mock(".")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

            # Second content part with reference [2]
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("Second part with ")
            yield create_text_delta_event_mock("[2](https://docs.rasa.com/guide2)")
            yield create_text_delta_event_mock(".")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

            # Third content part with reference [3]
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("Third part with ")
            yield create_text_delta_event_mock("[3](https://docs.rasa.com/guide3)")
            yield create_text_delta_event_mock(".")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(
            create_multi_part_stream_with_references()
        )
        # Process the stream to populate the buffer
        async for _ in handler.stream():
            pass

        # When
        with structlog.testing.capture_logs() as caplog:
            result = handler.extract_references(documents)

        # Then
        assert isinstance(result, ReferenceSection)
        assert len(result.references) == 3

        expected_references: List[Dict[str, Any]] = [
            {
                "index": 1,
                "title": "Rasa Guide 1",
                "url": "https://docs.rasa.com/guide1",
            },
            {
                "index": 2,
                "title": "Rasa Guide 2",
                "url": "https://docs.rasa.com/guide2",
            },
            {
                "index": 3,
                "title": "Rasa Guide 3",
                "url": "https://docs.rasa.com/guide3",
            },
        ]

        for i, (actual_ref, expected_ref) in enumerate(
            zip(result.references, expected_references)
        ):
            assert actual_ref.index == expected_ref["index"]
            assert actual_ref.title == expected_ref["title"]
            assert actual_ref.url == expected_ref["url"]

        # Verify no warnings were logged
        warning_logs = [log for log in caplog if log.get("log_level") == "warning"]
        assert len(warning_logs) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content,expected_reference_count,expected_reference",
        [
            # Controlled predictions should return empty references
            (
                ROLEPLAY_PREDICTION,
                0,
                None,
            ),
            (
                OUT_OF_SCOPE_PREDICTION,
                0,
                None,
            ),
            (
                KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION,
                0,
                None,
            ),
            # Normal content with references should return references
            (
                "Here is a reference: [1](https://docs.rasa.com/guide1)",
                1,
                {
                    "index": 1,
                    "title": "Rasa Guide 1",
                    "url": "https://docs.rasa.com/guide1",
                },
            ),
        ],
    )
    async def test_extract_references_controlled_predictions_and_normal_content(
        self,
        content: str,
        expected_reference_count: int,
        expected_reference: Dict[str, Any] | None,
    ):
        # Given
        documents = [
            Document(
                content="Guide 1",
                url="https://docs.rasa.com/guide1",
                title="Rasa Guide 1",
                metadata=None,
            ),
        ]

        async def create_stream() -> AsyncGenerator[StreamEvent, None]:
            # Create a stream with the content
            # Need to include start event, delta with content, done, and end events
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock(content)
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(create_stream())
        # Process the stream to populate generated_responses
        async for _ in handler.stream():
            pass

        # When
        result = handler.extract_references(documents)

        # Then
        assert isinstance(result, ReferenceSection)
        assert len(result.references) == expected_reference_count, (
            f"Expected {expected_reference_count} references, "
            f"but got {len(result.references)} references"
        )

        if expected_reference:
            assert result.references[0].index == expected_reference["index"]
            assert result.references[0].title == expected_reference["title"]
            assert result.references[0].url == expected_reference["url"]

    @pytest.mark.parametrize(
        "exception_type,exception_message,exception_at",
        [
            # Exception during initial stream iteration (before text content part start)
            (
                ValueError,
                "Stream initialization error",
                "initial_iteration",
            ),
            (
                RuntimeError,
                "Connection error during stream",
                "initial_iteration",
            ),
            (
                Exception,
                "Generic stream error",
                "initial_iteration",
            ),
            # Exception during text content part start event processing
            (
                ValueError,
                "Error processing start event",
                "after_start_event",
            ),
            (
                RuntimeError,
                "Error in start event handler",
                "after_start_event",
            ),
            # Exception during buffer exhaustion
            # (in _exhaust_text_content_part_buffer_for_early_detection)
            (
                ValueError,
                "Error during buffer exhaustion",
                "during_buffer_exhaustion",
            ),
            (
                RuntimeError,
                "Buffer processing error",
                "during_buffer_exhaustion",
            ),
            # Exception during _buffer_stream (which propagates up)
            (
                ValueError,
                "Error in buffer stream",
                "during_buffer_stream",
            ),
            (
                RuntimeError,
                "Stream buffer error",
                "during_buffer_stream",
            ),
            # Exception during text delta processing
            (
                ValueError,
                "Error processing text delta",
                "during_text_delta",
            ),
            (
                RuntimeError,
                "Text delta processing error",
                "during_text_delta",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_stream_exception_handling(
        self,
        exception_type: type[Exception],
        exception_message: str,
        exception_at: str,
    ):
        """Test that exceptions during streaming are caught and yield ExceptionContent.

        Args:
            exception_type: The type of exception to raise
            exception_message: The exception message
            exception_at: Where in the stream processing to raise the exception
        """
        # Given
        exception = exception_type(exception_message)

        async def exception_stream() -> AsyncGenerator[StreamEvent, None]:
            """Create a stream that raises an exception at the specified point."""
            if exception_at == "initial_iteration":
                # Raise exception immediately
                raise exception

            # Yield start event first
            yield create_text_content_part_start_event_mock()

            if exception_at == "after_start_event":
                # Raise exception after start event
                raise exception

            # Yield some text delta events
            if exception_at == "during_buffer_exhaustion":
                # Raise during buffer exhaustion (before we get to text deltas)
                # This happens in _exhaust_text_content_part_buffer_for_early_detection
                # which calls _buffer_stream, so we raise during the first
                # buffer stream call
                yield create_text_delta_event_mock("test ")
                # The next call to __anext__ will raise
                raise exception

            if exception_at == "during_buffer_stream":
                # Yield a few events, then raise during buffer stream
                yield create_text_delta_event_mock("test ")
                yield create_text_delta_event_mock("content ")
                # Next call will raise
                raise exception

            if exception_at == "during_text_delta":
                # Yield events normally, then raise during processing
                yield create_text_delta_event_mock("test ")
                yield create_text_delta_event_mock("content ")
                # Raise during next iteration
                raise exception

            # Default: yield end event if we got here
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(exception_stream())

        # When
        responses: List[CopilotOutput] = []
        async for response in handler.stream():
            responses.append(response)

        # Then
        # Should have at least one response (the ExceptionContent)
        assert len(responses) >= 1

        # Find the ExceptionContent response
        exception_responses = [r for r in responses if isinstance(r, ExceptionContent)]
        assert len(exception_responses) == 1

        exception_response = exception_responses[0]
        assert isinstance(exception_response, ExceptionContent)
        assert exception_response.content == EXCEPTION_RESPONSE
        assert exception_response.response_category == ResponseCategory.EXCEPTION
        assert exception_response.original_exception is not None
        assert isinstance(exception_response.original_exception, exception_type)
        assert str(exception_response.original_exception) == exception_message

        # Verify that the exception is tracked in generated_responses
        exception_in_generated = [
            r for r in handler.generated_responses if isinstance(r, ExceptionContent)
        ]
        assert len(exception_in_generated) == 1
        assert handler.extract_response_category() == ResponseCategory.EXCEPTION

    @pytest.mark.asyncio
    async def test_queues_drained_on_exception(self):
        """Test that MCP tool and plan queues are drained when an exception occurs.

        This test verifies the fix for a bug where queues were not drained in the
        exception path, causing tool call events from failed requests to leak into
        subsequent requests.

        Scenario:
        1. Create handler with queues that have items
        2. Simulate an exception during streaming
        3. Verify queues are empty after streaming (drained in finally block)
        """
        from rasa.builder.copilot.models import MCPToolCall, TodoItem, TodoPlanUpdate

        # Given: Create queues with pre-existing items (simulating events added
        # before an exception)
        mcp_queue: asyncio.Queue[MCPToolCall] = asyncio.Queue()
        plan_queue: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()

        # Add items to queues before streaming
        await mcp_queue.put(MCPToolCall(tool_name="test_tool", status="called"))
        await mcp_queue.put(MCPToolCall(tool_name="test_tool", status="completed"))
        plan_queue.put_nowait(
            TodoPlanUpdate(
                tasks=[TodoItem(id="1", content="Test task", status="pending")]
            )
        )

        # Verify queues have items
        assert mcp_queue.qsize() == 2
        assert plan_queue.qsize() == 1

        async def exception_stream() -> AsyncGenerator[StreamEvent, None]:
            """Create a stream that raises an exception."""
            # Yield some events first
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("Some content ")
            # Then raise an exception
            raise RuntimeError("Test exception")

        handler = AgentCopilotResponseHandler(
            exception_stream(),
            mcp_tool_queue=mcp_queue,
            plan_queue=plan_queue,
        )

        # When: Process the stream (which will raise an exception)
        responses: List[CopilotOutput] = []
        async for response in handler.stream():
            responses.append(response)

        # Then: Verify the exception was handled
        exception_responses = [r for r in responses if isinstance(r, ExceptionContent)]
        assert len(exception_responses) == 1

        # Verify queues are empty (drained in finally block)
        assert mcp_queue.empty(), (
            f"MCP queue should be empty after exception, "
            f"but has {mcp_queue.qsize()} items"
        )
        assert plan_queue.empty(), (
            f"Plan queue should be empty after exception, "
            f"but has {plan_queue.qsize()} items"
        )

    @pytest.mark.asyncio
    async def test_queues_drained_on_successful_completion(self):
        """Test that queues are also drained on successful stream completion.

        This ensures the finally block cleanup works for both success and error cases.
        """
        from rasa.builder.copilot.models import MCPToolCall, TodoPlanUpdate

        # Given: Create queues
        mcp_queue: asyncio.Queue[MCPToolCall] = asyncio.Queue()
        plan_queue: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()

        # Add items that won't be consumed during normal streaming
        # (simulating events added after the main loop but before finally)
        # We'll add them during the stream iteration

        async def stream_with_late_queue_items() -> AsyncGenerator[StreamEvent, None]:
            """Stream that adds items to queues during processing."""
            yield create_text_content_part_start_event_mock()
            yield create_text_delta_event_mock("Content ")
            # Normally the hooks would add items here during tool calls
            # For this test, we'll add them manually after yielding
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(
            stream_with_late_queue_items(),
            mcp_tool_queue=mcp_queue,
            plan_queue=plan_queue,
        )

        # When: Process the stream normally
        responses: List[CopilotOutput] = []
        async for response in handler.stream():
            responses.append(response)
            # Add items to queue during iteration (simulating late arrivals)
            if isinstance(response, CopilotTextContent):
                await mcp_queue.put(MCPToolCall(tool_name="late_tool", status="called"))

        # Then: Verify queues are empty after successful completion
        assert mcp_queue.empty(), (
            f"MCP queue should be empty after successful completion, "
            f"but has {mcp_queue.qsize()} items"
        )
        assert plan_queue.empty(), (
            f"Plan queue should be empty after successful completion, "
            f"but has {plan_queue.qsize()} items"
        )

    @pytest.mark.parametrize(
        (
            "max_tokens,rolling_buffer_size,stream_events,"
            "expected_final_buffer_reached,expected_tokens_present,expected_tokens_absent"
        ),
        [
            # Case 1: Stream ends early (before max tokens reached)
            (
                5,
                2,
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("token1 "),
                    create_text_delta_event_mock("token2 "),
                ],
                True,
                ["token1", "token2"],
                [],
            ),
            # Case 2: Method breaks at max tokens
            (
                5,
                2,
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("token1 "),
                    create_text_delta_event_mock("token2 "),
                    create_text_delta_event_mock("token3 "),
                    create_text_delta_event_mock("token4 "),
                    create_text_delta_event_mock("token5 "),
                    create_text_delta_event_mock("token6 "),
                    create_text_content_part_end_event_mock(),
                    create_text_delta_event_mock("token7 "),
                    create_text_delta_event_mock("token8 "),
                    create_text_delta_event_mock("token9 "),
                ],
                False,
                ["token1", "token2", "token3", "token4"],
                ["token5", "token6", "token7", "token8", "token9"],
            ),
            # Case 3: Stream ends at content part end event.
            # exhausted_events includes all yielded events.
            (
                5,
                2,
                [
                    create_text_content_part_start_event_mock(),
                    create_text_delta_event_mock("token1 "),
                    create_text_delta_event_mock("token2 "),
                    create_text_delta_event_mock("token3 "),
                    create_text_content_part_end_event_mock(),
                ],
                True,
                ["token1", "token2", "token3"],
                [],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_exhaust_text_content_part_buffer_for_early_detection(
        self,
        max_tokens: int,
        rolling_buffer_size: int,
        stream_events: List[StreamEvent],
        expected_final_buffer_reached: bool,
        expected_tokens_present: List[str],
        expected_tokens_absent: List[str],
    ):
        """Test _exhaust_text_content_part_buffer_for_early_detection method."""

        # Given
        async def create_test_stream() -> AsyncGenerator[StreamEvent, None]:
            """Create a test stream based on the scenario."""
            # Yield all the stream events (start, deltas, end events, etc.)
            for event in stream_events:
                yield event

        handler = AgentCopilotResponseHandler(
            create_test_stream(), rolling_buffer_size=rolling_buffer_size
        )
        handler._max_expected_special_response_tokens = max_tokens

        # When
        (
            exhausted_events,
            final_buffer_reached,
        ) = await handler._exhaust_text_content_part_buffer_for_early_detection()

        # Then
        # Verify the return values
        assert (
            final_buffer_reached == expected_final_buffer_reached
        ), f"Expected final_buffer_reached={expected_final_buffer_reached}, got {final_buffer_reached}"  # noqa: E501

        # Verify the number of exhausted events is correct
        # Should be <= max_tokens (method stops when limit is reached)
        assert len(exhausted_events) <= max_tokens, (
            f"Expected exhausted_events count <= {max_tokens}, "
            f"got {len(exhausted_events)}"
        )

        # Extract text from exhausted events (not from internal buffers)
        exhausted_text = extract_text_content_from_events(exhausted_events)

        # Verify exhausted events contain expected tokens
        for token in expected_tokens_present:
            assert (
                token in exhausted_text
            ), f"Expected '{token}' to be in exhausted events text: {exhausted_text}"

        # Verify exhausted events do not contain tokens that should be absent
        for token in expected_tokens_absent:
            assert token not in exhausted_text, (
                f"Expected '{token}' NOT to be in exhausted events text: "
                f"{exhausted_text}"
            )

    @pytest.mark.asyncio
    async def test_events_preserved_when_breaking_early_in_early_detection(self):
        """Test that events are preserved when breaking early in early detection.

        This test verifies that when
        _exhaust_text_content_part_buffer_for_early_detection() breaks early
        (after collecting max_expected_special_response_tokens), any remaining
        events in _rolling_buffer are properly handled and not lost.

        Scenario:
        - Small rolling buffer (5) and small max tokens (10) to create a
          scenario where events remain in the buffer after early break
        - First call collects 10 tokens and breaks, leaving some events in
          _rolling_buffer
        - Second call continues processing remaining events
        - All events should be preserved in the final buffer

        This test documents the expected behavior and serves as a regression
        test to ensure events are not lost due to maxlen behavior when
        breaking early.
        """

        # Given: A stream with enough events to trigger early break
        async def create_stream_for_early_break() -> AsyncGenerator[StreamEvent, None]:
            """Create a stream that triggers early break with remaining events."""
            yield create_text_content_part_start_event_mock()
            # Generate enough events to fill buffer multiple times
            for i in range(1, 25):
                yield create_text_delta_event_mock(f"Token{i} ")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(
            create_stream_for_early_break(), rolling_buffer_size=5
        )
        handler._max_expected_special_response_tokens = 10

        # When: Process the stream
        responses: List[CopilotOutput] = []
        async for response in handler.stream():
            responses.append(response)

        # Then: Verify all events are preserved
        all_text = handler.extract_text_from_stream_events()

        # Extract token numbers from the buffer content

        token_matches = re.findall(r"Token(\d+)", all_text)
        token_numbers = sorted(set(int(m) for m in token_matches))

        # Verify all early detection tokens (1-14) are present
        # These tokens are processed during the early detection phase and should
        # all be preserved, even if some remain in _rolling_buffer after break
        expected_early_tokens = list(range(1, 15))
        missing_tokens = [i for i in expected_early_tokens if i not in token_numbers]

        assert not missing_tokens, (
            f"Expected all early detection tokens {expected_early_tokens} to be "
            f"present, but tokens {missing_tokens} are missing. "
            f"Present tokens: {token_numbers}. "
            f"This indicates events may have been lost when breaking early "
            f"without transferring remaining events from _rolling_buffer."
        )

    @pytest.mark.asyncio
    async def test_no_duplication_when_copilot_final_buffer_reached(self):
        """Test that _process_final_content_with_suffix() doesn't duplicate content.

        This test verifies the fix for a bug where _process_final_content_with_suffix()
        was processing ALL content from _llm_stream_buffer, including events that
        were already yielded as initial_content and during streaming. This caused
        duplication for long streams exceeding ~40 events.

        Scenario:
        - Stream with >40 events (enough to fill rolling buffer multiple times)
        - Events go through exhaust phase (added to _llm_stream_buffer)
        - Initial content is yielded (from _process_initial_content_with_prefix())
        - More events are streamed individually (also added to _llm_stream_buffer)
        - Stream ends with CopilotFinalBufferReached
        - Final content should only contain unyielded events from rolling buffer
        """

        # Given: A stream with enough events to trigger the bug
        # We need >40 events with rolling_buffer_size=20 to ensure some events
        # remain in rolling buffer when stream ends
        async def create_long_stream() -> AsyncGenerator[StreamEvent, None]:
            """Create a long stream that triggers CopilotFinalBufferReached."""
            yield create_text_content_part_start_event_mock()
            # Generate enough events to exceed max_expected_special_response_tokens (20)
            # and fill rolling buffer multiple times
            for i in range(1, 50):  # 49 events should be enough
                yield create_text_delta_event_mock(f"token{i} ")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(
            create_long_stream(), rolling_buffer_size=20
        )

        # When: Process the stream and collect all yielded content
        yielded_contents: List[str] = []
        async for response in handler.stream():
            if isinstance(response, CopilotTextContent):
                yielded_contents.append(response.content)

        # Then: Verify no duplication
        # Join all yielded content to get the full text
        all_yielded_text = "".join(yielded_contents)

        token_matches = re.findall(r"token(\d+)", all_yielded_text)
        token_counts: Dict[int, int] = {}
        for token_str in token_matches:
            token_num = int(token_str)
            token_counts[token_num] = token_counts.get(token_num, 0) + 1

        # Find tokens that appear more than once (duplication)
        duplicated_tokens = [
            token for token, count in token_counts.items() if count > 1
        ]

        assert not duplicated_tokens, (
            f"Found duplicated tokens: {duplicated_tokens}. "
            f"This indicates _process_final_content_with_suffix() is processing "
            f"events that were already yielded. "
            f"Token counts: {token_counts}"
        )

        # Verify all tokens from 1-49 are present (no tokens lost)
        expected_tokens = set(range(1, 50))
        present_tokens = set(token_counts.keys())
        missing_tokens = expected_tokens - present_tokens

        assert not missing_tokens, (
            f"Missing tokens: {missing_tokens}. "
            f"All tokens should be present exactly once. "
            f"Present tokens: {sorted(present_tokens)}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_events,rolling_buffer_size,description",
        [
            (5, 20, "less events than buffer"),
            (20, 20, "same number of events as buffer"),
            (50, 20, "more events than buffer"),
        ],
    )
    async def test_extract_text_from_generated_responses_after_streaming(
        self, num_events: int, rolling_buffer_size: int, description: str
    ):
        """Test extract_text_from_generated_responses() returns all content."""

        # Given
        async def create_event_stream() -> AsyncGenerator[StreamEvent, None]:
            yield create_text_content_part_start_event_mock()
            for i in range(num_events):
                yield create_text_delta_event_mock(f"token{i} ")
            yield create_text_output_done_event_mock()
            yield create_text_content_part_end_event_mock()

        handler = AgentCopilotResponseHandler(
            response_stream=create_event_stream(),
            rolling_buffer_size=rolling_buffer_size,
        )

        # When
        async for _ in handler.stream():
            pass
        extracted_text = handler.extract_text_from_generated_responses()

        # Then
        for i in range(num_events):
            assert f"token{i}" in extracted_text, (
                f"Token {i} not found in extracted text for {description}. "
                f"Extracted text: {extracted_text}"
            )
