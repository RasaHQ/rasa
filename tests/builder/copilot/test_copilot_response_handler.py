import asyncio
from typing import Any, AsyncGenerator, Dict, List

import pytest
import structlog

from rasa.builder.copilot.copilot_response_handler import (
    KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION,
    LLM_PREFIXES_TO_SUFFIX_REMOVE,
    OUT_OF_SCOPE_PREDICTION,
    PREDICTION_RESPONSES,
    ROLEPLAY_PREDICTION,
    CopilotResponseHandler,
)
from rasa.builder.copilot.models import (
    CopilotOutput,
    GeneratedContent,
    ReferenceSection,
)
from rasa.builder.document_retrieval.models import Document


async def mock_response_stream(content: str) -> AsyncGenerator[str, None]:
    """Mock response stream that yields content in larger chunks to simulate tokens."""
    # Split content into words to simulate token-based streaming
    words = content.split()
    for word in words:
        yield word + " "
        # Small delay to simulate streaming
        await asyncio.sleep(0.01)


class TestCopilotResponseHandler:
    """Test class for CopilotResponseHandler."""

    @pytest.mark.parametrize(
        "input_content,"
        "expected_contains,"
        "check_if_suffix_is_removed,"
        "check_if_prefix_is_removed",
        [
            # Normal response - should pass through unchanged
            (
                "This is a normal response about Rasa development.",
                ["This is a normal response about Rasa development."],
                False,
                False,
            ),
            # Normal response wrapped in code block
            (
                "```This is a normal response about Rasa development.```",
                ["This is a normal response about Rasa development."],
                True,
                True,
            ),
            (
                '"""This is a normal response about Rasa development."""',
                ["This is a normal response about Rasa development."],
                True,
                True,
            ),
            (
                "```markdown\nThis is a normal response about Rasa development.\n```",
                ["This is a normal response about Rasa development."],
                True,
                True,
            ),
            # Roleplay detection
            (
                f"I detected a roleplay request: {ROLEPLAY_PREDICTION}",
                [PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0]],
                False,
                False,
            ),
            # We don't check for suffix or prefix removal because it doesn't matter.
            # The new response is generated from the template.
            (
                f"```{ROLEPLAY_PREDICTION}```",
                [PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0]],
                False,
                False,
            ),
            (
                f"`{ROLEPLAY_PREDICTION}`",
                [PREDICTION_RESPONSES[ROLEPLAY_PREDICTION][0]],
                False,
                False,
            ),
            # Out-of-scope detection
            (
                f"I detected an out-of-scope request: {OUT_OF_SCOPE_PREDICTION}",
                [PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0]],
                False,
                False,
            ),
            # We don't check for suffix or prefix removal because it doesn't matter.
            # The new response is generated from the template.,
            (
                f"I deteted that this is out of scope: {OUT_OF_SCOPE_PREDICTION}",
                [PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0]],
                False,
                False,
            ),
            (
                f"```{OUT_OF_SCOPE_PREDICTION}```",
                [PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0]],
                False,
                False,
            ),
            (
                f"`{OUT_OF_SCOPE_PREDICTION}`",
                [PREDICTION_RESPONSES[OUT_OF_SCOPE_PREDICTION][0]],
                False,
                False,
            ),
            # Knowledge base access requested detection
            (
                f"I detected a knowledge base access requested:"
                f"{KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION}",
                [PREDICTION_RESPONSES[KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION][0]],
                False,
                False,
            ),
            (
                f"```{KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION}```",
                [PREDICTION_RESPONSES[KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION][0]],
                False,
                False,
            ),
            (
                f"`{KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION}`",
                [PREDICTION_RESPONSES[KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION][0]],
                False,
                False,
            ),
            # Empty stream
            (
                "",
                [""],
                False,
                False,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_handle_response(
        self,
        input_content: str,
        expected_contains: List[str],
        check_if_suffix_is_removed: bool,
        check_if_prefix_is_removed: bool,
    ):
        # Given
        handler = CopilotResponseHandler()
        input_stream = mock_response_stream(input_content)

        # When
        responses: List[CopilotOutput] = []
        async for response in handler.handle_response(input_stream):
            responses.append(response)

        # Then
        response_contents: List[str] = []
        for response in responses:
            if isinstance(response, GeneratedContent):
                response_contents.append(response.content)
        response_content = " ".join(response_contents)

        # Check if any of the expected content is in the yielded responses
        assert any(expected in response_content for expected in expected_contains)

        # Check if the suffix and prefix are removed if expected
        for suffix, prefix in LLM_PREFIXES_TO_SUFFIX_REMOVE.items():
            if check_if_suffix_is_removed:
                assert not response_content.endswith(suffix)
            if check_if_prefix_is_removed:
                assert not response_content.startswith(prefix)

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
    def test_extract_references(
        self,
        buffer_content: str,
        documents: List[Document],
        expected_references: List[Dict[str, Any]],
        expected_warnings: List[str],  # type: ignore
    ):
        """Test the extract_references method with various scenarios."""
        # Given
        handler = CopilotResponseHandler()
        handler._llm_stream_buffer = [buffer_content]

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
