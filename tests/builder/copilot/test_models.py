from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

from rasa.builder.copilot.models import (
    CopilotRequest,
    GeneratedContent,
    ReferenceEntry,
    ReferenceSection,
    ResponseCategory,
    ResponseCompleteness,
)


@pytest.fixture()
def payload_with_button_and_link() -> Dict[str, Any]:
    return {
        "copilot_chat_history": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Build me a friendly banking assistant...",
                    }
                ],
                "timestamp": "2025-08-22T09:32:52.745Z",
            },
            {
                "role": "copilot",
                "content": [
                    {
                        "type": "text",
                        "text": "Welcome. Here are some next steps...",
                    }
                ],
                "timestamp": "2025-08-22T09:32:52.746Z",
            },
            {
                "role": "copilot",
                "content": [
                    {"type": "text", "text": "Your changes have been saved."},
                    {"type": "button", "label": "Try assistant", "payload": "chat"},
                    {"type": "link", "label": "Docs", "url": "https://rasa.com/docs"},
                ],
                "timestamp": "2025-08-22T09:33:06.973Z",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "how to try assistant"}],
                "timestamp": "2025-08-22T09:33:21.761Z",
            },
        ],
        "session_id": "751b572b-95a9-43ad-bb0e-816ccda73526",
        # Extra keys should be ignored by the request model
        "stream": True,
        "output_type": "markdown",
    }


class TestGeneratedContent:
    """Test cases for GeneratedContent model."""

    @pytest.mark.parametrize(
        "content,response_category,response_completeness",
        [
            (
                "Hello world",
                ResponseCategory.COPILOT,
                ResponseCompleteness.TOKEN,
            ),
            (
                "Out of scope detected",
                ResponseCategory.OUT_OF_SCOPE_DETECTION,
                ResponseCompleteness.COMPLETE,
            ),
            (
                "Roleplay detected",
                ResponseCategory.ROLEPLAY_DETECTION,
                ResponseCompleteness.COMPLETE,
            ),
        ],
    )
    def test_generated_content_creation(
        self,
        content: str,
        response_category: ResponseCategory,
        response_completeness: ResponseCompleteness,
    ):
        """Test that GeneratedContent can be created with different parameters."""
        # When
        generated_content = GeneratedContent(
            content=content,
            response_category=response_category,
            response_completeness=response_completeness,
        )

        # Then
        assert generated_content.content == content
        assert generated_content.response_category == response_category
        assert generated_content.response_completeness == response_completeness

    def test_generated_content_to_sse_event(self):
        """Test that GeneratedContent converts to SSE event correctly."""
        # Given
        content = GeneratedContent(
            content="Hello world",
            response_category=ResponseCategory.COPILOT,
            response_completeness=ResponseCompleteness.TOKEN,
        )

        # When
        sse_event = content.to_sse_event()

        # Then
        assert sse_event.event == "copilot_response"
        assert sse_event.data["content"] == "Hello world"
        assert sse_event.data["response_category"] == ResponseCategory.COPILOT.value
        assert sse_event.data["completeness"] == ResponseCompleteness.TOKEN.value


class TestReferenceEntry:
    @pytest.mark.parametrize(
        "response_category,expected_category,should_raise_error",
        [
            (None, ResponseCategory.REFERENCE_ENTRY, False),
            (ResponseCategory.REFERENCE_ENTRY, ResponseCategory.REFERENCE_ENTRY, False),
            (ResponseCategory.COPILOT, None, True),
        ],
    )
    def test_reference_entry_creation_with_response_category(
        self,
        response_category: Optional[ResponseCategory],
        expected_category: Optional[ResponseCategory],
        should_raise_error: bool,
    ):
        # Given
        kwargs = {
            "index": 1,
            "title": "Test Title",
            "url": "https://example.com",
        }
        if response_category is not None:
            kwargs["response_category"] = response_category

        # When/Then
        if should_raise_error:
            with pytest.raises(ValueError):
                ReferenceEntry(**kwargs)
        else:
            entry = ReferenceEntry(**kwargs)
            assert entry.index == 1
            assert entry.title == "Test Title"
            assert entry.url == "https://example.com"
            assert entry.response_category == expected_category
            assert entry.response_completeness == ResponseCompleteness.COMPLETE

    def test_reference_entry_to_sse_event(self):
        # Given
        entry = ReferenceEntry(index=1, title="Test Title", url="https://example.com")

        # When
        sse_event = entry.to_sse_event()

        # Then
        assert sse_event.event == "copilot_response"
        assert sse_event.data["index"] == 1
        assert sse_event.data["title"] == "Test Title"
        assert sse_event.data["url"] == "https://example.com"
        assert (
            sse_event.data["response_category"]
            == ResponseCategory.REFERENCE_ENTRY.value
        )
        assert sse_event.data["completeness"] == ResponseCompleteness.COMPLETE.value


class TestReferenceSection:
    @pytest.mark.parametrize(
        "response_category,expected_category,should_raise_error",
        [
            (None, ResponseCategory.REFERENCE, False),
            (ResponseCategory.REFERENCE, ResponseCategory.REFERENCE, False),
            (ResponseCategory.COPILOT, None, True),
        ],
    )
    def test_reference_section_creation_with_response_category(
        self,
        response_category: Optional[ResponseCategory],
        expected_category: Optional[ResponseCategory],
        should_raise_error: bool,
    ):
        # Given
        references = [
            ReferenceEntry(index=1, title="Title 1", url="https://example1.com"),
            ReferenceEntry(index=2, title="Title 2", url="https://example2.com"),
        ]

        kwargs = {"references": references}
        if response_category is not None:
            kwargs["response_category"] = response_category

        # When/Then
        if should_raise_error:
            with pytest.raises(ValueError):
                ReferenceSection(**kwargs)
        else:
            section = ReferenceSection(**kwargs)
            assert len(section.references) == len(references)
            assert section.response_category == expected_category
            assert section.response_completeness == ResponseCompleteness.COMPLETE

    def test_reference_section_to_sse_event(self):
        # Given
        references = [
            ReferenceEntry(index=1, title="Title 1", url="https://example1.com"),
            ReferenceEntry(index=2, title="Title 2", url="https://example2.com"),
        ]
        section = ReferenceSection(references=references)

        # When
        sse_event = section.to_sse_event()

        # Then
        assert sse_event.event == "copilot_response"
        assert len(sse_event.data["references"]) == 2
        assert sse_event.data["references"][0]["index"] == 1
        assert sse_event.data["references"][0]["title"] == "Title 1"
        assert sse_event.data["references"][0]["url"] == "https://example1.com"
        assert sse_event.data["references"][1]["index"] == 2
        assert sse_event.data["references"][1]["title"] == "Title 2"
        assert sse_event.data["references"][1]["url"] == "https://example2.com"
        assert sse_event.data["response_category"] == ResponseCategory.REFERENCE.value
        assert sse_event.data["completeness"] == ResponseCompleteness.COMPLETE.value


def test_copilot_request_accepts_button_and_link(
    payload_with_button_and_link: Dict[str, Any],
):
    req = CopilotRequest(**payload_with_button_and_link)
    message = req.copilot_chat_history[2]
    button = message.content[1]
    link = message.content[2]

    # Discriminator should yield the correct types/fields
    assert getattr(button, "type", None) == "button"
    assert getattr(button, "label", None) == "Try assistant"
    assert getattr(button, "payload", None) == "chat"

    assert getattr(link, "type", None) == "link"
    assert getattr(link, "label", None) == "Docs"
    assert getattr(link, "url", None) == "https://rasa.com/docs"


def test_copilot_request_rejects_bad_button_missing_fields(
    payload_with_button_and_link: Dict[str, Any],
):
    bad = payload_with_button_and_link
    # Remove label from the button
    bad["copilot_chat_history"][2]["content"][1] = {"type": "button", "payload": "chat"}
    with pytest.raises(ValidationError):
        CopilotRequest(**bad)


class TestCopilotRequest:
    """Test cases for CopilotRequest model initialization from dict/JSON data."""

    @pytest.mark.parametrize(
        "chat_history",
        [
            # User message
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello, how can I create a custom action?",
                        }
                    ],
                    "response_category": None,
                }
            ],
            # Copilot message
            [
                {
                    "role": "copilot",
                    "content": [
                        {
                            "type": "text",
                            "text": "To create a custom action, you need to...",
                        }
                    ],
                    "response_category": "copilot",
                }
            ],
            # Internal copilot request message
            [
                {
                    "role": "internal_copilot_request",
                    "content": [{"type": "text", "text": "Internal system message"}],
                    "response_category": "training_error_log_analysis",
                }
            ],
            # Mixed message types
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Help me with training errors"}
                    ],
                    "response_category": None,
                },
                {
                    "role": "copilot",
                    "content": [{"type": "text", "text": "I can help you with that"}],
                    "response_category": "copilot",
                },
                {
                    "role": "internal_copilot_request",
                    "content": [{"type": "text", "text": "Analyzing training logs..."}],
                    "response_category": "e2e_testing_error_log_analysis",
                },
            ],
        ],
    )
    def test_copilot_request_from_dict_parses_correctly(
        self, chat_history: List[Dict[str, Any]]
    ):
        """Test that CopilotRequest can be initialized from dict/JSON data."""
        # Test that CopilotRequest can be created from dict data
        request_data: Dict[str, Any] = {
            "copilot_chat_history": chat_history,
            "session_id": "test-session",
        }
        request = CopilotRequest(**request_data)

        # Verify chat history was parsed correctly
        assert len(request.copilot_chat_history) == len(chat_history)

        # Check that each message was parsed into the correct type based on role
        for i, original_message in enumerate(chat_history):
            parsed_message = request.copilot_chat_history[i]
            expected_role = original_message["role"]

            # Verify the role matches
            assert parsed_message.role == expected_role

            # Verify the message was parsed into the correct type
            if expected_role == "user":
                from rasa.builder.copilot.models import UserChatMessage

                assert isinstance(parsed_message, UserChatMessage)
                assert isinstance(parsed_message.get_flattened_text_content(), str)
            elif expected_role == "copilot":
                from rasa.builder.copilot.models import CopilotChatMessage

                assert isinstance(parsed_message, CopilotChatMessage)
                assert isinstance(parsed_message.get_flattened_text_content(), str)
            elif expected_role == "internal_copilot_request":
                from rasa.builder.copilot.models import (
                    InternalCopilotRequestChatMessage,
                )

                assert isinstance(parsed_message, InternalCopilotRequestChatMessage)
