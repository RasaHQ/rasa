from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotRequest,
    GeneratedContent,
    InternalCopilotRequestChatMessage,
    ReferenceEntry,
    ReferenceSection,
    ResponseCategory,
    ResponseCompleteness,
    UsageStatistics,
    UserChatMessage,
    create_chat_message_from_dict,
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

    def test_parse_chat_history_unknown_role_error(self) -> None:
        """Test that parsing chat history with unknown role raises ValueError."""
        # Given
        chat_history: List[Dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "unknown_role",
                "content": [{"type": "text", "text": "Invalid role"}],
            },
        ]

        # When / Then
        with pytest.raises(ValueError):
            CopilotRequest.parse_chat_history(chat_history)


class TestUsageStatistics:
    """Test class for UsageStatistics cost calculations and methods."""

    @pytest.mark.parametrize(
        "prompt_tokens,cached_prompt_tokens,completion_tokens,"
        "input_token_price,cached_token_price,output_token_price,"
        "expected_non_cached_tokens,"
        "expected_non_cached_cost,expected_cached_cost,"
        "expected_input_cost,expected_output_cost,"
        "expected_total_cost",
        [
            # Basic case with all tokens and prices
            (
                1000,
                200,
                500,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                800,
                0.008,  # non-cached tokens and cost
                0.001,
                0.009,  # cached cost and input cost
                0.01,
                0.019,  # output cost and total cost
            ),
            # No cached tokens (0)
            (
                1000,
                0,
                500,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                1000,
                0.01,  # non-cached tokens and cost
                0.0,
                0.01,  # cached cost and input cost
                0.01,
                0.02,  # output cost and total cost
            ),
            # No cached tokens (None)
            (
                1000,
                None,
                500,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                1000,
                0.01,  # non-cached tokens and cost
                None,
                0.01,  # cached cost and input cost
                0.01,
                0.02,  # output cost and total cost
            ),
            # All cached tokens
            (
                1000,
                1000,
                500,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                0,
                0.0,  # non-cached tokens and cost
                0.005,
                0.005,  # cached cost and input cost
                0.01,
                0.015,  # output cost and total cost
            ),
            # Missing prompt tokens
            (
                None,
                200,
                500,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                None,
                None,  # non-cached tokens and cost
                0.001,
                0.001,  # cached cost and input cost
                0.01,
                0.011,  # output cost and total cost
            ),
            # Missing completion tokens
            (
                1000,
                200,
                None,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                800,
                0.008,  # non-cached tokens and cost
                0.001,
                0.009,  # cached cost and input cost
                None,
                None,  # output cost and total cost
            ),
            # Zero prices
            (
                1000,
                200,
                500,  # tokens
                0.0,
                0.0,
                0.0,  # prices: input, cached, output
                800,
                0.0,  # non-cached tokens and cost
                0.0,
                0.0,  # cached cost and input cost
                0.0,
                0.0,  # output cost and total cost
            ),
            # Cached tokens > prompt tokens (should handle gracefully)
            (
                1000,
                1200,
                500,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                -200,
                -0.002,  # non-cached tokens and cost
                0.006,
                0.004,  # cached cost and input cost
                0.01,
                0.014,  # output cost and total cost
            ),
            # Zero token counts (should return 0.0 costs, not None)
            (
                0,
                0,
                0,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                0,
                0.0,  # non-cached tokens and cost
                0.0,
                0.0,  # cached cost and input cost
                0.0,
                0.0,  # output cost and total cost
            ),
            # Only cached tokens (non_cached_cost=None, cached_cost=0.001)
            (
                None,
                200,
                500,  # tokens
                0.01,
                0.005,
                0.02,  # prices: input, cached, output
                None,
                None,  # non-cached tokens and cost
                0.001,
                0.001,  # cached cost and input cost
                0.01,
                0.011,  # output cost and total cost
            ),
        ],
    )
    def test_usage_statistics_cost_calculations(
        self,
        prompt_tokens: Optional[int],
        cached_prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        input_token_price: float,
        cached_token_price: float,
        output_token_price: float,
        expected_non_cached_tokens: Optional[int],
        expected_non_cached_cost: Optional[float],
        expected_cached_cost: Optional[float],
        expected_input_cost: Optional[float],
        expected_output_cost: Optional[float],
        expected_total_cost: Optional[float],
    ) -> None:
        # Given / When
        usage_stats = UsageStatistics(
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            completion_tokens=completion_tokens,
            input_token_price=input_token_price,
            output_token_price=output_token_price,
            cached_token_price=cached_token_price,
        )

        # Then
        assert usage_stats.non_cached_prompt_tokens == expected_non_cached_tokens
        if expected_non_cached_cost is not None:
            assert usage_stats.non_cached_cost == pytest.approx(
                expected_non_cached_cost
            )
        else:
            assert usage_stats.non_cached_cost == expected_non_cached_cost
        if expected_cached_cost is not None:
            assert usage_stats.cached_cost == pytest.approx(expected_cached_cost)
        else:
            assert usage_stats.cached_cost == expected_cached_cost
        if expected_input_cost is not None:
            assert usage_stats.input_cost == pytest.approx(expected_input_cost)
        else:
            assert usage_stats.input_cost == expected_input_cost
        if expected_output_cost is not None:
            assert usage_stats.output_cost == pytest.approx(expected_output_cost)
        else:
            assert usage_stats.output_cost == expected_output_cost
        if expected_total_cost is not None:
            assert usage_stats.total_cost == pytest.approx(expected_total_cost)
        else:
            assert usage_stats.total_cost == expected_total_cost

    def test_update_token_prices(self):
        """Test updating token prices."""
        # Given / When
        usage_stats = UsageStatistics()

        # When / Then
        usage_stats.update_token_prices(0.01, 0.02, 0.005)

        assert usage_stats.input_token_price == 0.01
        assert usage_stats.output_token_price == 0.02
        assert usage_stats.cached_token_price == 0.005

    def test_reset(self):
        """Test resetting usage statistics."""
        # Given
        usage_stats = UsageStatistics(
            prompt_tokens=1000,
            cached_prompt_tokens=200,
            completion_tokens=500,
            total_tokens=1500,
            model="gpt-4",
        )

        # When
        usage_stats.reset()

        # Then
        assert usage_stats.prompt_tokens is None
        assert usage_stats.cached_prompt_tokens is None
        assert usage_stats.completion_tokens is None
        assert usage_stats.total_tokens is None
        assert usage_stats.model is None

    def test_from_chat_completion_response(self) -> None:
        """Test creating UsageStatistics from ChatCompletion response."""
        from unittest.mock import Mock

        # Test with usage data and cached tokens
        mock_prompt_tokens_details = Mock()
        mock_prompt_tokens_details.cached_tokens = 200

        mock_usage = Mock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 500
        mock_usage.total_tokens = 1500
        mock_usage.prompt_tokens_details = mock_prompt_tokens_details

        mock_response = Mock()
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4"

        usage_stats = UsageStatistics.from_chat_completion_response(
            mock_response, 0.01, 0.02, 0.005
        )

        assert usage_stats is not None
        assert usage_stats.prompt_tokens == 1000
        assert usage_stats.completion_tokens == 500
        assert usage_stats.total_tokens == 1500
        assert usage_stats.cached_prompt_tokens == 200
        assert usage_stats.model == "gpt-4"
        assert usage_stats.input_token_price == 0.01
        assert usage_stats.output_token_price == 0.02
        assert usage_stats.cached_token_price == 0.005

        # Test without usage data
        mock_response_no_usage = Mock()
        mock_response_no_usage.usage = None

        usage_stats_no_usage = UsageStatistics.from_chat_completion_response(
            mock_response_no_usage
        )

        assert usage_stats_no_usage is None

    @pytest.mark.parametrize(
        "initial_stats,chunk_usage,chunk_model,expected_stats",
        [
            # Test with usage data and cached tokens
            (
                UsageStatistics(),
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                    "cached_tokens": 200,
                },
                "gpt-4",
                UsageStatistics(
                    prompt_tokens=1000,
                    completion_tokens=500,
                    total_tokens=1500,
                    cached_prompt_tokens=200,
                    model="gpt-4",
                ),
            ),
            # Test without usage data (should reset existing values)
            (
                UsageStatistics(
                    prompt_tokens=500,
                    completion_tokens=300,
                    total_tokens=800,
                ),
                None,
                None,
                UsageStatistics(),
            ),
        ],
    )
    def test_update_from_stream_chunk(
        self,
        initial_stats: UsageStatistics,
        chunk_usage: Optional[Dict[str, Any]],
        chunk_model: Optional[str],
        expected_stats: UsageStatistics,
    ) -> None:
        """Test updating UsageStatistics from stream chunk."""
        # Setup mock chunk
        mock_chunk = Mock()
        mock_chunk.model = chunk_model

        if chunk_usage:
            mock_prompt_tokens_details = Mock()
            mock_prompt_tokens_details.cached_tokens = chunk_usage.get("cached_tokens")

            mock_usage = Mock()
            mock_usage.prompt_tokens = chunk_usage["prompt_tokens"]
            mock_usage.completion_tokens = chunk_usage["completion_tokens"]
            mock_usage.total_tokens = chunk_usage["total_tokens"]
            mock_usage.prompt_tokens_details = mock_prompt_tokens_details
            mock_chunk.usage = mock_usage
        else:
            mock_chunk.usage = None

        # Update usage statistics
        initial_stats.update_from_stream_chunk(mock_chunk)

        # Assertions - compare the two UsageStatistics objects
        assert initial_stats.prompt_tokens == expected_stats.prompt_tokens
        assert initial_stats.completion_tokens == expected_stats.completion_tokens
        assert initial_stats.total_tokens == expected_stats.total_tokens
        assert initial_stats.cached_prompt_tokens == expected_stats.cached_prompt_tokens
        assert initial_stats.model == expected_stats.model


class TestCreateChatMessageFromDict:
    """Test class for parse_chat_message_from_dict utility function."""

    @pytest.mark.parametrize(
        "message_data,expected_type,expected_role,should_raise,expected_error",
        [
            # Test case 1: Parsing user chat message
            (
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, how can I help you?"}],
                },
                UserChatMessage,
                "user",
                False,
                None,
            ),
            # Test case 2: Parsing Copilot chat message
            (
                {
                    "role": "copilot",
                    "content": [{"type": "text", "text": "I can help you with that!"}],
                },
                CopilotChatMessage,
                "copilot",
                False,
                None,
            ),
            # Test case 3: Parsing internal copilot chat message
            (
                {
                    "role": "internal_copilot_request",
                    "content": [
                        {"type": "text", "text": "Analyzing training error logs"}
                    ],
                    "response_category": "training_error_log_analysis",
                },
                InternalCopilotRequestChatMessage,
                "internal_copilot_request",
                False,
                None,
            ),
            # Test case 4: Parsing unknown role
            (
                {
                    "role": "unknown_role",
                    "content": [{"type": "text", "text": "This should fail"}],
                },
                None,
                None,
                True,
                ValueError,
            ),
            # Test case 5: Handling errors with correct role but wrong info
            (
                {
                    "role": "user",
                    "content": "invalid_content_format",  # Should be a list
                },
                None,
                None,
                True,
                ValidationError,
            ),
            # Test case 6: Missing role field
            (
                {
                    "content": [{"type": "text", "text": "This should fail"}],
                },
                None,
                None,
                True,
                ValueError,
            ),
        ],
    )
    def test_parse_chat_message_from_dict(
        self,
        message_data: Dict[str, Any],
        expected_type: type,
        expected_role: str,
        should_raise: bool,
        expected_error: type,
    ) -> None:
        """Test parse_chat_message_from_dict with various message types and errors."""
        if should_raise:
            with pytest.raises(expected_error):
                create_chat_message_from_dict(message_data)
        else:
            message = create_chat_message_from_dict(message_data)
            assert isinstance(message, expected_type)
            assert message.role == expected_role
