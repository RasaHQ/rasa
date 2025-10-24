"""Unit tests for the DatasetEntry model."""

from typing import Any, Dict, List, Optional

import pytest

from rasa.builder.copilot.models import (
    CopilotContext,
    ResponseCategory,
    UserChatMessage,
)
from rasa.builder.evaluator.dataset.models import (
    DatasetEntry,
    DatasetExpectedOutput,
    DatasetInput,
    DatasetMetadata,
    DatasetMetadataCopilotAdditionalContext,
)
from rasa.builder.shared.tracker_context import TrackerContext


class TestDatasetEntry:
    @pytest.mark.parametrize(
        "assistant_tracker_context,"
        "expected_tracker_context,"
        "assistant_files,"
        "expected_files,"
        "chat_history,"
        "expected_chat_history",
        (
            # With tracker context
            (
                {
                    "conversation_turns": [
                        {
                            "user_message": {"text": "Hello", "predicted_commands": []},
                            "assistant_messages": [{"text": "Hi there!"}],
                            "context_events": [],
                        }
                    ],
                    "current_state": {
                        "latest_message": "Hello",
                        "active_flow": None,
                        "flow_stack": None,
                        "slots": {"user_name": "John"},
                        "latest_action": "action_listen",
                        "followup_action": None,
                    },
                },
                True,  # Should have tracker context
                {"file1.py": "print('hello')"},
                {"file1.py": "print('hello')"},
                [
                    UserChatMessage(
                        role="user",
                        content=[{"type": "text", "text": "Hello"}],
                        response_category=None,
                    )
                ],
                1,  # Expected chat history length
            ),
            # Without tracker context
            (
                None,
                False,  # Should not have tracker context
                {},
                {},
                [],
                0,  # Expected chat history length
            ),
        ),
    )
    def test_to_copilot_context_with_and_without_tracker_context(
        self,
        assistant_tracker_context: Optional[Dict[str, Any]],
        expected_tracker_context: bool,
        assistant_files: Dict[str, str],
        expected_files: Dict[str, str],
        chat_history: List[UserChatMessage],
        expected_chat_history: int,
    ) -> None:
        """Test creating CopilotContext with and without tracker context."""
        # Given
        dataset_entry = DatasetEntry(
            id="test-id",
            input=DatasetInput(message="Test message"),
            expected_output=DatasetExpectedOutput(
                answer="Test answer",
                response_category=ResponseCategory.COPILOT,
                references=[],
            ),
            metadata=DatasetMetadata(
                ids={"experiment_id": "exp-123"},
                copilot_additional_context=DatasetMetadataCopilotAdditionalContext(
                    relevant_documents=[],
                    relevant_assistant_files=assistant_files,
                    assistant_tracker_context=assistant_tracker_context,
                    assistant_logs="Test logs",
                    copilot_chat_history=chat_history,
                ),
            ),
        )

        # When
        result = dataset_entry.to_copilot_context()

        # Then
        assert isinstance(result, CopilotContext)
        if expected_tracker_context:
            assert result.tracker_context is not None
            assert isinstance(result.tracker_context, TrackerContext)
        else:
            assert result.tracker_context is None
        assert result.assistant_logs == "Test logs"
        assert result.assistant_files == expected_files
        assert len(result.copilot_chat_history) == expected_chat_history
        if expected_chat_history > 0:
            assert result.copilot_chat_history[0].content[0].text == "Hello"

    def test_to_copilot_context_with_none_metadata_raises_error(self) -> None:
        """Test that ValueError is raised when metadata is None."""
        # Given
        dataset_entry = DatasetEntry(
            id="test-id",
            input=DatasetInput(message="Test message"),
            expected_output=DatasetExpectedOutput(
                answer="Test answer",
                response_category=ResponseCategory.COPILOT,
                references=[],
            ),
            metadata=DatasetMetadata(
                ids={},
                copilot_additional_context=DatasetMetadataCopilotAdditionalContext(),
            ),
        )
        # Manually set metadata to None to test the error case
        dataset_entry.metadata = None

        # When & Then
        expected_error_message = (
            "Cannot create CopilotContext from dataset item with id: test-id. "
            "Metadata is required but was None."
        )
        with pytest.raises(ValueError, match=expected_error_message):
            dataset_entry.to_copilot_context()

    @pytest.mark.parametrize(
        "input_data,"
        "expected_output_data,"
        "metadata_data,"
        "expected_message,"
        "expected_response_category",
        (
            # Minimal data
            (
                {"message": "Hello"},
                {
                    "answer": "Hi there!",
                    "response_category": "copilot",
                    "references": [],
                },
                {
                    "ids": {"experiment_id": "exp-123"},
                    "copilot_additional_context": {
                        "relevant_documents": [],
                        "relevant_assistant_files": {},
                        "assistant_tracker_context": None,
                        "assistant_logs": "",
                        "copilot_chat_history": [],
                    },
                },
                "Hello",
                ResponseCategory.COPILOT,
            ),
            # With tracker event attachments
            (
                {
                    "message": "Hello",
                    "tracker_event_attachments": [
                        {
                            "type": "event",
                            "event": "user_uttered",
                            "data": {"text": "Hello world"},
                        }
                    ],
                },
                {
                    "answer": "Hi there!",
                    "response_category": "reference",
                    "references": [
                        {
                            "index": 0,
                            "title": "Test Reference",
                            "url": "https://example.com",
                            "response_category": "reference_entry",
                            "response_completeness": "complete",
                        }
                    ],
                },
                {
                    "ids": {"experiment_id": "exp-123"},
                    "copilot_additional_context": {
                        "relevant_documents": [
                            {
                                "content": "Test document content",
                                "title": "Test Document",
                                "url": "https://example.com/doc",
                            }
                        ],
                        "relevant_assistant_files": {"config.yml": "version: 3.1"},
                        "assistant_tracker_context": {
                            "conversation_turns": [
                                {
                                    "user_message": {
                                        "text": "Hello",
                                        "predicted_commands": [],
                                    },
                                    "assistant_messages": [{"text": "Hi!"}],
                                    "context_events": [],
                                }
                            ],
                            "current_state": {
                                "latest_message": "Hello",
                                "active_flow": None,
                                "flow_stack": None,
                                "slots": {"user_name": "John"},
                                "latest_action": "action_listen",
                                "followup_action": None,
                            },
                        },
                        "assistant_logs": "Some logs here",
                        "copilot_chat_history": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Hello"}],
                                "response_category": None,
                            }
                        ],
                    },
                },
                "Hello",
                ResponseCategory.REFERENCE,
            ),
        ),
    )
    def test_from_raw_data_with_various_scenarios(
        self,
        input_data: Dict[str, Any],
        expected_output_data: Dict[str, Any],
        metadata_data: Dict[str, Any],
        expected_message: str,
        expected_response_category: ResponseCategory,
    ) -> None:
        """Test creating DatasetEntry with various data scenarios."""
        # Given
        test_id = "test-id"

        # When
        result = DatasetEntry.from_raw_data(
            id=test_id,
            input_data=input_data,
            expected_output_data=expected_output_data,
            metadata_data=metadata_data,
        )

        # Then
        assert isinstance(result, DatasetEntry)
        assert result.id == test_id
        assert result.input.message == expected_message
        assert result.expected_output.answer == expected_output_data["answer"]
        assert result.expected_output.response_category == expected_response_category
        assert result.metadata.ids == metadata_data["ids"]

        # Verify tracker event attachments if present
        if "tracker_event_attachments" in input_data:
            assert len(result.input.tracker_event_attachments) == len(
                input_data["tracker_event_attachments"]
            )
            assert result.input.tracker_event_attachments[0].type == "event"
        else:
            assert result.input.tracker_event_attachments == []

        # Verify references if present
        assert len(result.expected_output.references) == len(
            expected_output_data["references"]
        )
        if expected_output_data["references"]:
            assert result.expected_output.references[0].title == "Test Reference"

        # Verify metadata context
        copilot_context = metadata_data["copilot_additional_context"]
        assert (
            result.metadata.copilot_additional_context.assistant_logs
            == copilot_context.get("assistant_logs", "")
        )
        assert (
            result.metadata.copilot_additional_context.relevant_assistant_files
            == copilot_context.get("relevant_assistant_files", {})
        )

        # Verify tracker context
        if copilot_context.get("assistant_tracker_context") is not None:
            assert (
                result.metadata.copilot_additional_context.assistant_tracker_context
                is not None
            )
        else:
            assert (
                result.metadata.copilot_additional_context.assistant_tracker_context
                is None
            )

    def test_from_raw_data_invalid_data_raises_validation_error(self) -> None:
        """Test that invalid data raises validation errors."""
        # Given
        invalid_input_data: Dict[str, Any] = {
            "message": 123
        }  # Invalid type for message
        expected_output_data: Dict[str, Any] = {
            "answer": "Test",
            "response_category": "copilot",
            "references": [],
        }
        metadata_data: Dict[str, Dict[str, Any]] = {
            "ids": {},
            "copilot_additional_context": {},
        }

        # When & Then
        with pytest.raises(Exception):  # Pydantic validation error
            DatasetEntry.from_raw_data(
                id="test-id",
                input_data=invalid_input_data,
                expected_output_data=expected_output_data,
                metadata_data=metadata_data,
            )

    def test_from_raw_data_invalid_response_category_raises_error(self) -> None:
        """Test that invalid response category raises validation error."""
        # Given
        input_data: Dict[str, str] = {"message": "Test"}
        invalid_expected_output_data: Dict[str, Any] = {
            "answer": "Test",
            "response_category": "invalid_category",
            "references": [],
        }
        metadata_data: Dict[str, Dict[str, Any]] = {
            "ids": {},
            "copilot_additional_context": {},
        }

        # When & Then
        with pytest.raises(Exception):  # Pydantic validation error
            DatasetEntry.from_raw_data(
                id="test-id",
                input_data=input_data,
                expected_output_data=invalid_expected_output_data,
                metadata_data=metadata_data,
            )
