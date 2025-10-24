from typing import Any, Dict, List, Optional, Union

import structlog
from pydantic import BaseModel, Field, field_validator

from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotContext,
    EventContent,
    ReferenceEntry,
    ResponseCategory,
    create_chat_message_from_dict,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.shared.tracker_context import TrackerContext

structlogger = structlog.get_logger()


class DatasetInput(BaseModel):
    """Model for the input field of a dataset entry."""

    message: Optional[str] = None
    tracker_event_attachments: List[EventContent] = Field(default_factory=list)


class DatasetExpectedOutput(BaseModel):
    """Model for the expected_output field of a dataset entry."""

    answer: str
    response_category: ResponseCategory
    references: list[ReferenceEntry]


class DatasetMetadataCopilotAdditionalContext(BaseModel):
    """Model for the copilot_additional_context in metadata."""

    relevant_documents: List[Document] = Field(default_factory=list)
    relevant_assistant_files: Dict[str, str] = Field(default_factory=dict)
    assistant_tracker_context: Optional[Dict[str, Any]] = None
    assistant_logs: str = Field(default="")
    copilot_chat_history: List[ChatMessage] = Field(default_factory=list)

    @field_validator("copilot_chat_history", mode="before")
    @classmethod
    def parse_chat_history(
        cls, v: Union[List[Dict[str, Any]], List[ChatMessage]]
    ) -> List[ChatMessage]:
        """Manually parse chat history messages based on role field."""
        # If already parsed ChatMessage objects, return them as-is
        if (
            v
            and isinstance(v, list)
            and all(isinstance(item, ChatMessage) for item in v)
        ):
            return v  # type: ignore[return-value]

        # Check for mixed types (some ChatMessage, some not)
        if (
            v
            and isinstance(v, list)
            and any(isinstance(item, ChatMessage) for item in v)
        ):
            message = (
                "Mixed types in copilot_chat_history: cannot mix ChatMessage objects "
                "with other types."
            )
            structlogger.error(
                "dataset_entry.parse_chat_history.mixed_types",
                event_info=message,
                chat_history_types=[type(item) for item in v],
            )
            raise ValueError(message)

        # Otherwise, parse from dictionaries
        parsed_messages: List[ChatMessage] = []
        for message_data in v:
            chat_message = create_chat_message_from_dict(message_data)
            parsed_messages.append(chat_message)
        return parsed_messages


class DatasetMetadata(BaseModel):
    """Model for the metadata field of a dataset entry."""

    ids: Dict[str, str] = Field(default_factory=dict)
    copilot_additional_context: DatasetMetadataCopilotAdditionalContext = Field(
        default_factory=DatasetMetadataCopilotAdditionalContext
    )


class DatasetEntry(BaseModel):
    """Pydantic model for dataset entries from Langfuse ExperimentItem."""

    # Basic fields from ExperimentItem
    id: str
    input: DatasetInput
    expected_output: DatasetExpectedOutput
    metadata: DatasetMetadata

    def to_copilot_context(self) -> CopilotContext:
        """Create a CopilotContext from the dataset entry.

        Raises:
            ValueError: If the metadata is None, as it's required for creating a valid
                CopilotContext.

        Returns:
            CopilotContext with all the context information.
        """
        if self.metadata is None:
            message = (
                f"Cannot create CopilotContext from dataset item with id: {self.id}. "
                f"Metadata is required but was None."
            )
            structlogger.error(
                "dataset_entry.to_copilot_context.metadata_is_none",
                event_info=message,
                item_id=self.id,
                item_metadata=self.metadata,
            )
            raise ValueError(message)

        # Parse tracker context if available
        tracker_context = None
        if (
            self.metadata.copilot_additional_context.assistant_tracker_context
            is not None
        ):
            tracker_context = TrackerContext(
                **self.metadata.copilot_additional_context.assistant_tracker_context
            )

        return CopilotContext(
            tracker_context=tracker_context,
            assistant_logs=self.metadata.copilot_additional_context.assistant_logs,
            assistant_files=self.metadata.copilot_additional_context.relevant_assistant_files,
            copilot_chat_history=self.metadata.copilot_additional_context.copilot_chat_history,
        )

    @classmethod
    def from_raw_data(
        cls,
        id: str,
        input_data: Dict[str, Any],
        expected_output_data: Dict[str, Any],
        metadata_data: Dict[str, Any],
    ) -> "DatasetEntry":
        """Create a DatasetEntry from raw dictionary data.

        Args:
            id: The dataset entry ID.
            input_data: Raw input dictionary.
            expected_output_data: Raw expected output dictionary.
            metadata_data: Raw metadata dictionary with all the additional context
                used to generate the Copilot response.

        Returns:
            DatasetEntry with parsed data.
        """
        # Use Pydantic's model_validate to parse nested structures
        dataset_input = DatasetInput.model_validate(input_data)
        dataset_expected_output = DatasetExpectedOutput.model_validate(
            expected_output_data
        )
        dataset_metadata = DatasetMetadata.model_validate(metadata_data)

        return cls(
            id=id,
            input=dataset_input,
            expected_output=dataset_expected_output,
            metadata=dataset_metadata,
        )
