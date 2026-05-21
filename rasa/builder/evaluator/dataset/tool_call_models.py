"""Dataset models for tool call evaluation experiments.

Defines the schema for tool-call eval dataset entries stored in Langfuse.
Each entry maps a user query (plus optional copilot context) to its
ground-truth ordered list of expected tool calls.
"""

from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

from rasa.builder.copilot.models import (
    CopilotContext,
    EventContent,
    TextContent,
    UserChatMessage,
)
from rasa.builder.evaluator.dataset.classifier_models import (
    DatasetMetadataCopilotAdditionalContext,
)
from rasa.builder.shared.tracker_context import TrackerContext

structlogger = structlog.get_logger()


class ToolCallDatasetInput(BaseModel):
    """Input for a tool call evaluation item — the user query."""

    query: str = Field(description="The user query to evaluate.")
    tracker_event_attachments: List[EventContent] = Field(default_factory=list)


class ToolCallExpectedOutput(BaseModel):
    """Ground-truth labels for a tool call evaluation item."""

    expected_tools: List[str] = Field(
        description="Ordered list of tool names the copilot should call."
    )


class ToolCallDatasetMetadata(BaseModel):
    """Metadata for a tool call evaluation item."""

    category: str = Field(
        description="Query category (free-form tag, e.g. 'flow-edit', 'debugging')."
    )
    source: str = Field(
        description="Where the query came from: 'system_prompt', 'langfuse', 'manual'."
    )
    copilot_additional_context: DatasetMetadataCopilotAdditionalContext = Field(
        default_factory=DatasetMetadataCopilotAdditionalContext,
    )


class ToolCallDatasetEntry(BaseModel):
    """A single tool call evaluation dataset entry."""

    id: str = Field(description="Unique identifier for this dataset entry.")
    input: ToolCallDatasetInput
    expected_output: ToolCallExpectedOutput
    metadata: ToolCallDatasetMetadata

    def to_copilot_context(self) -> CopilotContext:
        """Build a CopilotContext for invoking the copilot.

        The user query is appended as a final ``UserChatMessage`` to any
        prior history carried in ``metadata.copilot_additional_context``.
        """
        additional = self.metadata.copilot_additional_context

        tracker_context: Optional[TrackerContext] = None
        if additional.assistant_tracker_context is not None:
            tracker_context = TrackerContext(**additional.assistant_tracker_context)

        chat_history = list(additional.copilot_chat_history)
        chat_history.append(
            UserChatMessage(content=[TextContent(type="text", text=self.input.query)])
        )

        return CopilotContext(
            tracker_context=tracker_context,
            assistant_logs=additional.assistant_logs,
            assistant_files=additional.relevant_assistant_files,
            copilot_chat_history=chat_history,
        )

    def to_langfuse_item_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for ``langfuse.create_dataset_item()``."""
        return {
            "id": self.id,
            "input": self.input.model_dump(),
            "expected_output": self.expected_output.model_dump(),
            "metadata": self.metadata.model_dump(),
        }

    @classmethod
    def from_raw_data(
        cls,
        id: str,
        input_data: Dict[str, Any],
        expected_output_data: Dict[str, Any],
        metadata_data: Dict[str, Any],
    ) -> "ToolCallDatasetEntry":
        """Create a ``ToolCallDatasetEntry`` from raw Langfuse dicts."""
        return cls(
            id=id,
            input=ToolCallDatasetInput.model_validate(input_data),
            expected_output=ToolCallExpectedOutput.model_validate(expected_output_data),
            metadata=ToolCallDatasetMetadata.model_validate(metadata_data),
        )
