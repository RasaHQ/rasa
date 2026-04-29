"""Dataset models for retrieval evaluation experiments.

Defines the schema for retrieval eval dataset entries stored in Langfuse.
Each entry maps a search query to its ground-truth relevant documentation pages,
labeled independently of any retrieval system.
"""

from typing import Any, Dict, List, Literal

import structlog
from pydantic import BaseModel, Field

structlogger = structlog.get_logger()


class RetrievalDatasetInput(BaseModel):
    """Input for a retrieval evaluation item — the search query."""

    query: str = Field(description="The documentation search query to evaluate.")


class RelevantPage(BaseModel):
    """A single documentation page that is relevant to a query."""

    doc_id: str = Field(
        description="Path-based page ID, e.g. 'pro/build/writing-flows'."
    )
    url: str = Field(
        description="Full canonical URL, e.g. 'https://rasa.com/docs/pro/build/writing-flows'."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="LLM judge confidence in the relevance label."
    )


class RetrievalExpectedOutput(BaseModel):
    """Ground-truth labels for a retrieval evaluation item."""

    relevant_pages: List[RelevantPage] = Field(
        description="Documentation pages that answer or are relevant to the query."
    )


class RetrievalDatasetMetadata(BaseModel):
    """Metadata for a retrieval evaluation item."""

    category: str = Field(
        description=(
            "Query category: 'how-to', 'concept', 'debugging', 'code-generation'"
        )
    )
    source: str = Field(
        description="Where the query came from: 'system_prompt', 'langfuse', 'manual'."
    )
    docs_repo_commit: str = Field(
        description="Git SHA of the docs repo when labels were generated."
    )


class RetrievalDatasetEntry(BaseModel):
    """A single retrieval evaluation dataset entry.

    Maps to a Langfuse dataset item with input, expected_output, and metadata.
    """

    id: str = Field(description="Unique identifier for this dataset entry.")
    input: RetrievalDatasetInput
    expected_output: RetrievalExpectedOutput
    metadata: RetrievalDatasetMetadata

    def to_langfuse_item_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for ``langfuse.create_dataset_item()``.

        Returns:
            Dict with ``id``, ``input``, ``expected_output``, and ``metadata``
            ready to be unpacked into the Langfuse API call.
        """
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
    ) -> "RetrievalDatasetEntry":
        """Create a RetrievalDatasetEntry from raw Langfuse ExperimentItem dicts.

        Args:
            id: The dataset entry ID.
            input_data: Raw input dictionary (must contain ``query``).
            expected_output_data: Raw expected output dictionary
                (must contain ``relevant_pages``).
            metadata_data: Raw metadata dictionary.

        Returns:
            Parsed RetrievalDatasetEntry.
        """
        return cls(
            id=id,
            input=RetrievalDatasetInput.model_validate(input_data),
            expected_output=RetrievalExpectedOutput.model_validate(
                expected_output_data
            ),
            metadata=RetrievalDatasetMetadata.model_validate(metadata_data),
        )
