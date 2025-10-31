import uuid
from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set

import structlog
from pydantic import BaseModel, Field

from rasa.builder.evaluator.content_processors.constants import (
    CODE_EVIDENCE_TYPE,
    DOCUMENTATION_EVIDENCE_TYPE,
)
from rasa.builder.evaluator.dataset.models import DatasetEntry

structlogger = structlog.get_logger()


class BaseEvidence(BaseModel, ABC):
    """Base model for evidence."""

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the evidence",
        frozen=True,
    )
    type: str = Field(..., description="Type of evidence", frozen=True)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
        frozen=True,
    )


class DocumentationEvidence(BaseEvidence):
    """Model for documentation evidence used in claim verification.

    Documentation evidence represents passages from retrieved documentation
    that serve as ground truth for evaluating claims in Copilot responses.
    """

    type: Literal["documentation_evidence"] = Field(
        default=DOCUMENTATION_EVIDENCE_TYPE,
        pattern=f"^{DOCUMENTATION_EVIDENCE_TYPE}",
        description="Type of evidence",
    )
    url: str = Field(..., description="URL of the documentation source")
    title: Optional[str] = Field(..., description="Title of the documentation source")
    content: str = Field(..., description="Content used by the LLM Judge")
    used: bool = Field(
        ..., description="Whether this evidence was used in the response"
    )

    @classmethod
    def from_dataset_entry(
        cls, dataset_entry: DatasetEntry
    ) -> List["DocumentationEvidence"]:
        """Create a list of DocumentationEvidence from a DatasetEntry.

        Args:
            dataset_entry: The dataset entry containing relevant documents.

        Returns:
            List of DocumentationEvidence instances from the dataset entry's relevant
            documents.
        """
        evidence_list: List[DocumentationEvidence] = []

        # Get the set of referenced document URLs for quick lookup
        referenced_urls: Set[str] = {
            reference.url for reference in dataset_entry.expected_output.references
        }

        for (
            retrieved_document
        ) in dataset_entry.metadata.copilot_additional_context.relevant_documents:
            # Validate URL is not None
            if retrieved_document.url is None:
                message = "Document's URL is None"
                structlogger.error(
                    "documentation_evidence.from_dataset_entry.document_url_is_none",
                    event_info=message,
                    document_url=retrieved_document.url,
                    document_title=retrieved_document.title,
                    document_content=retrieved_document.content,
                )
                raise ValueError(message)

            # Check if this document was actually referenced in the response
            was_used = retrieved_document.url in referenced_urls

            evidence = cls(
                url=retrieved_document.url,
                title=retrieved_document.title,
                content=retrieved_document.content,
                used=was_used,
                metadata={},
            )
            evidence_list.append(evidence)

        return evidence_list


class CodeEvidence(BaseEvidence):
    """Model for code evidence used in claim verification.

    Code evidence represents code snippets from retrieved code that serve as ground
    truth for evaluating claims in Copilot responses.
    """

    type: Literal["code_evidence"] = Field(
        default=CODE_EVIDENCE_TYPE,
        pattern=f"^{CODE_EVIDENCE_TYPE}",
        description="Type of evidence",
    )
    file_path: str = Field(..., description="Path to the code file")
    file_content: str = Field(..., description="Content of the code file")
    referenced: bool = Field(
        ..., description="Whether the code file was referenced in the response"
    )

    @classmethod
    def from_dataset_entry(
        cls,
        dataset_entry: DatasetEntry,
        use_only_files_referenced_in_response: bool = False,
    ) -> List["CodeEvidence"]:
        """Create a list of CodeEvidence from a DatasetEntry.

        Args:
            dataset_entry: The dataset entry containing relevant code.
            use_only_files_referenced_in_response: If True, only use files referenced
                in the response. If False, use all files provided in the dataset entry
                contex. Default is False.
        Returns:
            List of CodeEvidence instances from the dataset entry's relevant code.
        """
        evidence_list: List[CodeEvidence] = []

        # Extract file paths mentioned in the Copilot response
        response = dataset_entry.expected_output.answer
        mentioned_file_paths = cls._extract_file_paths_from_response(response)

        for (
            assistant_file_name,
            assistant_file_content,
        ) in dataset_entry.metadata.copilot_additional_context.relevant_assistant_files.items():  # noqa: E501
            # Check if this file was actually referenced in the response
            was_referenced_in_response = cls._check_if_path_is_used(
                assistant_file_name, mentioned_file_paths
            )

            # Skip this file if filtering is enabled and it wasn't referenced in
            # the response
            if use_only_files_referenced_in_response and not was_referenced_in_response:
                continue

            evidence = cls(
                file_path=assistant_file_name,
                file_content=assistant_file_content,
                metadata={},
                referenced=was_referenced_in_response,
            )
            evidence_list.append(evidence)

        return evidence_list

    @staticmethod
    def _extract_file_paths_from_response(response: str) -> Set[str]:
        """Extract file paths mentioned in Copilot response.

        Based on the system prompt format: **File: `path/to/file`**
        File paths are relative to project root
        (e.g., 'domain.yml', 'data/flows/booking.yml')

        Args:
            response: The Copilot response text.

        Returns:
            Set of file paths mentioned in the response.
        """
        import re

        # Pattern to match: **File: `path/to/file`** This is the pattern that the
        # Copilot is instructed to use in the system prompt.
        pattern = r"\*\*File:\s*`([^`]+)`\*\*"
        matches = re.findall(pattern, response)

        # Also look for other common file path patterns. Pattern for paths in code
        # blocks or inline mentions
        additional_patterns = [
            r"`([^`]*\.(?:py|yml|yaml|json|md|txt))`",
            r"`([^`]*/(?:domain|config|flows|actions|endpoints)\.yml)`",
            r"`([^`]*(?:domain|config|flows|actions|endpoints)\.yml)`",
        ]

        for pattern in additional_patterns:
            matches.extend(re.findall(pattern, response))

        return set(matches)

    @staticmethod
    def _check_if_path_is_used(referenced_file: str, file_paths: Set[str]) -> bool:
        """Check if a referenced file path is mentioned in the response.

        Args:
            referenced_file: The file path from assistant_files.
            file_paths: Set of file paths extracted from the response.

        Returns:
            True if the file is mentioned (fully or partially), False otherwise.
        """
        # Check for exact match
        if referenced_file in file_paths:
            return True

        # Check for partial matches (filename only, or path segments)
        referenced_filename = referenced_file.split("/")[-1]
        for response_path in file_paths:
            response_filename = response_path.split("/")[-1]

            # Check if filenames match
            if referenced_filename == response_filename:
                return True

            # Check if referenced file is a substring of response path
            if referenced_file in response_path:
                return True

            # Check if response path is a substring of referenced file
            if response_path in referenced_file:
                return True

        return False


class ClaimImportance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Claim(BaseModel):
    """Model for atomic, verifiable claims extracted from Copilot responses.

    Claims represent individual, testable statements that can be verified
    against evidence sources like documentation or code.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the claim",
        frozen=True,
    )
    importance: ClaimImportance = Field(
        ..., description="Importance level of the claim"
    )
    text: str = Field(..., description="Short textual content of the claim")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional metadata including targets (domain, flows, action, etc.)"
        ),
    )


class Claims(BaseModel):
    """Model for a list of claims."""

    claims: List[Claim] = Field(..., description="List of claims")

    def __len__(self) -> int:
        """Return the number of claims."""
        return len(self.claims)

    def __getitem__(self, index: int) -> Claim:
        """Get a claim by index."""
        return self.claims[index]

    @property
    def high_importance_claims(self) -> List[Claim]:
        """Get the high importance claims."""
        return self.get_claims_by_importance(ClaimImportance.HIGH)

    @property
    def medium_importance_claims(self) -> List[Claim]:
        """Get the medium importance claims."""
        return self.get_claims_by_importance(ClaimImportance.MEDIUM)

    @property
    def low_importance_claims(self) -> List[Claim]:
        """Get the low importance claims."""
        return self.get_claims_by_importance(ClaimImportance.LOW)

    def get_claims_by_importance(self, importance: ClaimImportance) -> List[Claim]:
        """Get claims filtered by the given importance level."""
        return [claim for claim in self.claims if claim.importance == importance]


class ClaimExtractionFailure(BaseModel):
    """Model for a claim extraction failure."""

    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error that occurred")
