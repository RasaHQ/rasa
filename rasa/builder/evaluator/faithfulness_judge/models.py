"""Models for faithfulness judge."""

import uuid
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from rasa.builder.evaluator.content_processors.models import (
    BaseEvidence,
    Claims,
    CodeEvidence,
    DocumentationEvidence,
)
from rasa.builder.evaluator.dataset.models import DatasetEntry


class FaithfulnessVerdictLabel(str, Enum):
    """Verdict label for a claim evaluation."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFO = "not_enough_info"


class ClaimVerdict(BaseModel):
    """Model for a single claim faithfulness verdict.

    Represents the faithfulness verdict for one claim, including which
    evidence was used and the rationale for the decision.
    """

    claim_id: uuid.UUID = Field(..., description="ID of the evaluated claim")
    evidence_ids: List[uuid.UUID] = Field(
        default_factory=list,
        description="IDs of evidence entries used to make the verdict on the claim",
    )

    verdict: FaithfulnessVerdictLabel = Field(..., description="Verdict for the claim")
    rationale: str = Field(
        ..., description="Short explanation of why this verdict was given"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge's confidence in the verdict (0-1)",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class FaithfulnessJudgeResult(BaseModel):
    """Model for faithfulness evaluation results for a single dataset entry."""

    verdicts: List[ClaimVerdict] = Field(
        ..., description="List of verdicts for each claim"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @property
    def supported_verdicts(self) -> List[ClaimVerdict]:
        """Return verdicts for supported claims."""
        return [
            v for v in self.verdicts if v.verdict == FaithfulnessVerdictLabel.SUPPORTED
        ]

    @property
    def contradicted_verdicts(self) -> List[ClaimVerdict]:
        """Return verdicts for contradicted claims."""
        return [
            v
            for v in self.verdicts
            if v.verdict == FaithfulnessVerdictLabel.CONTRADICTED
        ]

    @property
    def not_enough_info_verdicts(self) -> List[ClaimVerdict]:
        """Return verdicts for claims with insufficient information."""
        return [
            v
            for v in self.verdicts
            if v.verdict == FaithfulnessVerdictLabel.NOT_ENOUGH_INFO
        ]


class FaithfulnessEvaluationFailure(BaseModel):
    """Model for a faithfulness evaluation failure."""

    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error that occurred")


class FaithfulnessJudgeInput(BaseModel):
    """Model representing the single input for the faithfulness judge."""

    claims: Claims = Field(..., description="Claims extracted from Copilot response")
    evidence: List[BaseEvidence] = Field(
        ..., description="List of evidence (documentation and/or code)"
    )

    @property
    def documentation_evidence(self) -> List[DocumentationEvidence]:
        return [
            evidence
            for evidence in self.evidence
            if isinstance(evidence, DocumentationEvidence)
        ]

    @property
    def code_evidence(self) -> List[CodeEvidence]:
        return [
            evidence for evidence in self.evidence if isinstance(evidence, CodeEvidence)
        ]

    @classmethod
    def from_dataset_entry_and_claims(
        cls,
        dataset_entry: DatasetEntry,
        claims: Claims,
    ) -> "FaithfulnessJudgeInput":
        """Create a FaithfulnessJudgeInput from a DatasetEntry and Claims."""
        documentation_evidence = DocumentationEvidence.from_dataset_entry(dataset_entry)
        code_evidence = CodeEvidence.from_dataset_entry(dataset_entry)
        all_evidence: List[BaseEvidence] = [
            *documentation_evidence,
            *code_evidence,
        ]
        return cls(claims=claims, evidence=all_evidence)
