"""Models for completeness judge."""

import uuid
from typing import List

from pydantic import BaseModel, Field

from rasa.builder.copilot.models import ChatMessage
from rasa.builder.evaluator.content_processors.models import Claims
from rasa.builder.evaluator.dataset.models import DatasetEntry


class CompletenessJudgeInput(BaseModel):
    """Model representing the input for the completeness judge."""

    user_message: str = Field(..., description="The most recent user message")
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Full copilot chat history for context",
    )
    claims: Claims = Field(..., description="Claims extracted from Copilot response")

    @classmethod
    def from_dataset_entry_and_claims(
        cls,
        dataset_entry: DatasetEntry,
        claims: Claims,
    ) -> "CompletenessJudgeInput":
        user_message = dataset_entry.input.message or ""
        chat_history = (
            dataset_entry.metadata.copilot_additional_context.copilot_chat_history
        )
        return cls(
            user_message=user_message,
            chat_history=chat_history,
            claims=claims,
        )


class UserRequestCompletenessVerdict(BaseModel):
    """Model for the completeness verdict for a part of the user request.

    Represents one distinct aspect or question within the user's request.
    """

    part_text: str = Field(
        ..., description="Text describing this atomic part of the user request"
    )
    addressing_claims_ids: List[uuid.UUID] = Field(
        default_factory=list,
        description="List of claim UUIDs that address this part of the request",
    )
    rationale: str = Field(
        ...,
        description=(
            "Explanation of how the claims address (or don't address) this part"
        ),
    )

    @property
    def is_covered(self) -> bool:
        """Check if this part of the user request is covered by any claims."""
        return len(self.addressing_claims_ids) > 0


class CompletenessJudgeResult(BaseModel):
    """Model for the completeness judge result."""

    verdicts: List[UserRequestCompletenessVerdict] = Field(
        ..., description="List of verdicts for each part of the user request"
    )
    overall_rationale: str = Field(
        ..., description="Overall rationale for the completeness assessment"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the completeness assessment (0-1)",
    )

    @property
    def covered_parts(self) -> List[UserRequestCompletenessVerdict]:
        """Get the covered parts of the user request."""
        return [verdict for verdict in self.verdicts if verdict.is_covered]

    @property
    def uncovered_parts(self) -> List[UserRequestCompletenessVerdict]:
        """Get the uncovered parts of the user request."""
        return [verdict for verdict in self.verdicts if not verdict.is_covered]
