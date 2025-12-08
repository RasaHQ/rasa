"""Models for copilot response evaluator."""

from typing import Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from rasa.builder.evaluator.completeness_judge.models import (
    CompletenessJudgeResult,
    UserRequestCompletenessVerdict,
)
from rasa.builder.evaluator.content_processors.models import (
    Claim,
    ClaimExtractionFailure,
    ClaimImportance,
    Claims,
)
from rasa.builder.evaluator.dataset.models import DatasetEntry
from rasa.builder.evaluator.faithfulness_judge.models import (
    ClaimVerdict,
    FaithfulnessJudgeResult,
    FaithfulnessVerdictLabel,
)
from rasa.builder.evaluator.shared.models import EvaluationFailure


class ClaimWithVerdict(BaseModel):
    claim: Claim = Field(..., description="The extracted claim.")
    verdict: Optional[ClaimVerdict] = Field(
        default=None,
        description="The faithfulness verdict for this claim, if available",
    )


class CopilotResponseFaithfulnessReasoningMetadata(BaseModel):
    claims_with_verdicts: List[ClaimWithVerdict] = Field(
        ...,
        description="The claims with their associated verdicts, if available.",
    )


class PartWithClaims(BaseModel):
    """Model for a user request part paired with claims that address it."""

    part: UserRequestCompletenessVerdict = Field(
        ..., description="The user request part verdict."
    )
    addressing_claims: List[Claim] = Field(
        default_factory=list,
        description="The claims that address this part, if available.",
    )


class CopilotResponseCompletenessReasoningMetadata(BaseModel):
    parts_with_claims: List[PartWithClaims] = Field(
        ...,
        description=(
            "The user request parts with their addressing claims, if available."
        ),
    )


class CopilotResponseFaithfulnessMetrics(BaseModel):
    support_rate: float = Field(
        ...,
        description="Support rate (micro average): supported claims / total claims",
        ge=0.0,
        le=1.0,
    )
    weighted_support_rate: float = Field(
        ...,
        description=(
            "Weighted support rate: "
            "weighted by claim importance (high=3, medium=2, low=1)"
        ),
        ge=0.0,
        le=1.0,
    )
    avg_verdict_confidence: float = Field(
        ...,
        description="Average verdict confidence",
        ge=0.0,
        le=1.0,
    )
    supported_claims_count: int = Field(
        ...,
        description="Supported claims count",
        ge=0,
    )
    contradicted_claims_count: int = Field(
        ...,
        description="Contradicted claims count",
        ge=0,
    )
    not_enough_info_claims_count: int = Field(
        ...,
        description="Not enough info claims count",
        ge=0,
    )
    extracted_claims_count: int = Field(
        ...,
        description="Extracted claims count",
        ge=0,
    )


class CopilotResponseFaithfulnessEvaluationResult(BaseModel):
    entry_id: str = Field(..., description="ID of the dataset entry being evaluated")
    extracted_claims: Claims = Field(
        ..., description="Claims extracted from the Copilot response."
    )
    faithfulness_result: Union[FaithfulnessJudgeResult, EvaluationFailure] = Field(
        ..., description="Result from faithfulness judge"
    )

    @property
    def has_failures(self) -> bool:
        """Check if faithfulness evaluation failed.

        Returns:
            True if faithfulness evaluation failed
        """
        return isinstance(self.faithfulness_result, EvaluationFailure)

    @property
    def metrics(self) -> CopilotResponseFaithfulnessMetrics:
        """Calculate the metrics for the faithfulness evaluation."""
        # Return zero metrics for failures
        if (
            self.has_failures
            or len(self.extracted_claims) == 0
            or not isinstance(self.faithfulness_result, FaithfulnessJudgeResult)
        ):
            return CopilotResponseFaithfulnessMetrics(
                support_rate=0.0,
                weighted_support_rate=0.0,
                avg_verdict_confidence=0.0,
                supported_claims_count=0,
                contradicted_claims_count=0,
                not_enough_info_claims_count=0,
                extracted_claims_count=len(self.extracted_claims),
            )

        return CopilotResponseFaithfulnessMetrics(
            support_rate=self._calculate_micro_support_rate(),
            weighted_support_rate=self._calculate_weighted_support_rate(),
            avg_verdict_confidence=self._calculate_avg_verdict_confidence(),
            supported_claims_count=self._get_relevant_claim_count_by_type(
                FaithfulnessVerdictLabel.SUPPORTED
            ),
            contradicted_claims_count=self._get_relevant_claim_count_by_type(
                FaithfulnessVerdictLabel.CONTRADICTED
            ),
            not_enough_info_claims_count=self._get_relevant_claim_count_by_type(
                FaithfulnessVerdictLabel.NOT_ENOUGH_INFO
            ),
            extracted_claims_count=len(self.extracted_claims),
        )

    def _get_relevant_verdicts(self) -> list[ClaimVerdict]:
        """Get verdicts that correspond to extracted claims.

        Filters verdicts to only include those for claims that are in
        extracted_claims. This helps avoid counting verdicts for claims that
        were not extracted (hallucinated).

        Returns:
            List of verdicts for extracted claims only
        """
        if (
            self.has_failures
            or len(self.extracted_claims) == 0
            or not isinstance(self.faithfulness_result, FaithfulnessJudgeResult)
        ):
            return []

        result = self.faithfulness_result
        extracted_claim_ids = {claim.id for claim in self.extracted_claims.claims}

        return [
            verdict
            for verdict in result.verdicts
            if verdict.claim_id in extracted_claim_ids
        ]

    def _get_relevant_claim_count_by_type(
        self, verdict_type: FaithfulnessVerdictLabel
    ) -> int:
        """Get count of claims by verdict type.

        Only counts verdicts for claims that are in extracted_claims.

        Returns:
            Count of claims by verdict type
        """
        if (
            self.has_failures
            or len(self.extracted_claims) == 0
            or not isinstance(self.faithfulness_result, FaithfulnessJudgeResult)
        ):
            return 0

        relevant_verdicts = self._get_relevant_verdicts()
        return sum(
            1 for verdict in relevant_verdicts if verdict.verdict == verdict_type
        )

    def _calculate_micro_support_rate(self) -> float:
        """Calculate micro (unweighted) support rate.

        Only counts verdicts for claims that are in extracted_claims.

        Returns:
            Micro support rate as a float between 0.0 and 1.0
        """
        if (
            self.has_failures
            or len(self.extracted_claims) == 0
            or not isinstance(self.faithfulness_result, FaithfulnessJudgeResult)
        ):
            return 0.0

        relevant_verdicts = self._get_relevant_verdicts()

        if len(relevant_verdicts) == 0:
            return 0.0

        supported_count = sum(
            1
            for verdict in relevant_verdicts
            if verdict.verdict == FaithfulnessVerdictLabel.SUPPORTED
        )

        return supported_count / len(relevant_verdicts)

    def _calculate_weighted_support_rate(self) -> float:
        """Calculate weighted support rate by claim importance.

        Only counts verdicts for claims that are in extracted_claims.
        Uses weights: high=3.0, medium=2.0, low=1.0

        Returns:
            Weighted support rate as a float between 0.0 and 1.0
        """
        if (
            self.has_failures
            or len(self.extracted_claims) == 0
            or not isinstance(self.faithfulness_result, FaithfulnessJudgeResult)
        ):
            return 0.0

        # Create a mapping from claim ID to claim for importance lookup.
        claim_by_id = {claim.id: claim for claim in self.extracted_claims.claims}
        relevant_verdicts = self._get_relevant_verdicts()

        importance_weights = {
            ClaimImportance.HIGH: 3.0,
            ClaimImportance.MEDIUM: 2.0,
            ClaimImportance.LOW: 1.0,
        }

        total_weight = 0.0
        supported_weight = 0.0

        for verdict in relevant_verdicts:
            claim = claim_by_id[verdict.claim_id]
            weight = importance_weights.get(claim.importance, 1.0)
            total_weight += weight
            if verdict.verdict == FaithfulnessVerdictLabel.SUPPORTED:
                supported_weight += weight

        return supported_weight / total_weight if total_weight > 0 else 0.0

    def _calculate_avg_verdict_confidence(self) -> float:
        """Calculate average confidence across all verdicts.

        Only counts verdicts for claims that are in extracted_claims.

        Returns:
            Average confidence as a float between 0.0 and 1.0
        """
        if (
            self.has_failures
            or len(self.extracted_claims) == 0
            or not isinstance(self.faithfulness_result, FaithfulnessJudgeResult)
        ):
            return 0.0

        relevant_verdicts = self._get_relevant_verdicts()

        if len(relevant_verdicts) == 0:
            return 0.0

        return sum(verdict.confidence for verdict in relevant_verdicts) / len(
            relevant_verdicts
        )

    @property
    def evaluation_reasoning_data(self) -> CopilotResponseFaithfulnessReasoningMetadata:
        """Get evaluation reasoning data for metadata reporting."""
        claims_data: List[ClaimWithVerdict] = []

        # Create a mapping from claim_id to verdict for quick lookup
        verdict_by_claim_id = {
            verdict.claim_id: verdict for verdict in self._get_relevant_verdicts()
        }

        # Build the claims data with their verdicts
        for claim in self.extracted_claims.claims:
            claim_with_verdict = ClaimWithVerdict(claim=claim)

            # Add verdict if available
            if verdict := verdict_by_claim_id.get(claim.id):
                claim_with_verdict.verdict = verdict

            claims_data.append(claim_with_verdict)

        return CopilotResponseFaithfulnessReasoningMetadata(
            claims_with_verdicts=claims_data,
        )


class CopilotResponseCompletenessMetrics(BaseModel):
    completeness_rate: float = Field(
        ...,
        description="Completeness rate: covered parts / total parts",
        ge=0.0,
        le=1.0,
    )
    covered_parts_count: int = Field(
        ...,
        description="Number of user request parts that are covered",
        ge=0,
    )
    uncovered_parts_count: int = Field(
        ...,
        description="Number of user request parts that are not covered",
        ge=0,
    )
    total_parts_count: int = Field(
        ...,
        description="Total number of parts in the user request",
        ge=0,
    )
    confidence: float = Field(
        ...,
        description="Overall confidence in the completeness assessment",
        ge=0.0,
        le=1.0,
    )


class CopilotResponseCompletenessEvaluationResult(BaseModel):
    entry_id: str = Field(..., description="ID of the dataset entry being evaluated")
    extracted_claims: Claims = Field(
        ..., description="Claims extracted from the Copilot response."
    )
    completeness_result: Union[CompletenessJudgeResult, EvaluationFailure] = Field(
        ..., description="Result from completeness judge"
    )

    @property
    def has_failures(self) -> bool:
        """Check if completeness evaluation failed.

        Returns:
            True if completeness evaluation failed
        """
        return isinstance(self.completeness_result, EvaluationFailure)

    @property
    def metrics(self) -> CopilotResponseCompletenessMetrics:
        """Calculate the metrics for the completeness evaluation."""
        # Return zero metrics for failures
        if (
            self.has_failures
            or len(self.extracted_claims) == 0
            or not isinstance(self.completeness_result, CompletenessJudgeResult)
        ):
            return CopilotResponseCompletenessMetrics(
                completeness_rate=0.0,
                covered_parts_count=0,
                uncovered_parts_count=0,
                total_parts_count=0,
                confidence=0.0,
            )

        total_parts = len(self.completeness_result.verdicts)
        covered_count = len(self.completeness_result.covered_parts)
        uncovered_count = len(self.completeness_result.uncovered_parts)

        completeness_rate = covered_count / total_parts if total_parts > 0 else 0.0

        return CopilotResponseCompletenessMetrics(
            completeness_rate=completeness_rate,
            covered_parts_count=covered_count,
            uncovered_parts_count=uncovered_count,
            total_parts_count=total_parts,
            confidence=self.completeness_result.confidence,
        )

    @property
    def evaluation_reasoning_data(self) -> CopilotResponseCompletenessReasoningMetadata:
        """Get evaluation reasoning data for metadata reporting."""
        parts_data: List[PartWithClaims] = []

        # Create a mapping from claim_id to claim for quick lookup
        claim_by_id = {claim.id: claim for claim in self.extracted_claims.claims}

        if not self.has_failures and isinstance(
            self.completeness_result, CompletenessJudgeResult
        ):
            # Build the parts data with their addressing claims
            for verdict in self.completeness_result.verdicts:
                addressing_claims = [
                    claim_by_id[claim_id]
                    for claim_id in verdict.addressing_claims_ids
                    if claim_id in claim_by_id
                ]

                parts_data.append(
                    PartWithClaims(part=verdict, addressing_claims=addressing_claims)
                )

        return CopilotResponseCompletenessReasoningMetadata(
            parts_with_claims=parts_data,
        )


class ClaimExtractionStepResult:
    """Encapsulates dataset entries and their claim extraction results.

    This class represents the result of the claim extraction step within the copilot
    response evaluation pipeline. It provides structured access to entries, claims, and
    failure information.

    Note: This is distinct from what ClaimExtractor.extract() returns (a simple list).
    This class combines extraction results with their corresponding dataset entries for
    use in subsequent evaluation steps (e.g. faithfulness and completeness judges).
    """

    def __init__(
        self,
        dataset_entries: List[DatasetEntry],
        claims_or_failures: List[Claims | ClaimExtractionFailure],
    ):
        """Initialize claim extraction data.

        Args:
            dataset_entries: List of all dataset entries
            claims_or_failures: List of claims or extraction failures for each entry
        """
        self._dataset_entries = dataset_entries
        self._claims_or_failures = claims_or_failures
        self._failed_indices = self._get_indices_with_extraction_failures()
        self._successful_entries_data = self._get_successful_entries_data()
        self._original_idx_to_successful_position_map = (
            self._generate_original_idx_to_successful_position_map()
        )

    def _get_indices_with_extraction_failures(self) -> Set[int]:
        """Get indices where claim extraction failed."""
        failed_indices: Set[int] = set()
        for idx, claims_or_failure in enumerate(self._claims_or_failures):
            if isinstance(claims_or_failure, ClaimExtractionFailure):
                failed_indices.add(idx)
        return failed_indices

    def _get_successful_entries_data(
        self,
    ) -> List[tuple[int, DatasetEntry, Claims]]:
        """Get list of successful entries with their original indices."""
        return [
            (idx, entry, claims)
            for idx, (entry, claims) in enumerate(
                zip(self._dataset_entries, self._claims_or_failures)
            )
            if idx not in self._failed_indices and isinstance(claims, Claims)
        ]

    def _generate_original_idx_to_successful_position_map(self) -> Dict[int, int]:
        """Map original dataset indices to positions in successful entries list.

        This map translates between two different indexing systems:
        - Original dataset indices: positions in the full dataset (0, 1, 2, 3, ...)
        - Successful entry positions: positions in the filtered list of successful
          extractions (0, 1, 2, ...).

        The map is needed because judges only process successful claim extractions, so
        their results are in a filtered list. When building final evaluation results, we
        need to map these filtered results back to their original dataset indices.

        Example:
            If original dataset has indices [0, 1, 2, 3] and index 1 failed extraction,
            then:

            successful_entries_data = [
              (0, entry0, claims0),
              (2, entry2, claims2),
              (3, entry3, claims3),
            ]
            original_idx_to_successful_position_map = {0: 0, 2: 1, 3: 2}

            This allows:

            judge_results_list[original_idx_to_successful_position_map[2]]

            to get result for original index 2

        Returns:
            Dictionary mapping original dataset indices to their positions in
            successful_entries_data (which corresponds to judge result list positions).
        """
        return {
            original_idx: result_pos
            for result_pos, (original_idx, _, _) in enumerate(
                self.successful_entries_data
            )
        }

    @property
    def failed_indices(self) -> Set[int]:
        """Get indices where claim extraction failed."""
        return self._failed_indices

    @property
    def successful_entries_data(self) -> List[tuple[int, DatasetEntry, Claims]]:
        """Get list of (original_idx, entry, claims) for successful extractions."""
        return self._successful_entries_data

    @property
    def original_idx_to_successful_position_map(self) -> Dict[int, int]:
        """Map original dataset indices to positions in successful entries list."""
        return self._original_idx_to_successful_position_map

    def get_dataset_entry_by_idx(self, idx: int) -> DatasetEntry:
        """Get dataset entry at the given index."""
        return self._dataset_entries[idx]

    def get_extracted_claims_or_failure_by_idx(
        self, idx: int
    ) -> Claims | ClaimExtractionFailure:
        """Get claims or failure at the given index."""
        return self._claims_or_failures[idx]

    def is_failed(self, idx: int) -> bool:
        """Check if claim extraction failed at the given index."""
        return idx in self._failed_indices

    def __len__(self) -> int:
        """Get total number of entries."""
        return len(self._dataset_entries)
