"""Unit tests for copilot response evaluator models."""

import uuid

import pytest

from rasa.builder.copilot.models import ResponseCategory
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
from rasa.builder.evaluator.copilot_response_evaluator.models import (
    ClaimExtractionStepResult,
    ClaimWithVerdict,
    CopilotResponseCompletenessEvaluationResult,
    CopilotResponseCompletenessReasoningMetadata,
    CopilotResponseFaithfulnessEvaluationResult,
    CopilotResponseFaithfulnessReasoningMetadata,
    PartWithClaims,
)
from rasa.builder.evaluator.dataset.models import (
    DatasetEntry,
    DatasetExpectedOutput,
    DatasetInput,
    DatasetMetadata,
)
from rasa.builder.evaluator.faithfulness_judge.models import (
    ClaimVerdict,
    FaithfulnessJudgeResult,
    FaithfulnessVerdictLabel,
)
from rasa.builder.evaluator.shared.models import EvaluationFailure


class TestCopilotResponseFaithfulnessEvaluationResult:
    @pytest.mark.parametrize(
        "extracted_claims,"
        "faithfulness_result,"
        "expected_support_rate,"
        "expected_weighted_support_rate,"
        "expected_avg_confidence,"
        "expected_supported_count,"
        "expected_contradicted_count,"
        "expected_not_enough_info_count,"
        "expected_extracted_claims_count",
        [
            # Test case 1: All claims supported
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.9,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.8,
                        ),
                    ]
                ),
                1.0,  # support_rate: 2/2 = 1.0
                1.0,  # weighted_support_rate: (3+2)/(3+2) = 1.0
                0.85,  # avg_confidence: (0.9 + 0.8) / 2 = 0.85
                2,  # supported_count
                0,  # contradicted_count
                0,  # not_enough_info_count
                2,  # extracted_claims_count
            ),
            # Test case 2: All claims contradicted
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.CONTRADICTED,
                            rationale="Contradicted",
                            confidence=0.7,
                        ),
                    ]
                ),
                0.0,  # support_rate: 0/1 = 0.0
                0.0,  # weighted_support_rate: 0/3 = 0.0
                0.7,  # avg_confidence: 0.7
                0,  # supported_count
                1,  # contradicted_count
                0,  # not_enough_info_count
                1,  # extracted_claims_count
            ),
            # Test case 3: Mixed verdicts with different importance levels
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="High importance claim",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Medium importance claim",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                            importance=ClaimImportance.LOW,
                            text="Low importance claim",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.95,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            verdict=FaithfulnessVerdictLabel.CONTRADICTED,
                            rationale="Contradicted",
                            confidence=0.75,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                            verdict=FaithfulnessVerdictLabel.NOT_ENOUGH_INFO,
                            rationale="Not enough info",
                            confidence=0.65,
                        ),
                    ]
                ),
                1.0 / 3.0,  # support_rate: 1/3 = 0.333...
                3.0 / 6.0,  # weighted: 3/(3+2+1) = 0.5
                (0.95 + 0.75 + 0.65) / 3.0,  # avg: (0.95+0.75+0.65)/3 ≈ 0.783
                1,  # supported_count
                1,  # contradicted_count
                1,  # not_enough_info_count
                3,  # extracted_claims_count
            ),
            # Test case 4: Partial support with weighted calculation
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="High claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.HIGH,
                            text="High claim 2",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                            importance=ClaimImportance.LOW,
                            text="Low claim",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.9,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.85,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                            verdict=FaithfulnessVerdictLabel.CONTRADICTED,
                            rationale="Contradicted",
                            confidence=0.6,
                        ),
                    ]
                ),
                2.0 / 3.0,  # support_rate: 2/3 ≈ 0.667
                6.0 / 7.0,  # weighted: (3+3)/(3+3+1) = 6/7 ≈ 0.857
                (0.9 + 0.85 + 0.6) / 3.0,  # avg: (0.9+0.85+0.6)/3 ≈ 0.783
                2,  # supported_count
                1,  # contradicted_count
                0,  # not_enough_info_count
                3,  # extracted_claims_count
            ),
            # Test case 5: Empty claims
            (
                Claims(claims=[]),
                FaithfulnessJudgeResult(verdicts=[]),
                0.0,  # support_rate
                0.0,  # weighted_support_rate
                0.0,  # avg_confidence
                0,  # supported_count
                0,  # contradicted_count
                0,  # not_enough_info_count
                0,  # extracted_claims_count
            ),
            # Test case 6: Evaluation failure
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                EvaluationFailure(
                    error_message="Evaluation failed",
                    error_type="EvaluationError",
                ),
                0.0,  # support_rate
                0.0,  # weighted_support_rate
                0.0,  # avg_confidence
                0,  # supported_count
                0,  # contradicted_count
                0,  # not_enough_info_count
                1,  # extracted_claims_count
            ),
            # Test case 7: Hallucinated verdicts (verdicts not in extracted_claims)
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Extracted claim",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.9,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000099"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Hallucinated verdict",
                            confidence=0.8,
                        ),
                    ]
                ),
                1.0,  # support_rate: 1/1 = 1.0 (hallucinated ignored)
                1.0,  # weighted_support_rate: 3/3 = 1.0
                0.9,  # avg_confidence: 0.9 (only extracted claim counted)
                1,  # supported_count (hallucinated not counted)
                0,  # contradicted_count
                0,  # not_enough_info_count
                1,  # extracted_claims_count
            ),
            # Test case 8: Single claim with not enough info
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.NOT_ENOUGH_INFO,
                            rationale="Not enough info",
                            confidence=0.5,
                        ),
                    ]
                ),
                0.0,  # support_rate: 0/1 = 0.0
                0.0,  # weighted_support_rate: 0/2 = 0.0
                0.5,  # avg_confidence: 0.5
                0,  # supported_count
                0,  # contradicted_count
                1,  # not_enough_info_count
                1,  # extracted_claims_count
            ),
            # Test case 9: All supported with different confidence levels
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 2",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 3",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.95,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.85,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.75,
                        ),
                    ]
                ),
                1.0,  # support_rate: 3/3 = 1.0
                1.0,  # weighted_support_rate: 9/9 = 1.0
                (0.95 + 0.85 + 0.75) / 3.0,  # avg_confidence: (0.95+0.85+0.75)/3 = 0.85
                3,  # supported_count
                0,  # contradicted_count
                0,  # not_enough_info_count
                3,  # extracted_claims_count
            ),
        ],
    )
    def test_metrics_calculation(
        self,
        extracted_claims: Claims,
        faithfulness_result: FaithfulnessJudgeResult | EvaluationFailure,
        expected_support_rate: float,
        expected_weighted_support_rate: float,
        expected_avg_confidence: float,
        expected_supported_count: int,
        expected_contradicted_count: int,
        expected_not_enough_info_count: int,
        expected_extracted_claims_count: int,
    ) -> None:
        """Test that metrics are calculated correctly for various scenarios.

        Args:
            extracted_claims: Claims extracted from the response
            faithfulness_result: Result from faithfulness judge (or failure)
            expected_support_rate: Expected micro support rate
            expected_weighted_support_rate: Expected weighted support rate
            expected_avg_confidence: Expected average confidence
            expected_supported_count: Expected count of supported claims
            expected_contradicted_count: Expected count of contradicted claims
            expected_not_enough_info_count: Expected count of not enough info claims
            expected_extracted_claims_count: Expected total extracted claims count
        """
        # Given
        result = CopilotResponseFaithfulnessEvaluationResult(
            entry_id="test-entry",
            extracted_claims=extracted_claims,
            faithfulness_result=faithfulness_result,
        )

        # When
        metrics = result.metrics

        # Then
        assert metrics.support_rate == pytest.approx(
            expected_support_rate, abs=0.001
        ), (
            f"Support rate mismatch: expected {expected_support_rate}, "
            f"got {metrics.support_rate}"
        )
        assert metrics.weighted_support_rate == pytest.approx(
            expected_weighted_support_rate, abs=0.001
        ), (
            f"Weighted support rate mismatch: "
            f"expected {expected_weighted_support_rate}, "
            f"got {metrics.weighted_support_rate}"
        )
        assert metrics.avg_verdict_confidence == pytest.approx(
            expected_avg_confidence, abs=0.001
        ), (
            f"Average confidence mismatch: expected {expected_avg_confidence}, "
            f"got {metrics.avg_verdict_confidence}"
        )
        assert metrics.supported_claims_count == expected_supported_count, (
            f"Supported count mismatch: expected {expected_supported_count}, "
            f"got {metrics.supported_claims_count}"
        )
        assert metrics.contradicted_claims_count == expected_contradicted_count, (
            f"Contradicted count mismatch: expected {expected_contradicted_count}, "
            f"got {metrics.contradicted_claims_count}"
        )
        assert metrics.not_enough_info_claims_count == expected_not_enough_info_count, (
            f"Not enough info count mismatch: "
            f"expected {expected_not_enough_info_count}, "
            f"got {metrics.not_enough_info_claims_count}"
        )
        assert metrics.extracted_claims_count == expected_extracted_claims_count, (
            f"Extracted claims count mismatch: "
            f"expected {expected_extracted_claims_count}, "
            f"got {metrics.extracted_claims_count}"
        )

    @pytest.mark.parametrize(
        "extracted_claims,faithfulness_result,expected_verdict_labels_by_claim",
        [
            # Test case 1: All claims receive verdicts
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.9,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            verdict=FaithfulnessVerdictLabel.CONTRADICTED,
                            rationale="Contradicted",
                            confidence=0.7,
                        ),
                    ]
                ),
                [
                    FaithfulnessVerdictLabel.SUPPORTED,
                    FaithfulnessVerdictLabel.CONTRADICTED,
                ],
            ),
            # Test case 2: Evaluation failure returns claims without verdicts
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 1",
                            metadata={},
                        )
                    ]
                ),
                EvaluationFailure(
                    error_message="Evaluation failed",
                    error_type="EvaluationError",
                ),
                [None],
            ),
            # Test case 3: Claim without verdict remains None
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.LOW,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.NOT_ENOUGH_INFO,
                            rationale="Not enough info",
                            confidence=0.6,
                        ),
                    ]
                ),
                [
                    FaithfulnessVerdictLabel.NOT_ENOUGH_INFO,
                    None,
                ],
            ),
            # Test case 4: Hallucinated verdict is ignored
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                FaithfulnessJudgeResult(
                    verdicts=[
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            verdict=FaithfulnessVerdictLabel.SUPPORTED,
                            rationale="Supported",
                            confidence=0.8,
                        ),
                        ClaimVerdict(
                            claim_id=uuid.UUID("00000000-0000-0000-0000-000000000099"),
                            verdict=FaithfulnessVerdictLabel.CONTRADICTED,
                            rationale="Hallucinated",
                            confidence=0.4,
                        ),
                    ]
                ),
                [FaithfulnessVerdictLabel.SUPPORTED],
            ),
            # Test case 5: No extracted claims
            (
                Claims(claims=[]),
                FaithfulnessJudgeResult(verdicts=[]),
                [],
            ),
        ],
    )
    def test_evaluation_reasoning_data(
        self,
        extracted_claims: Claims,
        faithfulness_result: FaithfulnessJudgeResult | EvaluationFailure,
        expected_verdict_labels_by_claim: list[FaithfulnessVerdictLabel | None],
    ) -> None:
        """Test that evaluation_reasoning_data creates the object correctly.

        Args:
            extracted_claims: Claims extracted from the response
            faithfulness_result: Result from faithfulness judge (or failure)
            expected_verdict_labels_by_claim: Verdict label (or None) per claim
        """
        # Given
        result = CopilotResponseFaithfulnessEvaluationResult(
            entry_id="test-entry",
            extracted_claims=extracted_claims,
            faithfulness_result=faithfulness_result,
        )
        extracted_claims_list = list(extracted_claims.claims)

        # When
        reasoning_data = result.evaluation_reasoning_data

        # Then
        assert isinstance(reasoning_data, CopilotResponseFaithfulnessReasoningMetadata)
        assert len(reasoning_data.claims_with_verdicts) == len(extracted_claims_list)
        assert len(expected_verdict_labels_by_claim) == len(extracted_claims_list)

        for i, expected_label in enumerate(expected_verdict_labels_by_claim):
            claim_with_verdict = reasoning_data.claims_with_verdicts[i]
            assert isinstance(claim_with_verdict, ClaimWithVerdict)
            assert claim_with_verdict.claim == extracted_claims_list[i]

            verdict = claim_with_verdict.verdict
            if expected_label is None:
                assert verdict is None
            else:
                assert verdict is not None
                assert verdict.claim_id == extracted_claims_list[i].id
                assert verdict.verdict == expected_label


class TestCopilotResponseCompletenessEvaluationResult:
    """Tests for CopilotResponseCompletenessEvaluationResult.metrics property."""

    @pytest.mark.parametrize(
        "extracted_claims,"
        "completeness_result,"
        "expected_completeness_rate,"
        "expected_covered_parts_count,"
        "expected_uncovered_parts_count,"
        "expected_total_parts_count,"
        "expected_confidence",
        [
            # Test case 1: All parts covered
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001")
                            ],
                            rationale="Covered by claim 1",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 2",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000002")
                            ],
                            rationale="Covered by claim 2",
                        ),
                    ],
                    overall_rationale="All parts covered",
                    confidence=0.9,
                ),
                1.0,  # completeness_rate: 2/2 = 1.0
                2,  # covered_parts_count
                0,  # uncovered_parts_count
                2,  # total_parts_count
                0.9,  # confidence
            ),
            # Test case 2: All parts uncovered
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[],
                            rationale="Not covered",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 2",
                            addressing_claims_ids=[],
                            rationale="Not covered",
                        ),
                    ],
                    overall_rationale="No parts covered",
                    confidence=0.7,
                ),
                0.0,  # completeness_rate: 0/2 = 0.0
                0,  # covered_parts_count
                2,  # uncovered_parts_count
                2,  # total_parts_count
                0.7,  # confidence
            ),
            # Test case 3: Mixed coverage
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001")
                            ],
                            rationale="Covered",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 2",
                            addressing_claims_ids=[],
                            rationale="Not covered",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 3",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000002")
                            ],
                            rationale="Covered",
                        ),
                    ],
                    overall_rationale="Partial coverage",
                    confidence=0.8,
                ),
                2.0 / 3.0,  # completeness_rate: 2/3 ≈ 0.667
                2,  # covered_parts_count
                1,  # uncovered_parts_count
                3,  # total_parts_count
                0.8,  # confidence
            ),
            # Test case 4: Single part covered
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001")
                            ],
                            rationale="Covered",
                        ),
                    ],
                    overall_rationale="Single part covered",
                    confidence=0.95,
                ),
                1.0,  # completeness_rate: 1/1 = 1.0
                1,  # covered_parts_count
                0,  # uncovered_parts_count
                1,  # total_parts_count
                0.95,  # confidence
            ),
            # Test case 5: Empty claims
            (
                Claims(claims=[]),
                CompletenessJudgeResult(
                    verdicts=[],
                    overall_rationale="No verdicts",
                    confidence=0.0,
                ),
                0.0,  # completeness_rate
                0,  # covered_parts_count
                0,  # uncovered_parts_count
                0,  # total_parts_count
                0.0,  # confidence
            ),
            # Test case 6: Evaluation failure
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                EvaluationFailure(
                    error_message="Evaluation failed",
                    error_type="EvaluationError",
                ),
                0.0,  # completeness_rate
                0,  # covered_parts_count
                0,  # uncovered_parts_count
                0,  # total_parts_count
                0.0,  # confidence
            ),
            # Test case 7: Multiple claims addressing same part
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001"),
                                uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            ],
                            rationale="Covered by multiple claims",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 2",
                            addressing_claims_ids=[],
                            rationale="Not covered",
                        ),
                    ],
                    overall_rationale="Mixed coverage",
                    confidence=0.75,
                ),
                0.5,  # completeness_rate: 1/2 = 0.5
                1,  # covered_parts_count (part 1 is covered)
                1,  # uncovered_parts_count (part 2 is not covered)
                2,  # total_parts_count
                0.75,  # confidence
            ),
            # Test case 8: Empty verdicts list
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[],
                    overall_rationale="No parts identified",
                    confidence=0.5,
                ),
                0.0,  # completeness_rate: 0/0 = 0.0 (handled by division check)
                0,  # covered_parts_count
                0,  # uncovered_parts_count
                0,  # total_parts_count
                0.5,  # confidence
            ),
            # Test case 9: Partial coverage with different confidence
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 2",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 3",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001")
                            ],
                            rationale="Covered",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 2",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000002")
                            ],
                            rationale="Covered",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 3",
                            addressing_claims_ids=[],
                            rationale="Not covered",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 4",
                            addressing_claims_ids=[],
                            rationale="Not covered",
                        ),
                    ],
                    overall_rationale="Partial coverage",
                    confidence=0.65,
                ),
                0.5,  # completeness_rate: 2/4 = 0.5
                2,  # covered_parts_count
                2,  # uncovered_parts_count
                4,  # total_parts_count
                0.65,  # confidence
            ),
        ],
    )
    def test_metrics_calculation(
        self,
        extracted_claims: Claims,
        completeness_result: CompletenessJudgeResult | EvaluationFailure,
        expected_completeness_rate: float,
        expected_covered_parts_count: int,
        expected_uncovered_parts_count: int,
        expected_total_parts_count: int,
        expected_confidence: float,
    ) -> None:
        """Test that metrics are calculated correctly for various scenarios.

        Args:
            extracted_claims: Claims extracted from the response
            completeness_result: Result from completeness judge (or failure)
            expected_completeness_rate: Expected completeness rate
            expected_covered_parts_count: Expected count of covered parts
            expected_uncovered_parts_count: Expected count of uncovered parts
            expected_total_parts_count: Expected total parts count
            expected_confidence: Expected confidence value
        """
        # Given
        result = CopilotResponseCompletenessEvaluationResult(
            entry_id="test-entry",
            extracted_claims=extracted_claims,
            completeness_result=completeness_result,
        )

        # When
        metrics = result.metrics

        # Then
        assert metrics.completeness_rate == pytest.approx(
            expected_completeness_rate, abs=0.001
        ), (
            f"Completeness rate mismatch: expected {expected_completeness_rate}, "
            f"got {metrics.completeness_rate}"
        )
        assert metrics.covered_parts_count == expected_covered_parts_count, (
            f"Covered parts count mismatch: expected {expected_covered_parts_count}, "
            f"got {metrics.covered_parts_count}"
        )
        assert metrics.uncovered_parts_count == expected_uncovered_parts_count, (
            f"Uncovered parts count mismatch: "
            f"expected {expected_uncovered_parts_count}, "
            f"got {metrics.uncovered_parts_count}"
        )
        assert metrics.total_parts_count == expected_total_parts_count, (
            f"Total parts count mismatch: expected {expected_total_parts_count}, "
            f"got {metrics.total_parts_count}"
        )
        assert metrics.confidence == pytest.approx(expected_confidence, abs=0.001), (
            f"Confidence mismatch: expected {expected_confidence}, "
            f"got {metrics.confidence}"
        )

    @pytest.mark.parametrize(
        "extracted_claims,completeness_result,expected_claim_ids_by_part",
        [
            # Test case 1: All parts covered with claims
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001")
                            ],
                            rationale="Covered by claim 1",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 2",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000002")
                            ],
                            rationale="Covered by claim 2",
                        ),
                    ],
                    overall_rationale="All parts covered",
                    confidence=0.9,
                ),
                [
                    [uuid.UUID("00000000-0000-0000-0000-000000000001")],
                    [uuid.UUID("00000000-0000-0000-0000-000000000002")],
                ],
            ),
            # Test case 2: Evaluation failure
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                EvaluationFailure(
                    error_message="Evaluation failed",
                    error_type="EvaluationError",
                ),
                [],
            ),
            # Test case 3: Mixed coverage (some parts covered, some not)
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001")
                            ],
                            rationale="Covered",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 2",
                            addressing_claims_ids=[],
                            rationale="Not covered",
                        ),
                        UserRequestCompletenessVerdict(
                            part_text="Part 3",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000002")
                            ],
                            rationale="Covered",
                        ),
                    ],
                    overall_rationale="Partial coverage",
                    confidence=0.8,
                ),
                [
                    [uuid.UUID("00000000-0000-0000-0000-000000000001")],
                    [],
                    [uuid.UUID("00000000-0000-0000-0000-000000000002")],
                ],
            ),
            # Test case 4: Empty claims
            (
                Claims(claims=[]),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[],
                            rationale="Not covered",
                        ),
                    ],
                    overall_rationale="No claims",
                    confidence=0.5,
                ),
                [[]],
            ),
            # Test case 5: Part addressed by multiple claims
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.MEDIUM,
                            text="Claim 2",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001"),
                                uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            ],
                            rationale="Covered by multiple claims",
                        ),
                    ],
                    overall_rationale="Multiple claims per part",
                    confidence=0.75,
                ),
                [
                    [
                        uuid.UUID("00000000-0000-0000-0000-000000000001"),
                        uuid.UUID("00000000-0000-0000-0000-000000000002"),
                    ]
                ],
            ),
            # Test case 6: Hallucinated claim IDs filtered out
            (
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="Claim 1",
                            metadata={},
                        ),
                    ]
                ),
                CompletenessJudgeResult(
                    verdicts=[
                        UserRequestCompletenessVerdict(
                            part_text="Part 1",
                            addressing_claims_ids=[
                                uuid.UUID("00000000-0000-0000-0000-000000000001"),
                                uuid.UUID("00000000-0000-0000-0000-000000000099"),
                            ],
                            rationale="Covered by claim 1 and hallucinated",
                        ),
                    ],
                    overall_rationale="Hallucinated claim filtered",
                    confidence=0.7,
                ),
                [[uuid.UUID("00000000-0000-0000-0000-000000000001")]],
            ),
        ],
    )
    def test_evaluation_reasoning_data(
        self,
        extracted_claims: Claims,
        completeness_result: CompletenessJudgeResult | EvaluationFailure,
        expected_claim_ids_by_part: list[list[uuid.UUID]],
    ) -> None:
        """Test that evaluation_reasoning_data creates the object correctly.

        Args:
            extracted_claims: Claims extracted from the response
            completeness_result: Result from completeness judge (or failure)
            expected_claim_ids_by_part: Expected claim IDs for each verdict
        """
        # Given
        result = CopilotResponseCompletenessEvaluationResult(
            entry_id="test-entry",
            extracted_claims=extracted_claims,
            completeness_result=completeness_result,
        )
        expected_parts: list[UserRequestCompletenessVerdict] = (
            completeness_result.verdicts
            if isinstance(completeness_result, CompletenessJudgeResult)
            else []
        )

        # When
        reasoning_data = result.evaluation_reasoning_data

        # Then
        assert isinstance(reasoning_data, CopilotResponseCompletenessReasoningMetadata)
        assert len(reasoning_data.parts_with_claims) == len(expected_claim_ids_by_part)

        for i, expected_part in enumerate(expected_parts):
            part_with_claims = reasoning_data.parts_with_claims[i]
            assert isinstance(part_with_claims, PartWithClaims)
            assert part_with_claims.part == expected_part
            addressing_claim_ids = [
                claim.id for claim in part_with_claims.addressing_claims
            ]
            assert addressing_claim_ids == expected_claim_ids_by_part[i]


class TestClaimExtractionStepResult:
    """Tests for ClaimExtractionStepResult."""

    @pytest.fixture
    def sample_dataset_entries(self) -> list[DatasetEntry]:
        """Create sample dataset entries for testing."""
        return [
            DatasetEntry(
                id="entry-0",
                input=DatasetInput(message="Question 0"),
                expected_output=DatasetExpectedOutput(
                    answer="Answer 0",
                    response_category=ResponseCategory.COPILOT,
                    references=[],
                ),
                metadata=DatasetMetadata(),
            ),
            DatasetEntry(
                id="entry-1",
                input=DatasetInput(message="Question 1"),
                expected_output=DatasetExpectedOutput(
                    answer="Answer 1",
                    response_category=ResponseCategory.COPILOT,
                    references=[],
                ),
                metadata=DatasetMetadata(),
            ),
            DatasetEntry(
                id="entry-2",
                input=DatasetInput(message="Question 2"),
                expected_output=DatasetExpectedOutput(
                    answer="Answer 2",
                    response_category=ResponseCategory.COPILOT,
                    references=[],
                ),
                metadata=DatasetMetadata(),
            ),
        ]

    @pytest.fixture
    def sample_claims(self) -> list[Claims]:
        """Create sample claims for testing."""
        return [
            Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                        importance=ClaimImportance.HIGH,
                        text="Claim 0",
                        metadata={},
                    )
                ]
            ),
            Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                        importance=ClaimImportance.MEDIUM,
                        text="Claim 1",
                        metadata={},
                    )
                ]
            ),
            Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                        importance=ClaimImportance.LOW,
                        text="Claim 2",
                        metadata={},
                    )
                ]
            ),
        ]

    @pytest.fixture
    def sample_failures(self) -> list[ClaimExtractionFailure]:
        """Create sample extraction failures for testing."""
        return [
            ClaimExtractionFailure(error_message="Error 0", error_type="TestError"),
            ClaimExtractionFailure(error_message="Error 1", error_type="TestError"),
        ]

    @pytest.mark.parametrize(
        "pattern,expected_failed_indices,expected_successful_count,expected_map",
        [
            # Test case 1: All successful (C=claim, F=failure)
            ("CC", set(), 2, {0: 0, 1: 1}),
            # Test case 2: All failed
            ("FF", {0, 1}, 0, {}),
            # Test case 3: Mixed success and failure
            ("CFC", {1}, 2, {0: 0, 2: 1}),
        ],
    )
    def test_extraction_scenarios(
        self,
        sample_dataset_entries: list[DatasetEntry],
        sample_claims: list[Claims],
        sample_failures: list[ClaimExtractionFailure],
        pattern: str,
        expected_failed_indices: set[int],
        expected_successful_count: int,
        expected_map: dict[int, int],
    ) -> None:
        """Test ClaimExtractionStepResult with various extraction scenarios."""
        # Given
        dataset_entries = sample_dataset_entries[: len(pattern)]
        claims_or_failures_list: list[Claims | ClaimExtractionFailure] = []

        for i, char in enumerate(pattern):
            if char == "C":
                claims_or_failures_list.append(sample_claims[i])
            elif char == "F":
                claims_or_failures_list.append(sample_failures[i])

        # When
        result = ClaimExtractionStepResult(dataset_entries, claims_or_failures_list)

        # Then
        assert result.failed_indices == expected_failed_indices
        assert len(result.successful_entries_data) == expected_successful_count
        assert result.original_idx_to_successful_position_map == expected_map
        assert len(result) == len(dataset_entries)

    def test_empty_lists(self) -> None:
        """Test ClaimExtractionStepResult with empty input lists."""
        # Given
        dataset_entries: list[DatasetEntry] = []
        claims_or_failures: list[Claims | ClaimExtractionFailure] = []

        # When
        result = ClaimExtractionStepResult(dataset_entries, claims_or_failures)

        # Then
        assert result.failed_indices == set()
        assert len(result.successful_entries_data) == 0
        assert result.original_idx_to_successful_position_map == {}
        assert len(result) == 0

    def test_get_dataset_entry_by_idx(
        self, sample_dataset_entries: list[DatasetEntry], sample_claims: list[Claims]
    ) -> None:
        """Test the get_dataset_entry_by_idx method."""
        # Given
        dataset_entries = sample_dataset_entries[:2]
        claims_or_failures = sample_claims[:2]
        result = ClaimExtractionStepResult(dataset_entries, claims_or_failures)

        # Then
        for idx, expected_entry in enumerate(dataset_entries):
            assert result.get_dataset_entry_by_idx(idx) == expected_entry

    def test_get_extracted_claims_or_failure_by_idx(
        self,
        sample_dataset_entries: list[DatasetEntry],
        sample_claims: list[Claims],
        sample_failures: list[ClaimExtractionFailure],
    ) -> None:
        """Test the get_extracted_claims_or_failure_by_idx method."""
        # Given
        dataset_entries = sample_dataset_entries[:2]
        claims_or_failures = [sample_claims[0], sample_failures[0]]
        result = ClaimExtractionStepResult(dataset_entries, claims_or_failures)

        # Then
        assert result.get_extracted_claims_or_failure_by_idx(0) == sample_claims[0]
        assert result.get_extracted_claims_or_failure_by_idx(1) == sample_failures[0]

    def test_successful_entries_data_structure(
        self,
        sample_dataset_entries: list[DatasetEntry],
        sample_claims: list[Claims],
        sample_failures: list[ClaimExtractionFailure],
    ) -> None:
        """Test the structure of successful_entries_data."""
        # Given
        dataset_entries = sample_dataset_entries
        claims_or_failures = [
            sample_claims[0],
            sample_failures[0],
            sample_claims[2],
        ]
        result = ClaimExtractionStepResult(dataset_entries, claims_or_failures)

        # Then
        assert len(result.successful_entries_data) == 2

        # Verify first successful entry
        original_idx_0, entry_0, claims_0 = result.successful_entries_data[0]
        assert original_idx_0 == 0
        assert entry_0 == dataset_entries[0]
        assert claims_0 == sample_claims[0]
        assert isinstance(claims_0, Claims)

        # Verify second successful entry
        original_idx_2, entry_2, claims_2 = result.successful_entries_data[1]
        assert original_idx_2 == 2
        assert entry_2 == dataset_entries[2]
        assert claims_2 == sample_claims[2]
        assert isinstance(claims_2, Claims)

        # Verify mapping
        assert result.original_idx_to_successful_position_map[0] == 0
        assert result.original_idx_to_successful_position_map[2] == 1
