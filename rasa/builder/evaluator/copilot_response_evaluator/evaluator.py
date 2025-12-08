"""Copilot Response Evaluator for evaluating response quality.

This module provides functionality to evaluate Copilot responses on two dimensions:
- Faithfulness: Are the claims in the response grounded in evidence?
- Completeness: Does the response fully address all parts of the user's request?

The evaluator orchestrates the complete pipeline including claim extraction.
"""

import asyncio
from typing import List

import structlog

from rasa.builder.evaluator import config
from rasa.builder.evaluator.completeness_judge.judge import CompletenessJudge
from rasa.builder.evaluator.completeness_judge.models import (
    CompletenessJudgeInput,
    CompletenessJudgeResult,
)
from rasa.builder.evaluator.content_processors.claim_extractor import ClaimExtractor
from rasa.builder.evaluator.content_processors.models import (
    ClaimExtractionFailure,
    Claims,
)
from rasa.builder.evaluator.copilot_response_evaluator.models import (
    ClaimExtractionStepResult,
    CopilotResponseCompletenessEvaluationResult,
    CopilotResponseFaithfulnessEvaluationResult,
)
from rasa.builder.evaluator.dataset.models import DatasetEntry
from rasa.builder.evaluator.faithfulness_judge.judge import FaithfulnessJudge
from rasa.builder.evaluator.faithfulness_judge.models import (
    FaithfulnessJudgeInput,
    FaithfulnessJudgeResult,
)
from rasa.builder.evaluator.shared.models import EvaluationFailure

structlogger = structlog.get_logger()


class CopilotResponseEvaluator:
    """Evaluates Copilot responses on faithfulness and completeness.

    This evaluator represents the complete evaluation pipeline:
    1. ClaimExtractor: Extracts atomic claims from Copilot responses. The extractor
       processes all responses concurrently (up to max_concurrent_extractions).
    2. FaithfulnessJudge: Evaluates whether claims are grounded in evidence.
    3. CompletenessJudge: Evaluates whether the response fully addresses the user
       request

    The judges for faithfulness and completeness run concurrently (up to
    max_concurrent_evaluations/2 each).

    All steps are orchestrated to process multiple entries efficiently.
    """

    def __init__(
        self,
        claim_extractor: ClaimExtractor | None = None,
        faithfulness_judge: FaithfulnessJudge | None = None,
        completeness_judge: CompletenessJudge | None = None,
    ):
        """Initialize the copilot response evaluator.

        Args:
            claim_extractor: Optional claim extractor. If not provided, creates a
                default instance.
            faithfulness_judge: Optional faithfulness judge. If not provided, creates a
                default instance with configured parameters.
            completeness_judge: Optional completeness judge. If not provided, creates a
                default instance with configured parameters.
        """
        self._claim_extractor = claim_extractor or ClaimExtractor(
            model=config.CLAIM_EXTRACTOR_MODEL,
            temperature=config.CLAIM_EXTRACTOR_TEMPERATURE,
            max_tokens=config.CLAIM_EXTRACTOR_MAX_TOKENS,
            timeout=config.CLAIM_EXTRACTOR_TIMEOUT,
            max_concurrent_extractions=config.CLAIM_EXTRACTOR_MAX_CONCURRENT_EXTRACTIONS,
        )
        self._faithfulness_judge = faithfulness_judge or FaithfulnessJudge(
            model=config.FAITHFULNESS_JUDGE_MODEL,
            temperature=config.FAITHFULNESS_JUDGE_TEMPERATURE,
            timeout=config.FAITHFULNESS_JUDGE_TIMEOUT,
            max_tokens=config.FAITHFULNESS_JUDGE_MAX_TOKENS,
            max_concurrent_evaluations=config.FAITHFULNESS_JUDGE_MAX_CONCURRENT_EVALUATIONS,
        )
        self._completeness_judge = completeness_judge or CompletenessJudge(
            model=config.COMPLETENESS_JUDGE_MODEL,
            temperature=config.COMPLETENESS_JUDGE_TEMPERATURE,
            timeout=config.COMPLETENESS_JUDGE_TIMEOUT,
            max_tokens=config.COMPLETENESS_JUDGE_MAX_TOKENS,
            max_concurrent_evaluations=config.COMPLETENESS_JUDGE_MAX_CONCURRENT_EVALUATIONS,
        )

    async def evaluate(
        self, dataset_entries: List[DatasetEntry]
    ) -> tuple[
        List[CopilotResponseFaithfulnessEvaluationResult],
        List[CopilotResponseCompletenessEvaluationResult],
    ]:
        """Evaluate multiple copilot responses from dataset entries.

        This method orchestrates the complete evaluation pipeline:
        1. Extracts evidence from dataset entries
        2. Extracts claims from copilot responses (batched)
        3. Runs faithfulness and completeness judges (batched, concurrently)

        Args:
            dataset_entries: List of dataset entries containing:
            - User message/input
            - Output (copilot response)
            - Metadata with evidence (documents, code, chat history)

        Returns:
            Tuple of (faithfulness_results, completeness_results), each containing
            one result per entry in the same order as dataset_entries.
        """
        structlogger.info(
            "evaluator.copilot_response_evaluator.evaluate.start",
            total_entries=len(dataset_entries),
        )

        # Step 1: Extract claims from all copilot responses
        # The extractor processes all responses concurrently (up to
        # max_concurrent_extractions) using asyncio.gather internally with
        # semaphore-based rate limiting.
        structlogger.info(
            "evaluator.copilot_response_evaluator.evaluate.extract_claims.start",
        )

        copilot_responses = [entry.expected_output.answer for entry in dataset_entries]
        claims_or_failures = await self._claim_extractor.extract(copilot_responses)
        claim_extraction_result = ClaimExtractionStepResult(
            dataset_entries, claims_or_failures
        )

        structlogger.info(
            "evaluator.copilot_response_evaluator.extract_claims.complete",
            total_entries=len(dataset_entries),
            successful_extractions=sum(
                1 for c in claims_or_failures if isinstance(c, Claims)
            ),
            failed_extractions=sum(
                1 for c in claims_or_failures if isinstance(c, ClaimExtractionFailure)
            ),
        )

        # Step 2: Run both judges concurrently on all dataset entries where claim
        # extraction succeeded in a batched manner.
        faithfulness_results_list, completeness_results_list = await asyncio.gather(
            self._create_faithfulness_evaluation_task(
                claim_extraction_result.successful_entries_data
            ),
            self._create_completeness_evaluation_task(
                claim_extraction_result.successful_entries_data
            ),
        )

        # Step 3: Build result lists, mapping judge results back to original indices
        # and inserting failures for entries with failed claim extraction
        faithfulness_results = self._build_faithfulness_results(
            claim_extraction_result,
            faithfulness_results_list,
        )
        completeness_results = self._build_completeness_results(
            claim_extraction_result,
            completeness_results_list,
        )

        structlogger.info(
            "evaluator.copilot_response_evaluator.judge_evaluations.complete",
            total_entries=len(dataset_entries),
            faithfulness_failures=sum(
                1 for r in faithfulness_results if r.has_failures
            ),
            completeness_failures=sum(
                1 for r in completeness_results if r.has_failures
            ),
        )

        return faithfulness_results, completeness_results

    def _create_faithfulness_evaluation_task(
        self,
        successful_entries_data: List[tuple[int, DatasetEntry, Claims]],
    ) -> asyncio.Task[List[FaithfulnessJudgeResult | EvaluationFailure]]:
        """Create asyncio task for faithfulness evaluation.

        Args:
            successful_entries_data: List of (original_idx, entry, claims) tuples
                for entries where extraction succeeded

        Returns:
            Async task that evaluates faithfulness for all successful entries
        """
        faithfulness_inputs = [
            FaithfulnessJudgeInput.from_dataset_entry_and_claims(entry, claims)
            for _, entry, claims in successful_entries_data
        ]
        return asyncio.create_task(
            self._faithfulness_judge.evaluate(faithfulness_inputs)
        )

    def _create_completeness_evaluation_task(
        self,
        successful_entries_data: List[tuple[int, DatasetEntry, Claims]],
    ) -> asyncio.Task[List[CompletenessJudgeResult | EvaluationFailure]]:
        """Create asyncio task for completeness evaluation.

        Args:
            successful_entries_data: List of (original_idx, entry, claims) tuples
                for entries where extraction succeeded

        Returns:
            Async task that evaluates completeness for all successful entries
        """
        completeness_inputs = [
            CompletenessJudgeInput.from_dataset_entry_and_claims(entry, claims)
            for _, entry, claims in successful_entries_data
        ]
        return asyncio.create_task(
            self._completeness_judge.evaluate(completeness_inputs)
        )

    def _build_faithfulness_results(
        self,
        claim_extraction_result: ClaimExtractionStepResult,
        faithfulness_results_list: List[FaithfulnessJudgeResult | EvaluationFailure],
    ) -> List[CopilotResponseFaithfulnessEvaluationResult]:
        """Build faithfulness results, mapping judge results back to original indices.

        Args:
            claim_extraction_result: Result from claim extraction step
            faithfulness_results_list: Results from faithfulness judge

        Returns:
            List of faithfulness results, one per entry, in same order as
            dataset_entries
        """
        results: List[CopilotResponseFaithfulnessEvaluationResult] = []
        for idx in range(len(claim_extraction_result)):
            entry = claim_extraction_result.get_dataset_entry_by_idx(idx)
            entry_claims_or_failure = (
                claim_extraction_result.get_extracted_claims_or_failure_by_idx(idx)
            )

            if isinstance(entry_claims_or_failure, ClaimExtractionFailure):
                results.append(
                    CopilotResponseFaithfulnessEvaluationResult(
                        entry_id=entry.id,
                        extracted_claims=Claims(claims=[]),
                        faithfulness_result=EvaluationFailure(
                            error_message=(
                                f"Claim extraction failed: "
                                f"{entry_claims_or_failure.error_message}"
                            ),
                            error_type=entry_claims_or_failure.error_type,
                        ),
                    )
                )

            else:
                faithfulness_result = faithfulness_results_list[
                    claim_extraction_result.original_idx_to_successful_position_map[idx]
                ]
                results.append(
                    CopilotResponseFaithfulnessEvaluationResult(
                        entry_id=entry.id,
                        extracted_claims=entry_claims_or_failure,
                        faithfulness_result=faithfulness_result,
                    )
                )

        return results

    def _build_completeness_results(
        self,
        claim_extraction_result: ClaimExtractionStepResult,
        completeness_results_list: List[CompletenessJudgeResult | EvaluationFailure],
    ) -> List[CopilotResponseCompletenessEvaluationResult]:
        """Build completeness results, mapping judge results back to original indices.

        Args:
            claim_extraction_result: Result from claim extraction step
            completeness_results_list: Results from completeness judge

        Returns:
            List of completeness results, one per entry, in same order as
            dataset_entries
        """
        results: List[CopilotResponseCompletenessEvaluationResult] = []
        for idx in range(len(claim_extraction_result)):
            entry = claim_extraction_result.get_dataset_entry_by_idx(idx)
            entry_claims_or_failure = (
                claim_extraction_result.get_extracted_claims_or_failure_by_idx(idx)
            )
            if isinstance(entry_claims_or_failure, ClaimExtractionFailure):
                results.append(
                    CopilotResponseCompletenessEvaluationResult(
                        entry_id=entry.id,
                        extracted_claims=Claims(claims=[]),
                        completeness_result=EvaluationFailure(
                            error_message=(
                                f"Claim extraction failed: "
                                f"{entry_claims_or_failure.error_message}"
                            ),
                            error_type=entry_claims_or_failure.error_type,
                        ),
                    )
                )
            else:
                completeness_result = completeness_results_list[
                    claim_extraction_result.original_idx_to_successful_position_map[idx]
                ]
                results.append(
                    CopilotResponseCompletenessEvaluationResult(
                        entry_id=entry.id,
                        extracted_claims=entry_claims_or_failure,
                        completeness_result=completeness_result,
                    )
                )

        return results
