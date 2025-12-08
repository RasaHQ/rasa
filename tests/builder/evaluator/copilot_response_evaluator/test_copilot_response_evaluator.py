"""Unit tests for CopilotResponseEvaluator."""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, patch

import openai
import pytest

from rasa.builder.copilot.models import ReferenceEntry, ResponseCategory
from rasa.builder.document_retrieval.models import Document
from rasa.builder.evaluator.completeness_judge.judge import CompletenessJudge
from rasa.builder.evaluator.completeness_judge.models import (
    CompletenessJudgeResult,
    UserRequestCompletenessVerdict,
)
from rasa.builder.evaluator.content_processors.claim_extractor import ClaimExtractor
from rasa.builder.evaluator.content_processors.models import (
    Claim,
    ClaimImportance,
    Claims,
)
from rasa.builder.evaluator.copilot_response_evaluator.evaluator import (
    CopilotResponseEvaluator,
)
from rasa.builder.evaluator.dataset.models import (
    DatasetEntry,
    DatasetExpectedOutput,
    DatasetInput,
    DatasetMetadata,
    DatasetMetadataCopilotAdditionalContext,
)
from rasa.builder.evaluator.faithfulness_judge.judge import FaithfulnessJudge
from rasa.builder.evaluator.faithfulness_judge.models import (
    ClaimVerdict,
    FaithfulnessJudgeResult,
    FaithfulnessVerdictLabel,
)


@pytest.fixture(autouse=True)
def mock_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock OPENAI_API_KEY environment variable for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "copilot-response-evaluator-test-api-key")


class TestCopilotResponseEvaluator:
    """Tests for CopilotResponseEvaluator."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dataset_entries,"
        "claim_extraction_responses,"
        "faithfulness_responses,"
        "completeness_responses,"
        "expected_faithfulness_failures,"
        "expected_completeness_failures",
        [
            # Test case 1: Single entry - all succeed
            (
                [
                    DatasetEntry(
                        id="entry-1",
                        input=DatasetInput(message="How do I configure X?"),
                        expected_output=DatasetExpectedOutput(
                            answer="You can configure X in the config file.",
                            response_category=ResponseCategory.COPILOT,
                            references=[
                                ReferenceEntry(
                                    index=0,
                                    title="Test Doc",
                                    url="https://example.com/doc",
                                )
                            ],
                        ),
                        metadata=DatasetMetadata(
                            copilot_additional_context=DatasetMetadataCopilotAdditionalContext(
                                relevant_documents=[
                                    Document(
                                        url="https://example.com/doc",
                                        title="Test Doc",
                                        content="Test content",
                                    )
                                ],
                            )
                        ),
                    )
                ],
                [
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                                importance=ClaimImportance.HIGH,
                                text="You can configure X in the config file",
                                metadata={},
                            )
                        ]
                    ).model_dump_json()
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000001"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Claim is supported by evidence",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json()
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="How to configure X",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000001")
                                ],
                                rationale="Question fully addressed",
                            )
                        ],
                        overall_rationale="Complete response",
                        confidence=0.9,
                    ).model_dump_json()
                ],
                0,
                0,
            ),
            # Test case 2: Multiple entries - all succeed
            (
                [
                    DatasetEntry(
                        id=f"entry-{i}",
                        input=DatasetInput(message=f"Question {i}"),
                        expected_output=DatasetExpectedOutput(
                            answer=f"Answer {i}",
                            response_category=ResponseCategory.COPILOT,
                            references=[],
                        ),
                        metadata=DatasetMetadata(),
                    )
                    for i in range(3)
                ],
                [
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}"),
                                importance=ClaimImportance.HIGH,
                                text=f"Answer {i}",
                                metadata={},
                            )
                        ]
                    ).model_dump_json()
                    for i in range(3)
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    f"00000000-0000-0000-0000-00000000000{i}"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json()
                    for i in range(3)
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text=f"Question {i}",
                                addressing_claims_ids=[
                                    uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json()
                    for i in range(3)
                ],
                0,
                0,
            ),
            # Test case 3: Claim extraction fails for all entries
            (
                [
                    DatasetEntry(
                        id=f"entry-{i}",
                        input=DatasetInput(message=f"Question {i}"),
                        expected_output=DatasetExpectedOutput(
                            answer=f"Answer {i}",
                            response_category=ResponseCategory.COPILOT,
                            references=[],
                        ),
                        metadata=DatasetMetadata(),
                    )
                    for i in range(2)
                ],
                [
                    openai.OpenAIError("Extraction failed"),
                    openai.OpenAIError("Extraction failed"),
                ],
                [],
                [],
                2,
                2,
            ),
            # Test case 4: Faithfulness judge fails for some entries
            (
                [
                    DatasetEntry(
                        id=f"entry-{i}",
                        input=DatasetInput(message=f"Question {i}"),
                        expected_output=DatasetExpectedOutput(
                            answer=f"Answer {i}",
                            response_category=ResponseCategory.COPILOT,
                            references=[],
                        ),
                        metadata=DatasetMetadata(),
                    )
                    for i in range(3)
                ],
                [
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}"),
                                importance=ClaimImportance.HIGH,
                                text=f"Answer {i}",
                                metadata={},
                            )
                        ]
                    ).model_dump_json()
                    for i in range(3)
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000000"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                    openai.OpenAIError("Faithfulness failed"),
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000002"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text=f"Question {i}",
                                addressing_claims_ids=[
                                    uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json()
                    for i in range(3)
                ],
                1,
                0,
            ),
            # Test case 5: Completeness judge fails for some entries
            (
                [
                    DatasetEntry(
                        id=f"entry-{i}",
                        input=DatasetInput(message=f"Question {i}"),
                        expected_output=DatasetExpectedOutput(
                            answer=f"Answer {i}",
                            response_category=ResponseCategory.COPILOT,
                            references=[],
                        ),
                        metadata=DatasetMetadata(),
                    )
                    for i in range(3)
                ],
                [
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}"),
                                importance=ClaimImportance.HIGH,
                                text=f"Answer {i}",
                                metadata={},
                            )
                        ]
                    ).model_dump_json()
                    for i in range(3)
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    f"00000000-0000-0000-0000-00000000000{i}"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json()
                    for i in range(3)
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Question 0",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000000")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json(),
                    ValueError("Completeness failed"),
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Question 2",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000002")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json(),
                ],
                0,
                1,
            ),
            # Test case 6: Complex scattered failures - test order preservation
            # Pattern: FAIL, FAIL, EVAL, EVAL, EVAL, FAIL, EVAL, EVAL, FAIL, FAIL
            (
                [
                    DatasetEntry(
                        id=f"entry-{i}",
                        input=DatasetInput(message=f"Question {i}"),
                        expected_output=DatasetExpectedOutput(
                            answer=f"Answer {i}",
                            response_category=ResponseCategory.COPILOT,
                            references=[],
                        ),
                        metadata=DatasetMetadata(),
                    )
                    for i in range(10)
                ],
                # Indices 0, 1, 5, 8, 9 fail extraction
                [
                    openai.OpenAIError("Extraction failed 0"),
                    openai.OpenAIError("Extraction failed 1"),
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                                importance=ClaimImportance.HIGH,
                                text="Answer 2",
                                metadata={},
                            )
                        ]
                    ).model_dump_json(),
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                                importance=ClaimImportance.HIGH,
                                text="Answer 3",
                                metadata={},
                            )
                        ]
                    ).model_dump_json(),
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID("00000000-0000-0000-0000-000000000004"),
                                importance=ClaimImportance.HIGH,
                                text="Answer 4",
                                metadata={},
                            )
                        ]
                    ).model_dump_json(),
                    openai.OpenAIError("Extraction failed 5"),
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID("00000000-0000-0000-0000-000000000006"),
                                importance=ClaimImportance.HIGH,
                                text="Answer 6",
                                metadata={},
                            )
                        ]
                    ).model_dump_json(),
                    Claims(
                        claims=[
                            Claim(
                                id=uuid.UUID("00000000-0000-0000-0000-000000000007"),
                                importance=ClaimImportance.HIGH,
                                text="Answer 7",
                                metadata={},
                            )
                        ]
                    ).model_dump_json(),
                    openai.OpenAIError("Extraction failed 8"),
                    openai.OpenAIError("Extraction failed 9"),
                ],
                # Only 5 successful extractions: indices 2, 3, 4, 6, 7
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000002"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000003"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000004"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000006"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000007"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Question 2",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000002")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json(),
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Question 3",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000003")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json(),
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Question 4",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000004")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json(),
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Question 6",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000006")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json(),
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Question 7",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000007")
                                ],
                                rationale="Addressed",
                            )
                        ],
                        overall_rationale="Complete",
                        confidence=0.9,
                    ).model_dump_json(),
                ],
                5,  # expected_faithfulness_failures (all claim extraction failures)
                5,  # expected_completeness_failures (all claim extraction failures)
            ),
        ],
    )
    @patch.object(ClaimExtractor, "_call_llm", new_callable=AsyncMock)
    @patch.object(FaithfulnessJudge, "_call_llm", new_callable=AsyncMock)
    @patch.object(CompletenessJudge, "_call_llm", new_callable=AsyncMock)
    async def test_evaluate(
        self,
        mock_completeness_llm: AsyncMock,
        mock_faithfulness_llm: AsyncMock,
        mock_claim_extractor_llm: AsyncMock,
        dataset_entries: list[DatasetEntry],
        claim_extraction_responses: list[str],
        faithfulness_responses: list[str],
        completeness_responses: list[str],
        expected_faithfulness_failures: int,
        expected_completeness_failures: int,
    ) -> None:
        """Test evaluation with various success and failure scenarios."""
        # Given
        mock_claim_extractor_llm.side_effect = claim_extraction_responses
        mock_faithfulness_llm.side_effect = faithfulness_responses
        mock_completeness_llm.side_effect = completeness_responses

        evaluator = CopilotResponseEvaluator()

        # When
        faithfulness_results, completeness_results = await evaluator.evaluate(
            dataset_entries
        )

        # Then
        # Both result lists should have same length as input entries
        assert len(faithfulness_results) == len(dataset_entries)
        assert len(completeness_results) == len(dataset_entries)

        # Count failures independently for each dimension
        faithfulness_failed = [
            result for result in faithfulness_results if result.has_failures
        ]
        completeness_failed = [
            result for result in completeness_results if result.has_failures
        ]

        # Verify failure counts match expectations
        assert len(faithfulness_failed) == expected_faithfulness_failures
        assert len(completeness_failed) == expected_completeness_failures
        assert mock_claim_extractor_llm.call_count == len(dataset_entries)

    @pytest.mark.asyncio
    @patch.object(ClaimExtractor, "_call_llm", new_callable=AsyncMock)
    @patch.object(FaithfulnessJudge, "_call_llm", new_callable=AsyncMock)
    @patch.object(CompletenessJudge, "_call_llm", new_callable=AsyncMock)
    async def test_evaluate_preserves_order_with_scattered_failures(
        self,
        mock_completeness_llm: AsyncMock,
        mock_faithfulness_llm: AsyncMock,
        mock_claim_extractor_llm: AsyncMock,
    ) -> None:
        """Test that result order matches entry order with scattered failures.

        This test verifies that when claim extraction fails at various positions
        (beginning, middle, end), the results are still returned in the same order
        as the input entries, with proper entry_id linking.

        Pattern:
        FAIL, FAIL, SUCCESS, SUCCESS, SUCCESS, FAIL, SUCCESS, SUCCESS, FAIL, FAIL
        """
        # Given - 10 entries with failures at indices 0, 1, 5, 8, 9
        failed_indices = {0, 1, 5, 8, 9}
        dataset_entries = [
            DatasetEntry(
                id=f"entry-{i}",
                input=DatasetInput(message=f"Question {i}"),
                expected_output=DatasetExpectedOutput(
                    answer=f"Answer {i}",
                    response_category=ResponseCategory.COPILOT,
                    references=[],
                ),
                metadata=DatasetMetadata(),
            )
            for i in range(10)
        ]

        # Mock claim extraction: failures at indices 0, 1, 5, 8, 9
        mock_claim_extractor_llm.side_effect = [
            openai.OpenAIError("Extraction failed 0"),
            openai.OpenAIError("Extraction failed 1"),
            Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                        importance=ClaimImportance.HIGH,
                        text="Answer 2",
                        metadata={},
                    )
                ]
            ).model_dump_json(),
            Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                        importance=ClaimImportance.HIGH,
                        text="Answer 3",
                        metadata={},
                    )
                ]
            ).model_dump_json(),
            Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000004"),
                        importance=ClaimImportance.HIGH,
                        text="Answer 4",
                        metadata={},
                    )
                ]
            ).model_dump_json(),
            openai.OpenAIError("Extraction failed 5"),
            Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000006"),
                        importance=ClaimImportance.HIGH,
                        text="Answer 6",
                        metadata={},
                    )
                ]
            ).model_dump_json(),
            Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000007"),
                        importance=ClaimImportance.HIGH,
                        text="Answer 7",
                        metadata={},
                    )
                ]
            ).model_dump_json(),
            openai.OpenAIError("Extraction failed 8"),
            openai.OpenAIError("Extraction failed 9"),
        ]

        # Mock judge responses for successful extractions (indices 2, 3, 4, 6, 7)
        mock_faithfulness_llm.side_effect = [
            FaithfulnessJudgeResult(
                verdicts=[
                    ClaimVerdict(
                        claim_id=uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}"),
                        verdict=FaithfulnessVerdictLabel.SUPPORTED,
                        rationale=f"Supported {i}",
                        confidence=0.9,
                    )
                ]
            ).model_dump_json()
            for i in [2, 3, 4, 6, 7]
        ]

        mock_completeness_llm.side_effect = [
            CompletenessJudgeResult(
                verdicts=[
                    UserRequestCompletenessVerdict(
                        part_text=f"Question {i}",
                        addressing_claims_ids=[
                            uuid.UUID(f"00000000-0000-0000-0000-00000000000{i}")
                        ],
                        rationale=f"Addressed {i}",
                    )
                ],
                overall_rationale=f"Complete {i}",
                confidence=0.9,
            ).model_dump_json()
            for i in [2, 3, 4, 6, 7]
        ]

        evaluator = CopilotResponseEvaluator()

        # When
        faithfulness_results, completeness_results = await evaluator.evaluate(
            dataset_entries
        )

        # Then - verify we have 10 results (one per entry) for each result type
        assert len(faithfulness_results) == 10
        assert len(completeness_results) == 10

        # Verify order is preserved and entry_ids match
        for i in range(10):
            assert faithfulness_results[i].entry_id == f"entry-{i}", (
                f"Faithfulness result at index {i} should have entry_id 'entry-{i}', "
                f"but got '{faithfulness_results[i].entry_id}'"
            )
            assert completeness_results[i].entry_id == f"entry-{i}", (
                f"Completeness result at index {i} should have entry_id 'entry-{i}', "
                f"but got '{completeness_results[i].entry_id}'"
            )

        # Verify failures at expected indices (0, 1, 5, 8, 9)
        for i in range(10):
            if i in failed_indices:
                assert faithfulness_results[
                    i
                ].has_failures, f"Entry {i} should have faithfulness failures (claim extraction failed)"  # noqa: E501
                assert completeness_results[
                    i
                ].has_failures, f"Entry {i} should have completeness failures (claim extraction failed)"  # noqa: E501
            else:
                assert not faithfulness_results[
                    i
                ].has_failures, f"Entry {i} should not have faithfulness failures (claim extraction succeeded)"  # noqa: E501
                assert not completeness_results[
                    i
                ].has_failures, f"Entry {i} should not have completeness failures (claim extraction succeeded)"  # noqa: E501

    @pytest.mark.asyncio
    @patch.object(ClaimExtractor, "_call_llm", new_callable=AsyncMock)
    @patch.object(FaithfulnessJudge, "_call_llm", new_callable=AsyncMock)
    @patch.object(CompletenessJudge, "_call_llm", new_callable=AsyncMock)
    async def test_evaluate_concurrent_execution(
        self,
        mock_completeness_llm: AsyncMock,
        mock_faithfulness_llm: AsyncMock,
        mock_claim_extractor_llm: AsyncMock,
    ) -> None:
        """Test that evaluations run concurrently, not sequentially."""
        # Given
        evaluator = CopilotResponseEvaluator(
            claim_extractor=ClaimExtractor(
                model="test-gpt",
                temperature=0.0,
                max_tokens=2000,
                timeout=100,
                max_concurrent_extractions=10,
            ),
            faithfulness_judge=FaithfulnessJudge(
                model="test-gpt",
                temperature=0.0,
                max_tokens=4000,
                timeout=120,
                max_concurrent_evaluations=5,
            ),
            completeness_judge=CompletenessJudge(
                model="test-gpt",
                temperature=0.0,
                max_tokens=4000,
                timeout=120,
                max_concurrent_evaluations=5,
            ),
        )

        dataset_entries = [
            DatasetEntry(
                id=f"entry-{i}",
                input=DatasetInput(message=f"Question {i}"),
                expected_output=DatasetExpectedOutput(
                    answer=f"Answer {i}",
                    response_category=ResponseCategory.COPILOT,
                    references=[],
                ),
                metadata=DatasetMetadata(),
            )
            for i in range(5)
        ]

        claim_call_times = []

        async def slow_claim_extraction(prompt: str, client: openai.AsyncOpenAI) -> str:
            """Simulate a slow LLM call."""
            claim_call_times.append(time.time())
            await asyncio.sleep(0.1)  # 100ms delay
            return Claims(
                claims=[
                    Claim(
                        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                        importance=ClaimImportance.HIGH,
                        text="Test claim",
                        metadata={},
                    )
                ]
            ).model_dump_json()

        async def slow_faithfulness_call(
            prompt: str, client: openai.AsyncOpenAI
        ) -> str:
            """Simulate a slow LLM call."""
            await asyncio.sleep(0.1)  # 100ms delay
            return FaithfulnessJudgeResult(
                verdicts=[
                    ClaimVerdict(
                        claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                        verdict=FaithfulnessVerdictLabel.SUPPORTED,
                        rationale="Supported",
                        confidence=0.9,
                    )
                ]
            ).model_dump_json()

        async def slow_completeness_call(
            prompt: str, client: openai.AsyncOpenAI
        ) -> str:
            """Simulate a slow LLM call."""
            await asyncio.sleep(0.1)  # 100ms delay
            return CompletenessJudgeResult(
                verdicts=[
                    UserRequestCompletenessVerdict(
                        part_text="Part 1",
                        addressing_claims_ids=[
                            uuid.UUID("00000000-0000-0000-0000-000000000001")
                        ],
                        rationale="Addressed",
                    )
                ],
                overall_rationale="Complete",
                confidence=0.9,
            ).model_dump_json()

        mock_claim_extractor_llm.side_effect = slow_claim_extraction
        mock_faithfulness_llm.side_effect = slow_faithfulness_call
        mock_completeness_llm.side_effect = slow_completeness_call

        # When
        start_time = time.time()
        faithfulness_results, completeness_results = await evaluator.evaluate(
            dataset_entries
        )
        total_time = time.time() - start_time

        # Then
        faithfulness_successful = [
            result for result in faithfulness_results if not result.has_failures
        ]
        faithfulness_failed = [
            result for result in faithfulness_results if result.has_failures
        ]
        completeness_successful = [
            result for result in completeness_results if not result.has_failures
        ]
        completeness_failed = [
            result for result in completeness_results if result.has_failures
        ]

        assert len(faithfulness_successful) == 5
        assert len(faithfulness_failed) == 0
        assert len(completeness_successful) == 5
        assert len(completeness_failed) == 0

        # If running sequentially:
        # - claim extraction (5 * 0.1s) + judges (5 * 2 * 0.1s) = 1.5s
        # If running concurrently with default config (max_concurrent=10):
        # - Claim extraction: all 5 run together = 0.1s
        # - Entry evaluations: all 5 run together, each does 2 judges together = 0.1s
        # Total: ~0.2s
        assert (
            total_time < 0.5
        ), f"Expected concurrent execution (~0.2s), but took {total_time}s"

        # Verify claim extractions started close together
        if len(claim_call_times) >= 3:
            time_diff = claim_call_times[2] - claim_call_times[0]  # type: ignore
        assert (
            time_diff < 0.05
        ), f"First 3 claim extractions should start nearly simultaneously, but took {time_diff}s"  # noqa: E501
