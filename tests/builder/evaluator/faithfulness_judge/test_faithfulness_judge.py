"""Unit tests for FaithfulnessJudge."""

import asyncio
import math
import time
import uuid
from unittest.mock import AsyncMock, patch

import openai
import pytest

from rasa.builder.evaluator.content_processors.models import (
    Claim,
    ClaimImportance,
    Claims,
    CodeEvidence,
    DocumentationEvidence,
)
from rasa.builder.evaluator.faithfulness_judge.judge import FaithfulnessJudge
from rasa.builder.evaluator.faithfulness_judge.models import (
    ClaimVerdict,
    FaithfulnessJudgeInput,
    FaithfulnessJudgeResult,
    FaithfulnessVerdictLabel,
)
from rasa.builder.evaluator.shared.exceptions import EvaluationError
from rasa.builder.evaluator.shared.models import EvaluationFailure


@pytest.fixture(autouse=True)
def mock_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock OPENAI_API_KEY environment variable for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "failhfulness-judge-test-api-key")


class TestFaithfulnessJudge:
    """Tests for FaithfulnessJudge initialization."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "judge_inputs,"
        "llm_responses,"
        "expected_successful,"
        "expected_failed,"
        "expected_verdict_counts,"
        "expected_supported,"
        "expected_contradicted,"
        "expected_not_enough_info",
        [
            # Test case 1: Single supported verdict
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test claim",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/doc",
                                title="Test Doc",
                                content="Test content",
                                used=True,
                            )
                        ],
                    )
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
                1,
                0,
                [1],
                [1],
                [0],
                [0],
            ),
            # Test case 2: Multiple verdicts with different types
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test claim 1",
                                    metadata={},
                                ),
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000002"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test claim 2",
                                    metadata={},
                                ),
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000003"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test claim 3",
                                    metadata={},
                                ),
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/doc",
                                title="Test Doc",
                                content="Test content",
                                used=True,
                            )
                        ],
                    )
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000001"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported",
                                confidence=0.9,
                            ),
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000002"
                                ),
                                verdict=FaithfulnessVerdictLabel.CONTRADICTED,
                                rationale="Contradicted",
                                confidence=0.8,
                            ),
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000003"
                                ),
                                verdict=FaithfulnessVerdictLabel.NOT_ENOUGH_INFO,
                                rationale="Insufficient",
                                confidence=0.7,
                            ),
                        ]
                    ).model_dump_json()
                ],
                1,
                0,
                [3],
                [1],
                [1],
                [1],
            ),
            # Test case 3: Empty verdicts list
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com",
                                title="Doc",
                                content="Content",
                                used=True,
                            )
                        ],
                    )
                ],
                [FaithfulnessJudgeResult(verdicts=[]).model_dump_json()],
                1,
                0,
                [0],
                [0],
                [0],
                [0],
            ),
            # Test case 4: Multiple supported verdicts
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                ),
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000002"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                ),
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com",
                                title="Doc",
                                content="Content",
                                used=True,
                            )
                        ],
                    )
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000001"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="First support",
                                confidence=0.95,
                            ),
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000002"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Second support",
                                confidence=0.85,
                            ),
                        ]
                    ).model_dump_json()
                ],
                1,
                0,
                [2],
                [2],
                [0],
                [0],
            ),
            # Test case 5: OpenAI API error
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com",
                                title="Doc",
                                content="Content",
                                used=True,
                            )
                        ],
                    )
                ],
                [openai.OpenAIError("API rate limit exceeded")],
                0,
                1,
                [],
                [],
                [],
                [],
            ),
            # Test case 6: Timeout error
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com",
                                title="Doc",
                                content="Content",
                                used=True,
                            )
                        ],
                    )
                ],
                [asyncio.TimeoutError("Request timed out")],
                0,
                1,
                [],
                [],
                [],
                [],
            ),
            # Test case 7: Invalid JSON response
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com",
                                title="Doc",
                                content="Content",
                                used=True,
                            )
                        ],
                    )
                ],
                [ValueError("Invalid JSON")],
                0,
                1,
                [],
                [],
                [],
                [],
            ),
            # Test case 8: Generic exception
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com",
                                title="Doc",
                                content="Content",
                                used=True,
                            )
                        ],
                    )
                ],
                [Exception("Unexpected error occurred")],
                0,
                1,
                [],
                [],
                [],
                [],
            ),
            # Test case 9: Batch - all evaluations succeed
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Claim A",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/a",
                                title="Doc A",
                                content="Content A",
                                used=True,
                            )
                        ],
                    ),
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000002"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Claim B",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/b",
                                title="Doc B",
                                content="Content B",
                                used=True,
                            )
                        ],
                    ),
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000003"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Claim C",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/c",
                                title="Doc C",
                                content="Content C",
                                used=True,
                            )
                        ],
                    ),
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000001"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Verdict A",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000002"
                                ),
                                verdict=FaithfulnessVerdictLabel.CONTRADICTED,
                                rationale="Verdict B",
                                confidence=0.8,
                            )
                        ]
                    ).model_dump_json(),
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000003"
                                ),
                                verdict=FaithfulnessVerdictLabel.NOT_ENOUGH_INFO,
                                rationale="Verdict C",
                                confidence=0.7,
                            )
                        ]
                    ).model_dump_json(),
                ],
                3,
                0,
                [1, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ),
            # Test case 10: Batch - all evaluations fail
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Claim 1",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/1",
                                title="Doc 1",
                                content="Content 1",
                                used=True,
                            )
                        ],
                    ),
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Claim 2",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/2",
                                title="Doc 2",
                                content="Content 2",
                                used=True,
                            )
                        ],
                    ),
                ],
                [openai.OpenAIError("API error"), openai.OpenAIError("API error")],
                0,
                2,
                [],
                [],
                [],
                [],
            ),
            # Test case 11: Batch - some succeed, some fail (partial failure)
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Claim 1",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/1",
                                title="Doc 1",
                                content="Content 1",
                                used=True,
                            )
                        ],
                    ),
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Claim 2",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/2",
                                title="Doc 2",
                                content="Content 2",
                                used=True,
                            )
                        ],
                    ),
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Claim 3",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/3",
                                title="Doc 3",
                                content="Content 3",
                                used=True,
                            )
                        ],
                    ),
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000001"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Success 1",
                                confidence=0.9,
                            )
                        ]
                    ).model_dump_json(),
                    ValueError("LLM failed"),
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000002"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Success 2",
                                confidence=0.85,
                            )
                        ]
                    ).model_dump_json(),
                ],
                2,
                1,
                [1, 1],
                [1, 1],
                [0, 0],
                [0, 0],
            ),
            # Test case 12: Empty input list
            (
                [],
                [],
                0,
                0,
                [],
                [],
                [],
                [],
            ),
            # Test case 13: Mixed documentation and code evidence
            (
                [
                    FaithfulnessJudgeInput(
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test claim",
                                    metadata={},
                                )
                            ]
                        ),
                        evidence=[
                            DocumentationEvidence(
                                url="https://example.com/doc1",
                                title="Test Doc",
                                content="Test documentation content",
                                used=True,
                            ),
                            CodeEvidence(
                                file_path="test_file.py",
                                file_content="def test(): pass",
                                referenced=True,
                            ),
                        ],
                    )
                ],
                [
                    FaithfulnessJudgeResult(
                        verdicts=[
                            ClaimVerdict(
                                claim_id=uuid.UUID(
                                    "00000000-0000-0000-0000-000000000001"
                                ),
                                verdict=FaithfulnessVerdictLabel.SUPPORTED,
                                rationale="Supported by both doc and code",
                                confidence=0.95,
                            )
                        ]
                    ).model_dump_json()
                ],
                1,
                0,
                [1],
                [1],
                [0],
                [0],
            ),
        ],
    )
    @patch.object(FaithfulnessJudge, "_render_prompt", return_value="mocked prompt")
    @patch.object(FaithfulnessJudge, "_call_llm", new_callable=AsyncMock)
    async def test_evaluate(
        self,
        mock_call_llm: AsyncMock,
        mock_render_prompt: AsyncMock,
        judge_inputs: list[FaithfulnessJudgeInput],
        llm_responses: list[str],
        expected_successful: int,
        expected_failed: int,
        expected_verdict_counts: list[int],
        expected_supported: list[int],
        expected_contradicted: list[int],
        expected_not_enough_info: list[int],
    ) -> None:
        # Given
        judge = FaithfulnessJudge(
            model="test-gpt",
            max_concurrent_evaluations=10,
        )
        mock_call_llm.side_effect = llm_responses

        # When
        results = await judge.evaluate(judge_inputs)

        # Then
        successful = [
            result for result in results if isinstance(result, FaithfulnessJudgeResult)
        ]
        failed = [
            failure for failure in results if isinstance(failure, EvaluationFailure)
        ]

        assert len(successful) == expected_successful
        assert len(failed) == expected_failed
        assert mock_call_llm.call_count == len(judge_inputs)
        assert mock_render_prompt.call_count == len(judge_inputs)

        # Verify successful evaluations
        for i, expected_count in enumerate(expected_verdict_counts):
            assert len(successful[i].verdicts) == expected_count
            assert len(successful[i].supported_verdicts) == expected_supported[i]
            assert len(successful[i].contradicted_verdicts) == expected_contradicted[i]
            assert (
                len(successful[i].not_enough_info_verdicts)
                == expected_not_enough_info[i]
            )

        # Verify failed evaluations have proper error type
        if expected_failed > 0:
            for failure in failed:
                assert failure.error_type == EvaluationError.__name__
                assert "Failed to evaluate" in failure.error_message

    @pytest.mark.asyncio
    @patch.object(FaithfulnessJudge, "_call_llm", new_callable=AsyncMock)
    async def test_evaluate_concurrent_execution(
        self, mock_call_llm: AsyncMock
    ) -> None:
        """Test that evaluations run concurrently, not sequentially."""

        # Given
        max_concurrent_evaluations = 3
        number_of_inputs = 10
        time_per_evaluation = 0.1
        judge = FaithfulnessJudge(
            model="test-gpt",
            max_concurrent_evaluations=max_concurrent_evaluations,
        )
        expected_concurrent_time = (
            math.ceil(number_of_inputs / max_concurrent_evaluations)
            * time_per_evaluation
        )
        # Add 50% to the expected concurrent time to account for the overhead of the
        # concurrent execution
        expected_concurrent_time = expected_concurrent_time * 1.5

        # Create multiple inputs
        judge_inputs: list[FaithfulnessJudgeInput] = []
        for i in range(number_of_inputs):
            claims = Claims(
                claims=[
                    Claim(
                        importance=ClaimImportance.HIGH,
                        text=f"Test claim {i}",
                        metadata={},
                    )
                ]
            )
            evidence = [
                DocumentationEvidence(
                    url=f"https://example.com/doc{i}",
                    title=f"Test Doc {i}",
                    content=f"Test documentation content {i}",
                    used=True,
                )
            ]
            judge_inputs.append(
                FaithfulnessJudgeInput(claims=claims, evidence=evidence)
            )

        call_times = []

        async def slow_llm_call(prompt: str, client: openai.AsyncOpenAI) -> str:
            """Simulate a slow LLM call."""
            call_times.append(time.time())
            await asyncio.sleep(time_per_evaluation)
            return FaithfulnessJudgeResult(
                verdicts=[
                    ClaimVerdict(
                        claim_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                        verdict=FaithfulnessVerdictLabel.SUPPORTED,
                        rationale="Test verdict",
                        confidence=0.9,
                    )
                ]
            ).model_dump_json()

        mock_call_llm.side_effect = slow_llm_call

        # When
        start_time = time.time()
        results = await judge.evaluate(judge_inputs)
        total_time = time.time() - start_time

        # Then
        successful = [
            result for result in results if isinstance(result, FaithfulnessJudgeResult)
        ]
        failed = [
            failure for failure in results if isinstance(failure, EvaluationFailure)
        ]

        assert len(successful) == number_of_inputs
        assert len(failed) == 0
        assert (
            total_time <= expected_concurrent_time
        ), f"Expected concurrent execution, but took {total_time}s (max expected is {expected_concurrent_time}s)"  # noqa: E501

        # Verify calls started close together (within first batch)
        if len(call_times) >= max_concurrent_evaluations:
            # First max_concurrent_evaluations calls should start within
            # a short time window
            time_diff = call_times[max_concurrent_evaluations - 1] - call_times[0]
            assert (
                time_diff < 0.05
            ), f"First {max_concurrent_evaluations} calls should start nearly simultaneously (max expected is 0.05s)"  # noqa: E501
