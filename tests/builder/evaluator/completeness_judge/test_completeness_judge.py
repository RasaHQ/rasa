"""Unit tests for CompletenessJudge."""

import asyncio
import math
import time
import uuid
from typing import List
from unittest.mock import AsyncMock, patch

import openai
import pytest

from rasa.builder.copilot.constants import (
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_USER,
)
from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotChatMessage,
    InternalCopilotRequestChatMessage,
    LogContent,
    ResponseCategory,
    TextContent,
    UserChatMessage,
)
from rasa.builder.evaluator.completeness_judge.judge import CompletenessJudge
from rasa.builder.evaluator.completeness_judge.models import (
    CompletenessJudgeInput,
    CompletenessJudgeResult,
    UserRequestCompletenessVerdict,
)
from rasa.builder.evaluator.content_processors.models import (
    Claim,
    ClaimImportance,
    Claims,
)
from rasa.builder.evaluator.shared.exceptions import EvaluationError
from rasa.builder.evaluator.shared.models import EvaluationFailure


@pytest.fixture(autouse=True)
def mock_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock OPENAI_API_KEY environment variable for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "completeness-judge-test-api-key")


class TestCompletenessJudge:
    """Tests for CompletenessJudge."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "judge_inputs,"
        "llm_responses,"
        "expected_successful,"
        "expected_failed,"
        "expected_total_parts,"
        "expected_covered_parts,"
        "expected_uncovered_parts",
        [
            # Test case 1: Single part fully covered
            (
                [
                    CompletenessJudgeInput(
                        user_message="How do I add a new slot?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="You can add slots in the domain file",
                                    metadata={},
                                ),
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000002"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Slots store conversation information",
                                    metadata={},
                                ),
                            ]
                        ),
                    )
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="How to add a new slot",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000001")
                                ],
                                rationale="Claim explains how to add slots",
                            )
                        ],
                        overall_rationale="Request fully addressed",
                        confidence=0.9,
                    ).model_dump_json()
                ],
                1,
                0,
                [1],
                [1],
                [0],
            ),
            # Test case 2: Multiple parts with partial coverage
            (
                [
                    CompletenessJudgeInput(
                        user_message="How do I add a slot and migrate my flow?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="You can add slots in the domain file",
                                    metadata={},
                                ),
                            ]
                        ),
                    )
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="How to add a slot",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000001")
                                ],
                                rationale="Slot addition is covered",
                            ),
                            UserRequestCompletenessVerdict(
                                part_text="How to migrate flows",
                                addressing_claims_ids=[],
                                rationale="Flow migration not addressed",
                            ),
                        ],
                        overall_rationale="Partial coverage: slot addition covered, flow migration missing",  # noqa: E501
                        confidence=0.8,
                    ).model_dump_json()
                ],
                1,
                0,
                [2],
                [1],
                [1],
            ),
            # Test case 3: All parts uncovered
            (
                [
                    CompletenessJudgeInput(
                        user_message="How do I migrate my flow?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Slots store information",
                                    metadata={},
                                ),
                            ]
                        ),
                    )
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="How to migrate flows",
                                addressing_claims_ids=[],
                                rationale="No claims address flow migration",
                            )
                        ],
                        overall_rationale="Request not addressed",
                        confidence=0.95,
                    ).model_dump_json()
                ],
                1,
                0,
                [1],
                [0],
                [1],
            ),
            # Test case 4: Multiple parts all covered
            (
                [
                    CompletenessJudgeInput(
                        user_message="Explain slots, entities, and forms",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Slots store conversation data",
                                    metadata={},
                                ),
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000002"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Entities are extracted from user input",
                                    metadata={},
                                ),
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000003"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Forms collect required information",
                                    metadata={},
                                ),
                            ]
                        ),
                    )
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Explanation of slots",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000001")
                                ],
                                rationale="Slots explained",
                            ),
                            UserRequestCompletenessVerdict(
                                part_text="Explanation of entities",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000002")
                                ],
                                rationale="Entities explained",
                            ),
                            UserRequestCompletenessVerdict(
                                part_text="Explanation of forms",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000003")
                                ],
                                rationale="Forms explained",
                            ),
                        ],
                        overall_rationale="All parts fully addressed",
                        confidence=0.95,
                    ).model_dump_json()
                ],
                1,
                0,
                [3],
                [3],
                [0],
            ),
            # Test case 5: OpenAI API error
            (
                [
                    CompletenessJudgeInput(
                        user_message="How do I add a slot?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                    )
                ],
                [openai.OpenAIError("API rate limit exceeded")],
                0,
                1,
                [],
                [],
                [],
            ),
            # Test case 6: Timeout error
            (
                [
                    CompletenessJudgeInput(
                        user_message="How do I add a slot?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                    )
                ],
                [asyncio.TimeoutError("Request timed out")],
                0,
                1,
                [],
                [],
                [],
            ),
            # Test case 7: Invalid JSON response
            (
                [
                    CompletenessJudgeInput(
                        user_message="How do I add a slot?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                    )
                ],
                [ValueError("Invalid JSON")],
                0,
                1,
                [],
                [],
                [],
            ),
            # Test case 8: Generic exception
            (
                [
                    CompletenessJudgeInput(
                        user_message="How do I add a slot?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test",
                                    metadata={},
                                )
                            ]
                        ),
                    )
                ],
                [Exception("Unexpected error occurred")],
                0,
                1,
                [],
                [],
                [],
            ),
            # Test case 9: Batch - all evaluations succeed
            (
                [
                    CompletenessJudgeInput(
                        user_message="How do I add a slot?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Add slots in domain",
                                    metadata={},
                                )
                            ]
                        ),
                    ),
                    CompletenessJudgeInput(
                        user_message="What are entities?",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000002"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Entities extract info",
                                    metadata={},
                                )
                            ]
                        ),
                    ),
                    CompletenessJudgeInput(
                        user_message="Explain forms",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000003"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Forms collect data",
                                    metadata={},
                                )
                            ]
                        ),
                    ),
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Adding slots",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000001")
                                ],
                                rationale="Covered",
                            )
                        ],
                        overall_rationale="Fully addressed",
                        confidence=0.9,
                    ).model_dump_json(),
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Entity explanation",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000002")
                                ],
                                rationale="Covered",
                            )
                        ],
                        overall_rationale="Fully addressed",
                        confidence=0.85,
                    ).model_dump_json(),
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Forms explanation",
                                addressing_claims_ids=[],
                                rationale="Not covered",
                            )
                        ],
                        overall_rationale="Not addressed",
                        confidence=0.8,
                    ).model_dump_json(),
                ],
                3,
                0,
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 1],
            ),
            # Test case 10: Batch - all evaluations fail
            (
                [
                    CompletenessJudgeInput(
                        user_message="Request 1",
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test 1",
                                    metadata={},
                                )
                            ]
                        ),
                    ),
                    CompletenessJudgeInput(
                        user_message="Request 2",
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test 2",
                                    metadata={},
                                )
                            ]
                        ),
                    ),
                ],
                [openai.OpenAIError("API error"), openai.OpenAIError("API error")],
                0,
                2,
                [],
                [],
                [],
            ),
            # Test case 11: Batch - some succeed, some fail (partial failure)
            (
                [
                    CompletenessJudgeInput(
                        user_message="Request 1",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000001"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test 1",
                                    metadata={},
                                )
                            ]
                        ),
                    ),
                    CompletenessJudgeInput(
                        user_message="Request 2",
                        claims=Claims(
                            claims=[
                                Claim(
                                    importance=ClaimImportance.HIGH,
                                    text="Test 2",
                                    metadata={},
                                )
                            ]
                        ),
                    ),
                    CompletenessJudgeInput(
                        user_message="Request 3",
                        claims=Claims(
                            claims=[
                                Claim(
                                    id=uuid.UUID(
                                        "00000000-0000-0000-0000-000000000003"
                                    ),
                                    importance=ClaimImportance.HIGH,
                                    text="Test 3",
                                    metadata={},
                                )
                            ]
                        ),
                    ),
                ],
                [
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Part 1",
                                addressing_claims_ids=[
                                    uuid.UUID("00000000-0000-0000-0000-000000000001")
                                ],
                                rationale="Success 1",
                            )
                        ],
                        overall_rationale="Success",
                        confidence=0.9,
                    ).model_dump_json(),
                    ValueError("LLM failed"),
                    CompletenessJudgeResult(
                        verdicts=[
                            UserRequestCompletenessVerdict(
                                part_text="Part 3",
                                addressing_claims_ids=[],
                                rationale="Success 2",
                            )
                        ],
                        overall_rationale="Partial",
                        confidence=0.85,
                    ).model_dump_json(),
                ],
                2,
                1,
                [1, 1],
                [1, 0],
                [0, 1],
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
            ),
        ],
    )
    @patch.object(CompletenessJudge, "_render_prompt", return_value="mocked prompt")
    @patch.object(CompletenessJudge, "_call_llm", new_callable=AsyncMock)
    async def test_evaluate(
        self,
        mock_call_llm: AsyncMock,
        mock_render_prompt: AsyncMock,
        judge_inputs: list[CompletenessJudgeInput],
        llm_responses: list[str],
        expected_successful: int,
        expected_failed: int,
        expected_total_parts: list[int],
        expected_covered_parts: list[int],
        expected_uncovered_parts: list[int],
    ) -> None:
        # Given
        judge = CompletenessJudge(
            model="test-gpt",
            max_concurrent_evaluations=10,
        )
        mock_call_llm.side_effect = llm_responses

        # When
        results = await judge.evaluate(judge_inputs)

        # Then
        successful = [
            result for result in results if isinstance(result, CompletenessJudgeResult)
        ]
        failed = [
            failure for failure in results if isinstance(failure, EvaluationFailure)
        ]

        assert len(successful) == expected_successful
        assert len(failed) == expected_failed
        assert mock_call_llm.call_count == len(judge_inputs)
        assert mock_render_prompt.call_count == len(judge_inputs)

        # Verify successful evaluations
        for i, expected_total in enumerate(expected_total_parts):
            assert len(successful[i].verdicts) == expected_total
            assert len(successful[i].covered_parts) == expected_covered_parts[i]
            assert len(successful[i].uncovered_parts) == expected_uncovered_parts[i]

        # Verify failed evaluations have proper error type
        if expected_failed > 0:
            for failure in failed:
                assert failure.error_type == EvaluationError.__name__
                assert "Failed to evaluate" in failure.error_message

    @pytest.mark.asyncio
    @patch.object(CompletenessJudge, "_call_llm", new_callable=AsyncMock)
    async def test_evaluate_concurrent_execution(
        self, mock_call_llm: AsyncMock
    ) -> None:
        """Test that evaluations run concurrently, not sequentially."""
        # Given
        max_concurrent_evaluations = 3
        number_of_inputs = 10
        time_per_evaluation = 0.1
        judge = CompletenessJudge(
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
        judge_inputs: list[CompletenessJudgeInput] = []
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
            judge_inputs.append(
                CompletenessJudgeInput(user_message=f"Test request {i}", claims=claims)
            )

        call_times = []

        async def slow_llm_call(prompt: str, client: openai.AsyncOpenAI) -> str:
            """Simulate a slow LLM call."""
            call_times.append(time.time())
            await asyncio.sleep(time_per_evaluation)
            return CompletenessJudgeResult(
                verdicts=[
                    UserRequestCompletenessVerdict(
                        part_text="Test part",
                        addressing_claims_ids=[],
                        rationale="Test rationale",
                    )
                ],
                overall_rationale="Test overall",
                confidence=0.9,
            ).model_dump_json()

        mock_call_llm.side_effect = slow_llm_call

        # When
        start_time = time.time()
        results = await judge.evaluate(judge_inputs)
        total_time = time.time() - start_time

        # Then
        successful = [
            result for result in results if isinstance(result, CompletenessJudgeResult)
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
            assert time_diff < 0.05, (
                f"First {max_concurrent_evaluations} calls should start "
                "nearly simultaneously (max expected is 0.05s)"
            )

    @pytest.mark.parametrize(
        "chat_history,expected_output",
        [
            # Test case 1: Empty chat history
            ([], "No previous conversation history."),
            # Test case 2: Single user message
            (
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[
                            TextContent(type="text", text="Hello, how can I help?")
                        ],
                    )
                ],
                "User: Hello, how can I help?",
            ),
            # Test case 3: Single copilot message
            (
                [
                    CopilotChatMessage(
                        role=ROLE_COPILOT,
                        content=[
                            TextContent(type="text", text="I can help you with that!")
                        ],
                    )
                ],
                "Copilot: I can help you with that!",
            ),
            # Test case 4: Single internal copilot message with text only
            (
                [
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[
                            TextContent(type="text", text="Analyzing training error")
                        ],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    )
                ],
                "Copilot (Internal): Analyzing training error",
            ),
            # Test case 5: Single internal copilot message with logs only
            (
                [
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[
                            LogContent(type="log", content="Error: Training failed")
                        ],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    )
                ],
                "Copilot (Internal): Logs: Error: Training failed",
            ),
            # Test case 6: Single internal copilot message with both text and logs
            (
                [
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[
                            TextContent(type="text", text="Analyzing error"),
                            LogContent(type="log", content="Error: Training failed"),
                        ],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    )
                ],
                "Copilot (Internal): Analyzing error\nLogs: Error: Training failed",
            ),
            # Test case 7: Multiple messages - user and copilot
            (
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="What is Rasa?")],
                    ),
                    CopilotChatMessage(
                        role=ROLE_COPILOT,
                        content=[
                            TextContent(
                                type="text",
                                text="Rasa is an open source framework",
                            )
                        ],
                    ),
                ],
                "User: What is Rasa?\nCopilot: Rasa is an open source framework",
            ),
            # Test case 8: Multiple messages - user, copilot, and internal
            (
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="How do I add a slot?")],
                    ),
                    CopilotChatMessage(
                        role=ROLE_COPILOT,
                        content=[
                            TextContent(
                                type="text",
                                text="You can add slots in the domain file",
                            )
                        ],
                    ),
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[
                            TextContent(type="text", text="Checking domain file"),
                            LogContent(type="log", content="Domain file found"),
                        ],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    ),
                ],
                (
                    "User: How do I add a slot?\n"
                    "Copilot: You can add slots in the domain file\n"
                    "Copilot (Internal): Checking domain file\n"
                    "Logs: Domain file found"
                ),
            ),
            # Test case 9: Internal copilot message with no text or logs
            # (should be skipped)
            (
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="Hello")],
                    ),
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[],  # Empty content
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    ),
                ],
                "User: Hello",
            ),
            # Test case 10: Complex conversation with multiple message types
            (
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="Tell me about slots")],
                    ),
                    CopilotChatMessage(
                        role=ROLE_COPILOT,
                        content=[
                            TextContent(
                                type="text",
                                text="Slots store conversation data",
                            )
                        ],
                    ),
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="How do I add one?")],
                    ),
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[LogContent(type="log", content="Checking domain.yml")],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    ),
                    CopilotChatMessage(
                        role=ROLE_COPILOT,
                        content=[
                            TextContent(
                                type="text", text="Add slots in the domain file"
                            )
                        ],
                    ),
                ],
                (
                    "User: Tell me about slots\n"
                    "Copilot: Slots store conversation data\n"
                    "User: How do I add one?\n"
                    "Copilot (Internal): Logs: Checking domain.yml\n"
                    "Copilot: Add slots in the domain file"
                ),
            ),
        ],
    )
    def test_format_chat_history(
        self, chat_history: List[ChatMessage], expected_output: str
    ) -> None:
        """Test _format_chat_history with various chat history scenarios."""
        # Given
        judge = CompletenessJudge(
            model="test-gpt",
            max_concurrent_evaluations=10,
        )

        # When
        result = judge._format_chat_history(chat_history)

        # Then
        assert result == expected_output

    @pytest.mark.parametrize(
        "user_message,chat_history,claims,expected_contains",
        [
            # Test case 1: Basic prompt with user message and claims
            (
                "How do I add a slot?",
                [],
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                            importance=ClaimImportance.HIGH,
                            text="You can add slots in the domain file",
                            metadata={},
                        )
                    ]
                ),
                [
                    "How do I add a slot?",
                    "No previous conversation history",
                    "00000000-0000-0000-0000-000000000001",
                ],
            ),
            # Test case 2: Prompt with chat history
            (
                "How do I create one?",
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="Tell me about flows")],
                    ),
                    CopilotChatMessage(
                        role=ROLE_COPILOT,
                        content=[
                            TextContent(
                                type="text", text="Flows are conversation patterns"
                            )
                        ],
                    ),
                ],
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
                            importance=ClaimImportance.HIGH,
                            text="Create flows in the flows directory",
                            metadata={},
                        )
                    ]
                ),
                [
                    "How do I create one?",
                    "User: Tell me about flows",
                    "Copilot: Flows are conversation patterns",
                    "00000000-0000-0000-0000-000000000002",
                ],
            ),
            # Test case 3: Prompt with multiple claims
            (
                "Explain slots and entities",
                [],
                Claims(
                    claims=[
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
                            importance=ClaimImportance.HIGH,
                            text="Slots store data",
                            metadata={},
                        ),
                        Claim(
                            id=uuid.UUID("00000000-0000-0000-0000-000000000004"),
                            importance=ClaimImportance.MEDIUM,
                            text="Entities extract information",
                            metadata={},
                        ),
                    ]
                ),
                [
                    "Explain slots and entities",
                    "00000000-0000-0000-0000-000000000003",
                    "00000000-0000-0000-0000-000000000004",
                ],
            ),
            # Test case 4: Prompt with internal copilot message in history
            (
                "What went wrong?",
                [
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[
                            TextContent(type="text", text="Analyzing error"),
                            LogContent(type="log", content="Error: Training failed"),
                        ],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    )
                ],
                Claims(claims=[]),
                [
                    "What went wrong?",
                    "Copilot (Internal)",
                    "Analyzing error",
                    "Logs: Error: Training failed",
                ],
            ),
            # Test case 5: Empty claims
            (
                "How do I migrate?",
                [],
                Claims(claims=[]),
                ["How do I migrate?", "No previous conversation history"],
            ),
        ],
    )
    def test_render_prompt(
        self,
        user_message: str,
        chat_history: List[ChatMessage],
        claims: Claims,
        expected_contains: List[str],
    ) -> None:
        """Test _render_prompt with various inputs."""
        # Given
        judge = CompletenessJudge(
            model="test-gpt",
            max_concurrent_evaluations=10,
        )
        input_data = CompletenessJudgeInput(
            user_message=user_message,
            chat_history=chat_history,
            claims=claims,
        )

        # When
        result = judge._render_prompt(input_data)

        # Then
        assert isinstance(result, str)
        assert len(result) > 0
        # Check that all expected strings are in the rendered prompt
        for expected_str in expected_contains:
            assert (
                expected_str in result
            ), f"Expected '{expected_str}' to be in the rendered prompt"

    def test_get_prompt_template(self) -> None:
        """Test _get_prompt_template returns a valid Jinja2 template."""
        # Given
        judge = CompletenessJudge(
            model="test-gpt",
            max_concurrent_evaluations=10,
        )

        # When
        template = judge._get_prompt_template()

        # Then
        assert template is not None
        # Verify it's a Jinja2 Template by checking it has a render method
        assert hasattr(template, "render")
        assert callable(template.render)

        # Verify the template contains expected content by rendering with test
        # data
        test_data = {
            "user_message": "Test message",
            "chat_history": "Test history",
            "claims": '{"claims": []}',
        }
        rendered = template.render(**test_data)
        assert isinstance(rendered, str)
        assert "Test message" in rendered
        assert "Test history" in rendered

    def test_get_response_schema(self) -> None:
        """Test _get_response_schema returns a valid JSON schema."""
        # Given
        judge = CompletenessJudge(
            model="test-gpt",
            max_concurrent_evaluations=10,
        )

        # When
        schema = judge._get_response_schema()

        # Then
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Verify required fields
        required_fields = schema["required"]
        assert "verdicts" in required_fields
        assert "overall_rationale" in required_fields
        assert "confidence" in required_fields

        # Verify properties structure
        properties = schema["properties"]
        assert "verdicts" in properties
        assert "overall_rationale" in properties
        assert "confidence" in properties

        # Verify verdicts structure
        verdicts_schema = properties["verdicts"]
        assert verdicts_schema["type"] == "array"
        assert "items" in verdicts_schema
        verdict_item_schema = verdicts_schema["items"]
        assert "properties" in verdict_item_schema
        verdict_properties = verdict_item_schema["properties"]
        assert "part_text" in verdict_properties
        assert "addressing_claims_ids" in verdict_properties
        assert "rationale" in verdict_properties

        # Verify confidence is a number between 0 and 1
        confidence_schema = properties["confidence"]
        assert confidence_schema["type"] == "number"
        assert confidence_schema["minimum"] == 0
        assert confidence_schema["maximum"] == 1
