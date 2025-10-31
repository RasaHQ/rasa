"""Unit tests for ClaimExtractor."""

import asyncio
from unittest.mock import AsyncMock, patch

import openai
import pytest
from jinja2 import Template

from rasa.builder.evaluator.content_processors.claim_extractor import ClaimExtractor
from rasa.builder.evaluator.content_processors.models import (
    ClaimExtractionFailure,
    Claims,
)
from rasa.builder.evaluator.exceptions import ClaimExtractionError


class TestClaimExtractor:
    """Tests for ClaimExtractor initialization."""

    @pytest.fixture(autouse=True)
    def setup_openai_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set OPENAI_API_KEY for all tests in this class."""
        monkeypatch.setenv("OPENAI_API_KEY", "api_key_claim_extraction_test")

    def test_init(self) -> None:
        """Test that ClaimExtractor initializes with correctly."""
        # When
        extractor = ClaimExtractor(
            model="test_gpt",
            temperature=0.0,
            max_tokens=2000,
            timeout=100,
        )

        # Then
        assert hasattr(extractor, "_prompt_template")
        assert isinstance(extractor._prompt_template, Template)
        assert hasattr(extractor, "_response_schema")
        assert isinstance(extractor._response_schema, dict)
        assert "claims" in extractor._response_schema["properties"]
        rendered_prompt = extractor._prompt_template.render(response="test response")
        assert rendered_prompt.startswith(
            "You are an expert at analyzing technical responses and extracting atomic, verifiable"  # noqa: E501
        )
        assert "test response" in rendered_prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "llm_response_json,"
        "expected_claim_count,"
        "expected_high_count,"
        "expected_medium_count,"
        "expected_low_count",
        [
            # Test case 1: Single high importance claim
            (
                '{"claims": [{"importance": "high", "text": "The system supports authentication", "metadata": {}}]}',  # noqa: E501
                1,
                1,
                0,
                0,
            ),
            # Test case 2: Multiple claims with different importance levels
            (
                '{"claims": ['
                '{"importance": "high", "text": "Critical feature", "metadata": {}},'
                '{"importance": "medium", "text": "Secondary feature", "metadata": {}},'
                '{"importance": "low", "text": "Minor detail", "metadata": {}}'
                "]}",
                3,
                1,
                1,
                1,
            ),
            # Test case 3: Empty claims list
            ('{"claims": []}', 0, 0, 0, 0),
            # Test case 4: Multiple high importance claims
            (
                '{"claims": ['
                '{"importance": "high", "text": "First critical claim", "metadata": {}},'  # noqa: E501
                '{"importance": "high", "text": "Second critical claim", "metadata": {}}'  # noqa: E501
                "]}",
                2,
                2,
                0,
                0,
            ),
        ],
    )
    @patch.object(ClaimExtractor, "_call_llm", new_callable=AsyncMock)
    async def test_extract_success(
        self,
        mock_call_llm: AsyncMock,
        llm_response_json: str,
        expected_claim_count: int,
        expected_high_count: int,
        expected_medium_count: int,
        expected_low_count: int,
    ) -> None:
        """Test successful claim extraction with various valid responses."""
        # Given
        extractor = ClaimExtractor(
            model="test_gpt",
            temperature=0.0,
            max_tokens=2000,
            timeout=100,
        )
        copilot_response = "Test copilot response"
        mock_call_llm.return_value = llm_response_json

        # When
        results = await extractor.extract([copilot_response])

        # Then
        assert len(results) == 1
        assert isinstance(results[0], Claims)
        claims = results[0]
        assert len(claims.claims) == expected_claim_count
        assert len(claims.high_importance_claims) == expected_high_count
        assert len(claims.medium_importance_claims) == expected_medium_count
        assert len(claims.low_importance_claims) == expected_low_count
        mock_call_llm.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_type,exception_message,expected_error_substring",
        [
            # Test case 1: OpenAI API error
            (
                openai.OpenAIError,
                "API rate limit exceeded",
                "Failed to extract claims",
            ),
            # Test case 2: Timeout error
            (
                asyncio.TimeoutError,
                "Request timed out",
                "Failed to extract claims",
            ),
            # Test case 3: Invalid JSON response
            (
                ValueError,
                "Invalid JSON",
                "Failed to extract claims",
            ),
            # Test case 4: Generic exception
            (
                Exception,
                "Unexpected error occurred",
                "Failed to extract claims",
            ),
        ],
    )
    @patch.object(ClaimExtractor, "_call_llm", new_callable=AsyncMock)
    async def test_extract_exceptions(
        self,
        mock_call_llm: AsyncMock,
        exception_type: type,
        exception_message: str,
        expected_error_substring: str,
    ) -> None:
        """Test claim extraction error handling for various exception types."""
        # Given
        extractor = ClaimExtractor(
            model="test_gpt",
            temperature=0.0,
            max_tokens=2000,
            timeout=100,
        )
        copilot_response = "Test copilot response"
        mock_call_llm.side_effect = exception_type(exception_message)

        # When
        results = await extractor.extract([copilot_response])

        # Then
        assert len(results) == 1
        assert isinstance(results[0], ClaimExtractionFailure)
        failure = results[0]
        assert expected_error_substring in failure.error_message
        mock_call_llm.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "responses,"
        "llm_responses,"
        "expected_successful,"
        "expected_failed,"
        "expected_claim_texts,"
        "failed_indices",
        [
            # Test case 1: All extractions succeed
            (
                ["Response A", "Response B", "Response C"],
                [
                    '{"claims": [{"importance": "high", "text": "Claim A", "metadata": {}}]}',  # noqa: E501
                    '{"claims": [{"importance": "medium", "text": "Claim B", "metadata": {}}]}',  # noqa: E501
                    '{"claims": [{"importance": "low", "text": "Claim C", "metadata": {}}]}',  # noqa: E501
                ],
                3,
                0,
                ["Claim A", "Claim B", "Claim C"],
                [],
            ),
            # Test case 2: All extractions fail
            (
                ["Response 1", "Response 2"],
                [openai.OpenAIError("API error"), openai.OpenAIError("API error")],
                0,
                2,
                [],
                [0, 1],
            ),
            # Test case 3: Some succeed, some fail (partial failure)
            (
                ["Response 1", "Response 2", "Response 3"],
                [
                    '{"claims": [{"importance": "high", "text": "Success 1", "metadata": {}}]}',  # noqa: E501
                    ValueError("LLM failed"),
                    '{"claims": [{"importance": "high", "text": "Success 2", "metadata": {}}]}',  # noqa: E501
                ],
                2,
                1,
                ["Success 1", "Success 2"],
                [1],
            ),
            # Test case 4: Empty response list
            (
                [],
                [],
                0,
                0,
                [],
                [],
            ),
            # Test case 5: Single response
            (
                ["Single response"],
                [
                    '{"claims": [{"importance": "high", "text": "Single claim", "metadata": {}}]}'  # noqa: E501
                ],
                1,
                0,
                ["Single claim"],
                [],
            ),
        ],
    )
    @patch.object(ClaimExtractor, "_call_llm", new_callable=AsyncMock)
    async def test_extract_batch(
        self,
        mock_call_llm: AsyncMock,
        responses: list[str],
        llm_responses: list[str],
        expected_successful: int,
        expected_failed: int,
        expected_claim_texts: list[str],
        failed_indices: list[int],
    ) -> None:
        """Test batch extraction with various scenarios."""
        # Given
        extractor = ClaimExtractor(
            model="test_gpt",
            temperature=0.0,
            max_tokens=2000,
            timeout=100,
        )
        mock_call_llm.side_effect = llm_responses

        # When
        results = await extractor.extract(responses)

        # Then
        # Separate successful and failed results
        successful = [r for r in results if isinstance(r, Claims)]
        failed = [r for r in results if isinstance(r, ClaimExtractionFailure)]

        assert len(successful) == expected_successful
        assert len(failed) == expected_failed
        assert mock_call_llm.call_count == len(responses)

        # Verify successful extractions and order preservation
        for i, expected_text in enumerate(expected_claim_texts):
            assert successful[i].claims[0].text == expected_text

        # Verify failed extraction count
        if expected_failed > 0:
            # Verify all failures have ClaimExtractionError type
            for failure in failed:
                assert failure.error_type == ClaimExtractionError.__name__

    @pytest.mark.asyncio
    @patch.object(ClaimExtractor, "_call_llm", new_callable=AsyncMock)
    async def test_extract_concurrent_execution(self, mock_call_llm: AsyncMock) -> None:
        """Test that extractions run concurrently, not sequentially."""
        import time

        # Given
        extractor = ClaimExtractor(
            model="test_gpt",
            temperature=0.0,
            max_tokens=2000,
            timeout=100,
            max_concurrent_extractions=3,
        )
        responses = [
            "Response 1",
            "Response 2",
            "Response 3",
            "Response 4",
            "Response 5",
        ]

        call_times = []

        async def slow_llm_call(prompt: str, client: openai.AsyncOpenAI) -> str:
            """Simulate a slow LLM call."""
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # 100ms delay
            return '{"claims": [{"importance": "high", "text": "Test claim", "metadata": {}}]}'  # noqa: E501

        mock_call_llm.side_effect = slow_llm_call

        # When
        start_time = time.time()
        results = await extractor.extract(responses)
        total_time = time.time() - start_time

        # Then
        successful = [r for r in results if isinstance(r, Claims)]
        failed = [r for r in results if isinstance(r, ClaimExtractionFailure)]

        assert len(successful) == 5
        assert len(failed) == 0

        # If running sequentially, would take 5 * 0.1 = 0.5 seconds
        # If running concurrently with limit of 3, should be faster
        # We expect: batch1 (3 items, 0.1s) + batch2 (2 items, 0.1s) = ~0.2s
        assert (
            total_time < 0.35
        ), f"Expected concurrent execution, but took {total_time}s (max expected is 0.35s)"  # noqa: E501

        # Verify calls started close together (within first batch)
        if len(call_times) >= 3:
            # First 3 calls should start within a short time window
            time_diff = call_times[2] - call_times[0]
            assert (
                time_diff < 0.05
            ), "First 3 calls should start nearly simultaneously (max expected is 0.05s)"  # noqa: E501
