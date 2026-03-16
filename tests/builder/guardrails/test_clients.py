"""Unit tests for guardrails clients."""

import asyncio
from typing import Any, Dict, Optional, Type
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from rasa.builder import config
from rasa.builder.guardrails.clients import LakeraAIGuardrails
from rasa.builder.guardrails.exceptions import GuardrailsError
from rasa.builder.guardrails.models import GuardrailResponse, LakeraGuardrailRequest
from rasa.builder.guardrails.utils import create_guardrail_request


class MockPostResponse:
    """Mock class that implements async context manager protocol for post responses."""

    def __init__(
        self,
        status: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        text: str = "",
    ) -> None:
        self.response = AsyncMock()
        self.response.status = status
        self.response.json = AsyncMock(return_value=json_data or {})
        self.response.text = AsyncMock(return_value=text)

    async def __aenter__(self) -> AsyncMock:
        return self.response

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        return None


class TestGuardrailsClient:
    @pytest.fixture(autouse=True)
    def clear_schedule_check_cache(self) -> None:
        """Ensure schedule_check LRU cache is clean between tests."""
        # Clear the cache for all LakeraAIGuardrails instances
        LakeraAIGuardrails.schedule_check.cache_clear()

    @pytest.mark.asyncio
    async def test_schedule_check_caches_tasks(self, monkeypatch: pytest.MonkeyPatch):
        """Test that schedule_check method caches tasks correctly."""
        # Given

        client = LakeraAIGuardrails(api_key="test_key")

        mock_response = GuardrailResponse(flagged=False)
        mock_send_request = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(client, "send_request", mock_send_request)

        # Create requests with same content - should get same cached task
        request1 = create_guardrail_request(
            client_type=type(client),
            user_text="hello",
            hello_rasa_user_id="user-1",
            hello_rasa_project_id="proj-1",
            lakera_project_id="lakera-1",
        )
        request2 = create_guardrail_request(
            client_type=type(client),
            user_text="hello",
            hello_rasa_user_id="user-1",
            hello_rasa_project_id="proj-1",
            lakera_project_id="lakera-1",
        )

        # Different arguments - different task
        request3 = create_guardrail_request(
            client_type=type(client),
            user_text="hello",
            hello_rasa_user_id="user-1",
            hello_rasa_project_id="proj-2",
            lakera_project_id="lakera-1",
        )
        request4 = create_guardrail_request(
            client_type=type(client),
            user_text="hello2",
            hello_rasa_user_id="user-1",
            hello_rasa_project_id="proj-1",
            lakera_project_id="lakera-1",
        )
        t1 = client.schedule_check(request1)
        t2 = client.schedule_check(request2)
        assert t1 is t2

        t3 = client.schedule_check(request3)
        t4 = client.schedule_check(request4)
        assert t3 is not t1
        assert t4 is not t1

        # Await tasks to avoid warnings about pending tasks
        res1, res2, res3 = await asyncio.gather(t1, t3, t4)
        assert isinstance(res1, GuardrailResponse)
        assert isinstance(res2, GuardrailResponse)
        assert isinstance(res3, GuardrailResponse)
        assert not res1.flagged
        assert not res2.flagged
        assert not res3.flagged


class TestLakeraAIGuardrails:
    """Test cases for LakeraAIGuardrails class."""

    @pytest.fixture
    def lakera_guardrails(self) -> LakeraAIGuardrails:
        """Create a LakeraAIGuardrails instance for testing."""
        return LakeraAIGuardrails(api_key="test_lakera_ai_api_key")

    @pytest.fixture
    def sample_request(self) -> LakeraGuardrailRequest:
        """Create a sample LakeraGuardrailRequest for testing."""
        return LakeraGuardrailRequest(
            hello_rasa_user_id="test_user_id",
            hello_rasa_project_id="test_project_id",
            lakera_project_id="lakera_project_id",
            messages=[
                {"role": "user", "content": "Hello there."},
            ],
        )

    @pytest.mark.asyncio
    @patch.object(LakeraAIGuardrails, "_get_session")
    @pytest.mark.parametrize(
        "raw_response,expected_flagged,expected_detections_count",
        [
            # Test case 1: Request flagged with policy violations
            (
                {
                    "flagged": True,
                    "metadata": {"test": "metadata"},
                    "breakdown": [
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_prompt_attack_0",
                            "detector_type": "prompt_attack",
                            "detected": True,
                            "message_id": 0,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_profanity_0",
                            "detector_type": "moderated_content/profanity",
                            "detected": True,
                            "message_id": 0,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_pii_0",
                            "detector_type": "pii/iban_code",
                            "detected": False,
                            "message_id": None,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_pii_1",
                            "detector_type": "pii/address",
                            "detected": False,
                            "message_id": None,
                        },
                    ],
                },
                True,
                2,
            ),
            # Test case 2: Request not flagged
            (
                {
                    "flagged": False,
                    "metadata": {"test": "metadata"},
                    "breakdown": [
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_prompt_attack_1",
                            "detector_type": "prompt_attack",
                            "detected": False,
                            "message_id": None,
                        },
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_moderated_content_1",
                            "detector_type": "moderated_content/profanity",
                            "detected": False,
                            "message_id": None,
                        },
                    ],
                },
                False,
                0,
            ),
            # Test case 3: Request flagged but no breakdown provided
            (
                {
                    "flagged": True,
                    "metadata": {"test": "metadata"},
                },
                True,
                0,
            ),
            # Test case 4: Request flagged with custom detector
            (
                {
                    "flagged": True,
                    "metadata": {"test": "metadata"},
                    "breakdown": [
                        {
                            "project_id": "proj_123",
                            "policy_id": "policy_123",
                            "detector_id": "detector_custom_0",
                            "detector_type": "custom",
                            "detected": True,
                            "message_id": 0,
                        },
                    ],
                },
                True,
                1,
            ),
        ],
    )
    async def test_send_request_successful(
        self,
        mock_get_session: Mock,
        lakera_guardrails: LakeraAIGuardrails,
        sample_request: LakeraGuardrailRequest,
        raw_response: Dict[str, Any],
        expected_flagged: bool,
        expected_detections_count: int,
    ) -> None:
        """Test send_request method with various successful scenarios."""
        # Given
        # Create a real session but mock its post method
        session = aiohttp.ClientSession()
        mock_post = MagicMock(return_value=MockPostResponse(json_data=raw_response))
        session.post = mock_post  # type: ignore
        mock_get_session.return_value.__aenter__.return_value = session
        mock_get_session.return_value.__aexit__.return_value = None

        # When
        response = await lakera_guardrails.send_request(sample_request)

        # Then
        session.post.assert_called_once()
        call_args = session.post.call_args
        assert call_args[0][0] == lakera_guardrails.guard_endpoint
        assert call_args[1]["json"] == sample_request.to_json_payload()

        assert response.flagged == expected_flagged
        assert response.processing_time_ms is not None
        assert response.processing_time_ms >= 0

        if expected_detections_count > 0:
            assert response.detections is not None
            assert len(response.detections) == expected_detections_count
        else:
            assert len(response.detections) == 0

    @pytest.mark.asyncio
    @patch.object(LakeraAIGuardrails, "_get_session")
    @pytest.mark.parametrize(
        "status_code,error_text,expected_exception",
        [  # type: ignore[misc]
            # Test case 1: HTTP 400 error
            (
                400,
                "Bad Request",
                GuardrailsError,
            ),
            # Test case 2: HTTP 401 error
            (
                401,
                "Unauthorized",
                GuardrailsError,
            ),
            # Test case 3: HTTP 500 error
            (
                500,
                "Internal Server Error",
                GuardrailsError,
            ),
        ],
    )
    async def test_send_request_http_errors(
        self,
        mock_get_session: Mock,
        lakera_guardrails: LakeraAIGuardrails,
        sample_request: LakeraGuardrailRequest,
        status_code: int,
        error_text: str,
        expected_exception: Type[Exception],
    ) -> None:
        """Test send_request method with HTTP error responses."""
        # Given
        session = aiohttp.ClientSession()
        mock_post = MagicMock(
            return_value=MockPostResponse(status=status_code, text=error_text)
        )
        session.post = mock_post  # type: ignore

        mock_get_session.return_value.__aenter__.return_value = session
        mock_get_session.return_value.__aexit__.return_value = None

        # When / Then
        with pytest.raises(expected_exception):
            await lakera_guardrails.send_request(sample_request)

    @pytest.mark.asyncio
    @patch.object(LakeraAIGuardrails, "_get_session")
    async def test_send_request_timeout_error(
        self,
        mock_get_session: Mock,
        lakera_guardrails: LakeraAIGuardrails,
        sample_request: LakeraGuardrailRequest,
    ) -> None:
        """Test send_request method with timeout error."""
        # Given
        session = aiohttp.ClientSession()
        mock_post = MagicMock(side_effect=asyncio.TimeoutError())
        session.post = mock_post  # type: ignore
        mock_get_session.return_value.__aenter__.return_value = session
        mock_get_session.return_value.__aexit__.return_value = None

        # When / Then
        with pytest.raises(GuardrailsError):
            await lakera_guardrails.send_request(sample_request)

    @pytest.mark.asyncio
    @patch.object(LakeraAIGuardrails, "_get_session")
    async def test_send_request_json_parsing_error(
        self,
        mock_get_session: Mock,
        lakera_guardrails: LakeraAIGuardrails,
        sample_request: LakeraGuardrailRequest,
    ) -> None:
        """Test send_request method with JSON parsing error."""
        # Given
        session = aiohttp.ClientSession()
        mock_post = MagicMock(return_value=MockPostResponse())
        session.post = mock_post  # type: ignore

        # Override the json method to raise an error
        mock_post.return_value.response.json = AsyncMock(
            side_effect=ValueError("Invalid JSON")
        )

        mock_get_session.return_value.__aenter__.return_value = session
        mock_get_session.return_value.__aexit__.return_value = None

        # When / Then
        with pytest.raises(ValueError):
            await lakera_guardrails.send_request(sample_request)

    def test_lakera_client_proxy_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proxy = "https://hello-llm-proxy.example"
        license_token = "rasa-license-jwt"
        monkeypatch.setattr(config, "PROXY_URL", proxy)
        monkeypatch.setattr(config, "LAKERA_BASE_URL", f"{proxy}/guardrails")
        monkeypatch.setattr(config, "RASA_PRO_LICENSE", license_token)

        guardrails = LakeraAIGuardrails(api_key="lakera-direct-key")

        expected_base = f"{proxy}/guardrails"
        assert guardrails.guard_endpoint == f"{expected_base}/guard"
        assert guardrails.guard_results_endpoint == f"{expected_base}/guard/results"

        headers = guardrails._get_headers()
        assert headers.get("Authorization") == f"Bearer {license_token}"
