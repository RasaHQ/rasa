"""Tests for MCP bot interaction tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder.copilot.mcp_server.models import (
    TalkToAssistantResponse,
    TrackerContextOutput,
)
from rasa.builder.copilot.mcp_server.tools.bot_interaction import (
    _get_tracker_context,
    talk_to_assistant,
)
from rasa.cli.tools.constants import DEFAULT_RASA_SERVER_URL

RASA_SERVER_URL = DEFAULT_RASA_SERVER_URL


class TestTalkToAssistant:
    """Test talk_to_assistant function."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_talk_to_assistant_success(self, mock_client_session):
        """Test successful conversation with assistant."""
        # Mock responses
        bot_responses = [{"text": "Hello! How can I help you?"}]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=bot_responses)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_ctx)

        mock_tracker_ctx = MagicMock()
        mock_tracker_response = MagicMock()
        mock_tracker_response.status = 200
        mock_tracker_response.json = AsyncMock(
            return_value={"conversation_turns": [], "current_state": {}}
        )
        mock_tracker_ctx.__aenter__ = AsyncMock(return_value=mock_tracker_response)
        mock_tracker_ctx.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_tracker_ctx)

        mock_client_session.return_value = mock_session

        result = await talk_to_assistant(["Hello!"], RASA_SERVER_URL)

        assert isinstance(result, TalkToAssistantResponse)
        assert result.success is True
        assert result.message_count == 1
        assert len(result.conversation) == 1
        assert result.conversation[0].user_message == "Hello!"
        assert result.conversation[0].bot_responses == bot_responses

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_talk_to_assistant_multiple_messages(self, mock_client_session):
        """Test conversation with multiple messages."""
        messages = ["Hello", "What can you do?", "Help me with flows"]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[{"text": "Response"}])

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_ctx)

        mock_tracker_ctx = MagicMock()
        mock_tracker_response = MagicMock()
        mock_tracker_response.status = 200
        mock_tracker_response.json = AsyncMock(
            return_value={"conversation_turns": [], "current_state": {}}
        )
        mock_tracker_ctx.__aenter__ = AsyncMock(return_value=mock_tracker_response)
        mock_tracker_ctx.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_tracker_ctx)

        mock_client_session.return_value = mock_session

        result = await talk_to_assistant(messages, RASA_SERVER_URL)

        assert result.success is True
        assert result.message_count == 3
        assert len(result.conversation) == 3

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_talk_to_assistant_request_failure(self, mock_client_session):
        """Test conversation when request fails."""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_ctx)

        mock_client_session.return_value = mock_session

        result = await talk_to_assistant(["Hello!"], RASA_SERVER_URL)

        assert result.success is False
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_talk_to_assistant_connection_error(self):
        """Test conversation when connection fails."""
        import aiohttp
        from aioresponses import aioresponses

        with aioresponses() as m:
            # Mock webhook to raise connection error
            m.post(
                f"{RASA_SERVER_URL}/webhooks/rest/webhook",
                exception=aiohttp.ClientError("Connection refused"),
            )

            result = await talk_to_assistant(["Hello!"], RASA_SERVER_URL)

            assert result.success is False
            assert "Connection error" in result.error

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_talk_to_assistant_empty_messages(self, mock_client_session):
        """Test conversation with empty messages list."""
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_tracker_ctx = MagicMock()
        mock_tracker_response = MagicMock()
        mock_tracker_response.status = 200
        mock_tracker_response.json = AsyncMock(
            return_value={"conversation_turns": [], "current_state": {}}
        )
        mock_tracker_ctx.__aenter__ = AsyncMock(return_value=mock_tracker_response)
        mock_tracker_ctx.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_tracker_ctx)

        mock_client_session.return_value = mock_session

        result = await talk_to_assistant([], RASA_SERVER_URL)

        assert result.success is True
        assert result.message_count == 0
        assert len(result.conversation) == 0

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_uses_explicit_url(self, mock_client_session):
        """Verify talk_to_assistant uses the provided URL for the webhook."""
        bot_responses = [{"text": "Hi!"}]
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=bot_responses)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_post_ctx = MagicMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_tracker_response = MagicMock()
        mock_tracker_response.status = 200
        mock_tracker_response.json = AsyncMock(
            return_value={"conversation_turns": [], "current_state": {}}
        )
        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_tracker_response)
        mock_get_ctx.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        mock_client_session.return_value = mock_session

        custom_url = "http://my-custom-server:9999"
        result = await talk_to_assistant(["Hello!"], custom_url)

        assert result.success is True
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == f"{custom_url}/webhooks/rest/webhook"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_strips_trailing_slash_from_url(self, mock_client_session):
        """Verify trailing slash is stripped from the URL."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[{"text": "Hi!"}])

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_post_ctx = MagicMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_tracker_response = MagicMock()
        mock_tracker_response.status = 200
        mock_tracker_response.json = AsyncMock(
            return_value={"conversation_turns": [], "current_state": {}}
        )
        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock(return_value=mock_tracker_response)
        mock_get_ctx.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get_ctx)

        mock_client_session.return_value = mock_session

        result = await talk_to_assistant(["Hello!"], "http://localhost:5005/")

        assert result.success is True
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:5005/webhooks/rest/webhook"


class TestGetTrackerContext:
    """Test _get_tracker_context function."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_get_tracker_context_success(self, mock_client_session):
        """Test successful tracker context retrieval."""
        tracker_data = {
            "conversation_turns": [{"user": "Hello", "bot": "Hi there!"}],
            "current_state": {"active_flow": "greet_flow"},
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=tracker_data)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_ctx)

        mock_client_session.return_value = mock_session

        result = await _get_tracker_context("test-session-123", RASA_SERVER_URL)

        assert isinstance(result, TrackerContextOutput)
        assert len(result.conversation_turns) == 1
        assert result.current_state["active_flow"] == "greet_flow"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_get_tracker_context_not_found(self, mock_client_session):
        """Test tracker context retrieval when session not found."""
        mock_response = MagicMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_ctx)

        mock_client_session.return_value = mock_session

        result = await _get_tracker_context("nonexistent-session", RASA_SERVER_URL)

        assert result is None

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_get_tracker_context_connection_error(self, mock_client_session):
        """Test tracker context retrieval when connection fails."""
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(side_effect=Exception("Connection failed"))

        mock_client_session.return_value = mock_session

        result = await _get_tracker_context("test-session", RASA_SERVER_URL)

        assert result is None
