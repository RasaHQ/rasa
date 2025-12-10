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


class TestTalkToAssistant:
    """Test talk_to_assistant function."""

    @pytest.mark.asyncio
    async def test_talk_to_assistant_success(self):
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

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await talk_to_assistant(["Hello!"])

            assert isinstance(result, TalkToAssistantResponse)
            assert result.success is True
            assert result.message_count == 1
            assert len(result.conversation) == 1
            assert result.conversation[0].user_message == "Hello!"
            assert result.conversation[0].bot_responses == bot_responses

    @pytest.mark.asyncio
    async def test_talk_to_assistant_multiple_messages(self):
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

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await talk_to_assistant(messages)

            assert result.success is True
            assert result.message_count == 3
            assert len(result.conversation) == 3

    @pytest.mark.asyncio
    async def test_talk_to_assistant_request_failure(self):
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

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await talk_to_assistant(["Hello!"])

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
                "http://localhost:5002/webhooks/rest/webhook",
                exception=aiohttp.ClientError("Connection refused"),
            )

            result = await talk_to_assistant(["Hello!"])

            assert result.success is False
            assert "Connection error" in result.error

    @pytest.mark.asyncio
    async def test_talk_to_assistant_empty_messages(self):
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

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await talk_to_assistant([])

            assert result.success is True
            assert result.message_count == 0
            assert len(result.conversation) == 0


class TestGetTrackerContext:
    """Test _get_tracker_context function."""

    @pytest.mark.asyncio
    async def test_get_tracker_context_success(self):
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

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _get_tracker_context("test-session-123")

            assert isinstance(result, TrackerContextOutput)
            assert len(result.conversation_turns) == 1
            assert result.current_state["active_flow"] == "greet_flow"

    @pytest.mark.asyncio
    async def test_get_tracker_context_not_found(self):
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

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _get_tracker_context("nonexistent-session")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_tracker_context_connection_error(self):
        """Test tracker context retrieval when connection fails."""
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session.get = MagicMock(side_effect=Exception("Connection failed"))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _get_tracker_context("test-session")

            assert result is None
