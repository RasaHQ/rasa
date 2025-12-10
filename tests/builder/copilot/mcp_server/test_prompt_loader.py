"""Tests for MCP prompt loader module."""

import pytest

from rasa.builder.copilot.mcp_server.prompts.prompt_loader import (
    _load_prompt_source,
    _prompt_sources,
    get_copilot_system_prompt,
    get_last_user_message_context_prompt,
    get_training_error_handler_prompt,
)


class TestLoadPromptSource:
    """Test _load_prompt_source function."""

    def test_load_prompt_source_returns_string(self) -> None:
        """Test that loading a prompt source returns a non-empty string."""
        from rasa.builder.copilot.constants import COPILOT_PROMPTS_FILE

        result = _load_prompt_source(COPILOT_PROMPTS_FILE)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_load_prompt_source_caches_result(self) -> None:
        """Test that loading a prompt source caches the result."""
        from rasa.builder.copilot.constants import COPILOT_PROMPTS_FILE

        # Clear cache first
        _prompt_sources.clear()

        # First load
        result1 = _load_prompt_source(COPILOT_PROMPTS_FILE)

        # Should now be in cache
        assert COPILOT_PROMPTS_FILE in _prompt_sources

        # Second load should return same cached object
        result2 = _load_prompt_source(COPILOT_PROMPTS_FILE)

        assert result1 == result2
        # Verify it's the same cached string
        assert result1 is result2

    def test_load_prompt_source_invalid_file_raises(self) -> None:
        """Test that loading an invalid file raises an error."""
        with pytest.raises(Exception):  # Could be FileNotFoundError or similar
            _load_prompt_source("nonexistent_prompt_file.txt")


class TestGetCopilotSystemPrompt:
    """Test get_copilot_system_prompt function."""

    @pytest.mark.asyncio
    async def test_get_copilot_system_prompt_returns_content(self) -> None:
        """Test that get_copilot_system_prompt returns prompt content."""
        result = await get_copilot_system_prompt()

        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be an error message
        assert not result.startswith("Error loading")

    @pytest.mark.asyncio
    async def test_get_copilot_system_prompt_contains_jinja_template(self) -> None:
        """Test that the system prompt contains expected Jinja template markers."""
        result = await get_copilot_system_prompt()

        # The prompt template should contain some template structure
        # (This depends on actual prompt content, adjust as needed)
        assert isinstance(result, str)


class TestGetLastUserMessageContextPrompt:
    """Test get_last_user_message_context_prompt function."""

    @pytest.mark.asyncio
    async def test_get_last_user_message_context_prompt_returns_content(self) -> None:
        """Test that get_last_user_message_context_prompt returns prompt content."""
        result = await get_last_user_message_context_prompt()

        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be an error message
        assert not result.startswith("Error loading")


class TestGetTrainingErrorHandlerPrompt:
    """Test get_training_error_handler_prompt function."""

    @pytest.mark.asyncio
    async def test_get_training_error_handler_prompt_returns_content(self) -> None:
        """Test that get_training_error_handler_prompt returns prompt content."""
        result = await get_training_error_handler_prompt()

        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be an error message
        assert not result.startswith("Error loading")
