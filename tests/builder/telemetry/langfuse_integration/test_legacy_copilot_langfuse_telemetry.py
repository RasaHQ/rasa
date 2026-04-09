import asyncio
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import Mock, patch

from rasa.builder.copilot.models import UsageStatistics
from rasa.builder.telemetry.langfuse_integration.legacy_copilot_langfuse_telemetry import (  # noqa: E501
    LegacyCopilotLangfuseTelemetry,
)


class TestLegacyCopilotLangfuseTelemetry:
    """Test class for LegacyCopilotLangfuseTelemetry streaming generation methods."""

    @patch(
        "rasa.builder.telemetry.langfuse_integration.legacy_copilot_langfuse_telemetry.langfuse.get_client"
    )
    def test_trace_legacy_copilot_streaming_generation_decorator(
        self, mock_get_client: Mock
    ) -> None:
        """Test the trace_legacy_copilot_streaming_generation decorator."""
        # Given
        copilot_input_messages = [{"role": "user", "content": "Hello"}]

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_generation = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_generation)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_client.start_as_current_generation.return_value = mock_context_manager

        # Create a mock Copilot instance
        mock_copilot = Mock()
        mock_copilot.__class__.__name__ = "LegacyCopilot"
        mock_copilot.llm_config = {"temperature": 0.7, "model": "gpt-4"}
        mock_copilot.usage_statistics = UsageStatistics(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4",
        )

        # Create the decorator and apply it to the mock streaming function
        decorator = LegacyCopilotLangfuseTelemetry.trace_streaming_generation

        async def mock_streaming_function(
            self: Any, messages: List[Dict[str, Any]]
        ) -> AsyncGenerator[str, None]:
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        decorated_function = decorator(mock_streaming_function)

        # When
        async def run_test() -> List[str]:
            result: List[str] = []
            async for chunk in decorated_function(mock_copilot, copilot_input_messages):
                result.append(chunk)
            return result

        result: List[str] = asyncio.run(run_test())

        # Then
        assert result == ["chunk1", "chunk2", "chunk3"]

        mock_client.start_as_current_generation.assert_called_once()
        assert (
            mock_client.start_as_current_generation.call_args[1]["input"]["messages"]
            == copilot_input_messages
        )

        # The update on the span is called twice:
        # - Once for the model parameters and output
        # - Once for the usage statistics
        assert mock_generation.update.call_count == 2

        assert (
            mock_generation.update.call_args_list[0].kwargs["model_parameters"]
            == mock_copilot.llm_config
        )
        assert (
            mock_generation.update.call_args_list[0].kwargs["output"]
            == "chunk1chunk2chunk3"
        )
        assert "usage_details" in mock_generation.update.call_args_list[1].kwargs
        assert "cost_details" in mock_generation.update.call_args_list[1].kwargs
        assert "model" in mock_generation.update.call_args_list[1].kwargs
