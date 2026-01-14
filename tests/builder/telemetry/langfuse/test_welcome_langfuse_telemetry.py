"""Tests for general Langfuse telemetry utilities."""

from unittest.mock import MagicMock, patch

import pytest

from rasa.builder.telemetry.langfuse.welcome_langfuse_telemetry import (
    WelcomeMessageGenerationLangfuseTelemetry,
)


class TestWelcomeMessageGenerationLangfuseTelemetry:
    """Test cases for WelcomeMessageGenerationLangfuseTelemetry class."""

    @pytest.fixture
    def mock_langfuse_client(self):
        """Create a mock Langfuse client."""
        mock_client = MagicMock()
        mock_client.update_current_span = MagicMock()
        return mock_client

    @patch("langfuse.get_client")
    def test_update_welcome_message_generation_input(
        self, mock_get_client, mock_langfuse_client
    ):
        """Test updating welcome message generation input."""
        mock_get_client.return_value = mock_langfuse_client
        flows = """{'hello_world_greeting': {'steps':
[{'action': 'utter_hello_world', 'next': 'END'}],
'description': "Responds to user greetings by saying 'Hello World'."}}"""
        prompt = """Generate sample prompts to trigger the generated flows,
by fitting the flow name and description:
The prompts should:
- Be in imperative mood
- Be specific per flow
- Be under 72 characters

Generated flows:
{'hello_world_greeting': {'steps':
[{'action': 'utter_hello_world', 'next': 'END'}],
'description': "Responds to user greetings by saying 'Hello World'."}}

Chose most important flows.
Generate only one prompt per flow
Return max 3 prompts ( at least one ) in a bullet point list: e.g.
- *Why is my internet slow?*
- *How do i reboot my router?*"""

        WelcomeMessageGenerationLangfuseTelemetry.update_welcome_message_generation_input(
            flows=flows,
            prompt=prompt,
        )

        mock_langfuse_client.update_current_span.assert_called_once_with(
            input={
                "flows": flows,
                "prompt": prompt,
            }
        )

    @patch("langfuse.get_client")
    def test_update_welcome_message_generation_output(
        self, mock_get_client, mock_langfuse_client
    ):
        """Test updating welcome message generation output."""
        mock_get_client.return_value = mock_langfuse_client
        response_content = """- *Say hello to me*
- *Show me the top FAQs about Rasa and CALM*
- *Transfer me to a human agent*"""
        welcome_message = """👋 Welcome to Hello Rasa!

I'm your **Copilot** — here to help you explore Rasa and start customizing your agent.
You can ask me how your agent works, how to add new skills,
or how to connect integrations.

Your custom agent has been created and trained successfully.

### ▶️ **First step: try it out**
Ask your agent (in the chat preview on the right):
- *Say hello to me*
- *Show me the top FAQs about Rasa and CALM*
- *Transfer me to a human agent*
"""

        WelcomeMessageGenerationLangfuseTelemetry.update_welcome_message_generation_output(
            response_content=response_content,
            welcome_message=welcome_message,
        )

        mock_langfuse_client.update_current_span.assert_called_once_with(
            output={
                "response_content": response_content,
                "welcome_message": welcome_message,
            }
        )
