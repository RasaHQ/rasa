"""Tests for the copilot template endpoint."""

import json
from unittest.mock import Mock, patch

import pytest

from rasa.builder.service import get_copilot_internal_message_template


class TestCopilotTemplateEndpoint:
    """Test cases for the copilot template endpoint."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service with copilot internal message formatter."""
        mock_service = Mock()
        mock_templated_message_provider = Mock()
        mock_service.copilot_templated_message_provider = (
            mock_templated_message_provider
        )
        return mock_service

    @pytest.mark.asyncio
    async def test_get_valid_template(self, mock_llm_service: Mock):
        """Test getting a valid template for a template name."""
        # Given
        expected_template = (
            "The assistant training failed. Your task is to "
            "analyze provided error logs\n"
            "and help me fix the issue.\n"
        )
        mock_request = Mock()
        template_name = "training_error_log_analysis"

        mock_templates_func = Mock(return_value={template_name: expected_template})

        # When
        with patch(
            "rasa.builder.service.copilot_internal_message_templates",
            mock_templates_func,
        ):
            response = await get_copilot_internal_message_template(
                mock_request, template_name
            )

        # Then
        assert response.status == 200
        response_data = json.loads(response.body.decode())
        assert "template" in response_data
        assert "template_name" in response_data
        assert response_data["template"] == expected_template
        assert response_data["template_name"] == template_name

        # Verify the formatter was called correctly
        mock_templates_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_nonexistent_template(self, mock_llm_service: Mock):
        """Test getting a template for a template name that doesn't exist."""
        # Mock request
        mock_request = Mock()

        # Test with template name that has no template
        template_name = "copilot"

        mock_templates_func = Mock(return_value={})

        with patch(
            "rasa.builder.service.copilot_internal_message_templates",
            mock_templates_func,
        ):
            response = await get_copilot_internal_message_template(
                mock_request, template_name
            )

        # Verify the response
        assert response.status == 404
        response_data = json.loads(response.body.decode())
        assert "error" in response_data
        assert response_data["error"] == "Template not found"
        assert "details" in response_data
        assert response_data["details"]["template_name"] == template_name
