import json
from dataclasses import fields
from unittest.mock import Mock

import pytest
from litellm.utils import ChatCompletionMessageToolCall
from pydantic import ValidationError

from rasa.shared.exceptions import LLMToolResponseDecodeError
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall


class TestLLMResponse:
    def test_ensure_llm_response_with_llm_response(
        self, llm_response_object: LLMResponse
    ):
        result = LLMResponse.ensure_llm_response(llm_response_object)
        assert result == llm_response_object

    def test_ensure_llm_response_with_string(self):
        response = "test_response"
        result = LLMResponse.ensure_llm_response(response)
        class_fields = fields(LLMResponse)
        empty_names = [field.name for field in class_fields if field.name != "choices"]
        for field in empty_names:
            assert getattr(result, field) is None
        assert result.choices == [response]


class TestLLMToolCall:
    """Test cases for LLMToolCall class."""

    def test_llm_tool_call_creation(self):
        """Test direct instantiation of LLMToolCall."""
        tool_call = LLMToolCall(
            id="call_123",
            tool_name="test_tool",
            tool_args={"param1": "value1", "param2": 42},
            type="function",
        )

        assert tool_call.id == "call_123"
        assert tool_call.tool_name == "test_tool"
        assert tool_call.tool_args == {"param1": "value1", "param2": 42}
        assert tool_call.type == "function"

    def test_llm_tool_call_default_type(self):
        """Test that type defaults to 'function'."""
        tool_call = LLMToolCall(
            id="call_123", tool_name="test_tool", tool_args={"param": "value"}
        )

        assert tool_call.type == "function"

    def test_llm_tool_call_from_dict(self):
        """Test creating LLMToolCall from dictionary."""
        data = {
            "id": "call_456",
            "tool_name": "another_tool",
            "tool_args": {"key": "value", "number": 123},
            "type": "function",
        }

        tool_call = LLMToolCall.from_dict(data)

        assert tool_call.id == "call_456"
        assert tool_call.tool_name == "another_tool"
        assert tool_call.tool_args == {"key": "value", "number": 123}
        assert tool_call.type == "function"

    def test_llm_tool_call_from_dict_minimal(self):
        """Test creating LLMToolCall from minimal dictionary."""
        data = {"id": "call_789", "tool_name": "minimal_tool", "tool_args": {}}

        tool_call = LLMToolCall.from_dict(data)

        assert tool_call.id == "call_789"
        assert tool_call.tool_name == "minimal_tool"
        assert tool_call.tool_args == {}
        assert tool_call.type == "function"  # default value

    def test_llm_tool_call_from_litellm_valid(self):
        """Test creating LLMToolCall with valid JSON."""
        # Create mock LiteLLM tool call
        mock_function = Mock()
        mock_function.name = "test_function"
        mock_function.arguments = (
            '{"param1": "value1", "param2": 42, "nested": {"key": "value"}}'
        )

        mock_tool_call = Mock(spec=ChatCompletionMessageToolCall)
        mock_tool_call.id = "call_litellm_123"
        mock_tool_call.function = mock_function
        mock_tool_call.type = "function"

        tool_call = LLMToolCall.from_litellm(mock_tool_call)

        assert tool_call.id == "call_litellm_123"
        assert tool_call.tool_name == "test_function"
        assert tool_call.tool_args == {
            "param1": "value1",
            "param2": 42,
            "nested": {"key": "value"},
        }
        assert tool_call.type == "function"

    @pytest.mark.parametrize(
        "tool_name,invalid_arguments,expected_error_fragment",
        [
            (
                "test_function",
                '{"invalid": json, "missing": quote}',
                '`{"invalid": json, "missing": quote}`',
            ),
            (
                "another_function",
                '{"invalid": "json", "missing"',
                '`{"invalid": "json", "missing"',
            ),
            ("malformed_function", "not json at all", "`not json at all`"),
        ],
    )
    def test_llm_tool_call_from_litellm_invalid_json(
        self, tool_name: str, invalid_arguments: str, expected_error_fragment: str
    ):
        """Test creating LLMToolCall from LiteLLM with invalid JSON arguments."""
        # Create mock LiteLLM tool call with invalid JSON
        mock_function = Mock()
        mock_function.name = tool_name
        mock_function.arguments = invalid_arguments

        mock_tool_call = Mock(spec=ChatCompletionMessageToolCall)
        mock_tool_call.id = f"call_litellm_{tool_name}"
        mock_tool_call.function = mock_function
        mock_tool_call.type = "function"

        with pytest.raises(LLMToolResponseDecodeError) as exc_info:
            LLMToolCall.from_litellm(mock_tool_call)

        assert f"Invalid arguments for tool call - `{tool_name}`" in str(exc_info.value)
        assert expected_error_fragment in str(exc_info.value)
        assert isinstance(exc_info.value.original_exception, json.JSONDecodeError)

    def test_llm_tool_call_from_litellm_complex_arguments(self):
        """Test creating LLMToolCall from LiteLLM with complex nested arguments."""
        complex_args = {
            "string_param": "test",
            "number_param": 123,
            "boolean_param": True,
            "array_param": [1, 2, 3],
            "nested_object": {"inner_key": "inner_value", "inner_number": 456},
        }

        mock_function = Mock()
        mock_function.name = "complex_function"
        mock_function.arguments = json.dumps(complex_args)

        mock_tool_call = Mock(spec=ChatCompletionMessageToolCall)
        mock_tool_call.id = "call_litellm_complex"
        mock_tool_call.function = mock_function
        mock_tool_call.type = "function"

        tool_call = LLMToolCall.from_litellm(mock_tool_call)

        assert tool_call.id == "call_litellm_complex"
        assert tool_call.tool_name == "complex_function"
        assert tool_call.tool_args == complex_args
        assert tool_call.type == "function"

    def test_llm_tool_call_validation_error(self):
        """Test that LLMToolCall raises ValidationError for invalid data."""
        with pytest.raises(ValidationError):
            LLMToolCall(
                id="call_123",
                # Missing required tool_name
                tool_args={"param": "value"},
            )
