from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, Mock

import pytest
from litellm import ModelResponse
from litellm.utils import Usage
from pytest import MonkeyPatch

from rasa.shared.exceptions import (
    LLMToolResponseDecodeError,
    ProviderClientAPIException,
)
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.llm.llm_response import LLMResponse


class TestLiteLLMClient(_BaseLiteLLMClient):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "_BaseLiteLLMClient":
        return cls()

    @property
    def config(self) -> dict:
        return {}

    @property
    def _litellm_model_name(self) -> str:
        return "openai/test_model"

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        return {"test_parameter": "test_value"}


class TestBaseLLMClient:
    @pytest.fixture
    def client(self) -> TestLiteLLMClient:
        return TestLiteLLMClient()

    @pytest.fixture
    def litellm_model_response(self, client: LLMClient) -> ModelResponse:
        return ModelResponse(
            id="id123",
            choices=[
                {"message": {"content": "Hello from LiteLLM!", "role": "assistant"}}
            ],
            created=1234567890,
            model="test_model",
            object="text_completion",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    @pytest.fixture
    def mock_completion(
        self, monkeypatch: MonkeyPatch, litellm_model_response: ModelResponse
    ) -> Mock:
        # Create a mock object
        mock = Mock(return_value=litellm_model_response)
        # Replace the 'completion' function in its module with the mock
        monkeypatch.setattr(
            "rasa.shared.providers.llm._base_litellm_client.completion", mock
        )
        return mock

    @pytest.fixture
    def mock_acompletion(
        self, monkeypatch: MonkeyPatch, litellm_model_response: ModelResponse
    ) -> AsyncMock:
        # Create a mock object
        mock = AsyncMock(return_value=litellm_model_response)
        # Replace the 'completion' function in its module with the mock
        monkeypatch.setattr(
            "rasa.shared.providers.llm._base_litellm_client.acompletion", mock
        )
        return mock

    def test_format_response(
        self, client: TestLiteLLMClient, litellm_model_response: ModelResponse
    ):
        # When
        formated_response = client._format_response(litellm_model_response)

        # Then
        assert formated_response.id == "id123"
        assert formated_response.created == 1234567890
        assert formated_response.model == "test_model"
        assert formated_response.usage.prompt_tokens == 10
        assert formated_response.usage.completion_tokens == 10
        assert formated_response.usage.total_tokens == 20

    @pytest.mark.parametrize("usage", (None, Usage()))
    def test_format_response_with_uninitialized_usage(
        self, usage: Optional[Usage], client: TestLiteLLMClient
    ):
        # Given
        model_response = ModelResponse(
            id="id123",
            choices=[
                {"message": {"content": "Hello from LiteLLM!", "role": "assistant"}}
            ],
            created=1234567890,
            model="test_model",
            object="text_completion",
            usage=usage,
        )

        # When
        formated_response = client._format_response(model_response)

        # Then
        assert formated_response.id == "id123"
        assert formated_response.created == 1234567890
        assert formated_response.model == "test_model"
        assert formated_response.usage is not None
        assert formated_response.usage.prompt_tokens == 0
        assert formated_response.usage.completion_tokens == 0
        assert formated_response.usage.total_tokens == 0

    def test_conforms_to_protocol(self, client):
        assert isinstance(client, LLMClient)

    @pytest.mark.parametrize(
        "test_prompt",
        [
            # Send the prompt as a list
            ["Hello, this is a test prompt."],
            # Send the prompt as a str
            "Hello, this is a test prompt.",
        ],
    )
    def test_completion(
        self,
        test_prompt: str,
        client: TestLiteLLMClient,
        mock_completion: Mock,
    ):
        # Given
        prompt_content = test_prompt if isinstance(test_prompt, str) else test_prompt[0]

        # When
        response = client.completion(test_prompt)

        # Then
        mock_completion.assert_called_once_with(
            messages=[{"content": prompt_content, "role": "user"}],
            model=client._litellm_model_name,
            drop_params=False,
            test_parameter="test_value",
        )
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Hello from LiteLLM!"]

    def test_completion_encounters_an_error(
        self, client: LLMClient, mock_completion: Mock
    ) -> None:
        mock_completion.side_effect = Exception("API exception raised!")
        with pytest.raises(ProviderClientAPIException):
            client.completion(["test message"])

    @pytest.mark.parametrize(
        "test_prompt",
        [
            # Send the prompt as a list
            ["Hello, this is a test prompt."],
            # Send the prompt as a str
            "Hello, this is a test prompt.",
        ],
    )
    async def test_acompletion(
        self,
        test_prompt: Union[List[str], str],
        client: TestLiteLLMClient,
        mock_acompletion: Mock,
    ):
        # Given
        prompt_content = test_prompt if isinstance(test_prompt, str) else test_prompt[0]

        # When
        response = await client.acompletion(test_prompt)

        # Then
        mock_acompletion.assert_called_once_with(
            messages=[{"content": prompt_content, "role": "user"}],
            model=client._litellm_model_name,
            drop_params=False,
            test_parameter="test_value",
        )
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Hello from LiteLLM!"]

    async def test_acompletion_encounters_an_error(
        self, client: LLMClient, mock_acompletion: AsyncMock
    ):
        mock_acompletion.side_effect = Exception("API exception raised!")
        with pytest.raises(ProviderClientAPIException):
            await client.acompletion(["test message"])

    @pytest.mark.parametrize(
        "test_prompt",
        [
            # Send the preformatted prompt as a list
            ([{"content": "Hello, this is a test prompt.", "role": "user"}]),
            ([{"content": "Hello, this is a test prompt.", "role": "system"}]),
            (
                [
                    {"content": "Hello, this is a test prompt.", "role": "user"},
                    {"content": "Hello, this is a test prompt.", "role": "system"},
                ]
            ),
        ],
    )
    def test_completion_with_preformatted_messages(
        self, test_prompt: str, client: TestLiteLLMClient, mock_completion: Mock
    ):
        # When
        response = client.completion(test_prompt)

        # Then
        mock_completion.assert_called_once_with(
            messages=test_prompt,
            model=client._litellm_model_name,
            drop_params=False,
            test_parameter="test_value",
        )
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Hello from LiteLLM!"]

    @pytest.mark.parametrize(
        "test_prompt",
        [
            # Send the preformatted prompt as a list
            ([{"content": "Hello, this is a test prompt.", "role": "user"}]),
            ([{"content": "Hello, this is a test prompt.", "role": "system"}]),
            (
                [
                    {"content": "Hello, this is a test prompt.", "role": "user"},
                    {"content": "Hello, this is a test prompt.", "role": "system"},
                ]
            ),
        ],
    )
    async def test_acompletion_with_preformatted_messages(
        self,
        test_prompt: Union[List[str], str],
        client: TestLiteLLMClient,
        mock_acompletion: Mock,
    ):
        # When
        response = await client.acompletion(test_prompt)

        # Then
        mock_acompletion.assert_called_once_with(
            messages=test_prompt,
            model=client._litellm_model_name,
            drop_params=False,
            test_parameter="test_value",
        )
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Hello from LiteLLM!"]

    def test_completion_with_kwargs(
        self, client: TestLiteLLMClient, mock_completion: Mock
    ):
        # Given
        test_prompt = "Hello, this is a test prompt."

        # When
        response = client.completion(test_prompt, tools=["tool1", "tool2"])

        # Then
        expected_args = {
            "messages": [{"content": test_prompt, "role": "user"}],
            "model": client._litellm_model_name,
            "drop_params": False,
            "test_parameter": "test_value",
            "tools": ["tool1", "tool2"],
        }
        mock_completion.assert_called_once_with(**expected_args)
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Hello from LiteLLM!"]

    async def test_acompletion_with_kwargs(
        self, client: TestLiteLLMClient, mock_acompletion: Mock
    ):
        # Given
        test_prompt = "Hello, this is a test prompt."

        # When
        response = await client.acompletion(test_prompt, tools=["tool1", "tool2"])

        # Then
        expected_args = {
            "messages": [{"content": test_prompt, "role": "user"}],
            "model": client._litellm_model_name,
            "drop_params": False,
            "test_parameter": "test_value",
            "tools": ["tool1", "tool2"],
        }
        mock_acompletion.assert_called_once_with(**expected_args)
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Hello from LiteLLM!"]

    @pytest.fixture
    def litellm_model_response_with_tool_calls(self) -> ModelResponse:
        """Create a ModelResponse with tool calls."""
        tool_call = {
            "id": "call_litellm_123",
            "function": {
                "name": "test_function",
                "arguments": '{"param1": "value1", "param2": 42, "nested": {"key": "value"}}',  # noqa: E501
            },
            "type": "function",
        }

        return ModelResponse(
            id="id123",
            choices=[
                {
                    "message": {
                        "content": "Hello from LiteLLM!",
                        "role": "assistant",
                        "tool_calls": [tool_call],
                    }
                }
            ],
            created=1234567890,
            model="test_model",
            object="text_completion",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    @pytest.fixture
    def mock_tool_acompletion(
        self,
        monkeypatch: MonkeyPatch,
        litellm_model_response_with_tool_calls: ModelResponse,
    ) -> AsyncMock:
        # Create a mock object
        mock = AsyncMock(return_value=litellm_model_response_with_tool_calls)
        # Replace the 'acompletion' function in its module with the mock
        monkeypatch.setattr(
            "rasa.shared.providers.llm._base_litellm_client.acompletion", mock
        )
        return mock

    async def test_acompletion_with_tool_calls_valid(
        self, client: TestLiteLLMClient, mock_tool_acompletion: Mock
    ):
        """Test acompletion with tool calls using the mock_tool_acompletion fixture."""
        # Given
        test_prompt = "Hello, this is a test prompt with tools."

        # When
        response = await client.acompletion(test_prompt, tools=["test_tool"])

        # Then
        expected_args = {
            "messages": [{"content": test_prompt, "role": "user"}],
            "model": client._litellm_model_name,
            "drop_params": False,
            "test_parameter": "test_value",
            "tools": ["test_tool"],
        }
        mock_tool_acompletion.assert_called_once_with(**expected_args)
        assert isinstance(response, LLMResponse)
        assert response.choices == ["Hello from LiteLLM!"]
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_litellm_123"
        assert response.tool_calls[0].tool_name == "test_function"
        assert response.tool_calls[0].tool_args == {
            "param1": "value1",
            "param2": 42,
            "nested": {"key": "value"},
        }
        assert response.tool_calls[0].type == "function"

    @pytest.fixture
    def litellm_model_response_with_malformed_tool_calls(self) -> ModelResponse:
        """Create a ModelResponse with malformed tool calls."""
        tool_call = {
            "id": "call_litellm_malformed",
            "function": {
                "name": "malformed_function",
                "arguments": "not json at all",  # Not JSON at all
            },
            "type": "function",
        }

        return ModelResponse(
            id="id123",
            choices=[
                {
                    "message": {
                        "content": "Hello from LiteLLM!",
                        "role": "assistant",
                        "tool_calls": [tool_call],
                    }
                }
            ],
            created=1234567890,
            model="test_model",
            object="text_completion",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    @pytest.fixture
    def mock_malformed_tool_acompletion(
        self,
        monkeypatch: MonkeyPatch,
        litellm_model_response_with_malformed_tool_calls: ModelResponse,
    ) -> AsyncMock:
        # Create a mock object
        mock = AsyncMock(return_value=litellm_model_response_with_malformed_tool_calls)
        # Replace the 'acompletion' function in its module with the mock
        monkeypatch.setattr(
            "rasa.shared.providers.llm._base_litellm_client.acompletion", mock
        )
        return mock

    async def test_acompletion_with_malformed_tool_calls(
        self, client: TestLiteLLMClient, mock_malformed_tool_acompletion: AsyncMock
    ):
        """Test acompletion with malformed tool calls raises an exception."""
        # Given
        test_prompt = "Hello, this is a test prompt with malformed tools."

        # When/Then
        with pytest.raises(ProviderClientAPIException) as exc_info:
            await client.acompletion(test_prompt, tools=["malformed_tool"])

        mock_malformed_tool_acompletion.assert_called_once_with(
            messages=[{"content": test_prompt, "role": "user"}],
            model=client._litellm_model_name,
            drop_params=False,
            test_parameter="test_value",
            tools=["malformed_tool"],
        )

        # Verify the error message contains the expected information
        assert "Invalid arguments for tool call - `malformed_function`" in str(
            exc_info.value
        )
        assert "`not json at all`" in str(exc_info.value)
        assert isinstance(exc_info.value.original_exception, LLMToolResponseDecodeError)

        # Verify the original exception details
        json_decode_error = exc_info.value.original_exception.original_exception
        assert hasattr(json_decode_error, "msg")
        assert hasattr(json_decode_error, "pos")

    @pytest.mark.asyncio
    async def test_acompletion_timeout_enforcement(
        self, client: TestLiteLLMClient, monkeypatch: MonkeyPatch
    ):
        """Test that timeout error message correctly shows
        'time taken' is equivalent to 'timeout value' defined in 'endpoints.yml'."""
        import asyncio
        from unittest.mock import PropertyMock

        # Set small timeout value for timeout
        timeout_value = 0.00001
        monkeypatch.setattr(
            type(client),
            "_litellm_extra_parameters",
            PropertyMock(return_value={"timeout": timeout_value}),
        )

        # Mock acompletion with a slow operation that will timeout
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(1)

        monkeypatch.setattr(
            "rasa.shared.providers.llm._base_litellm_client.acompletion",
            AsyncMock(side_effect=slow_operation),
        )

        # Verify timeout is enforced and error message is correct
        with pytest.raises(ProviderClientAPIException) as exc_info:
            await client.acompletion("test message")

        # Verify error message shows 'time taken' to be
        # equivalent to 'timeout value'
        error_message = str(exc_info.value.original_exception)
        assert "APITimeoutError" in error_message
        assert f"timeout value={timeout_value:.6f}" in error_message
        assert f"time taken={timeout_value:.6f} seconds" in error_message
