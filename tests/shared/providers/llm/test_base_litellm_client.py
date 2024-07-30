from typing import Dict, Any, Union, List, Optional
from unittest.mock import Mock, AsyncMock
from litellm.utils import Usage
import pytest
from litellm import ModelResponse
from pytest import MonkeyPatch

from rasa.shared.exceptions import ProviderClientAPIException
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
        self, test_prompt: str, client: TestLiteLLMClient, mock_completion: Mock
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
