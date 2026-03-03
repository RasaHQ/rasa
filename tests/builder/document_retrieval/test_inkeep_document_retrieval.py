"""Unit tests for InKeepDocumentRetrieval functionality."""

import asyncio
import importlib
import json
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import openai
import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice

from rasa.builder import config
from rasa.builder.document_retrieval.constants import (
    INKEEP_API_KEY_ENV_VAR,
    INKEEP_BASE_URL_ENV_VAR,
)
from rasa.builder.document_retrieval.inkeep_document_retrieval import (
    Document,
    InKeepDocumentRetrieval,
)
from rasa.builder.exceptions import DocumentRetrievalError


@pytest.fixture(autouse=True)
def mock_langfuse():
    """Mock Langfuse client to prevent network connections during tests."""
    with patch("langfuse.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_generation = MagicMock()
        mock_generation.__enter__ = MagicMock(return_value=mock_generation)
        mock_generation.__exit__ = MagicMock(return_value=None)
        mock_client.start_as_current_generation.return_value = mock_generation
        mock_get_client.return_value = mock_client
        yield mock_client


class TestInKeepDocumentRetrieval:
    """Test cases for InKeepDocumentRetrieval class."""

    @pytest.fixture
    def mock_api_key(self) -> str:
        """Provide a mock API key for testing."""
        return "test_inkeep_document_retrieval_api_key_12345"

    @pytest.fixture
    def sample_rag_response(self) -> Dict[str, Any]:
        """Provide a sample RAG response for testing."""
        return {
            "content": [
                {
                    "title": "Test Document 1",
                    "url": "https://example.com/doc1",
                    "type": "documentation",
                    "record_type": "test record type 1",
                    "context": "Testing context",
                    "source": {
                        "media_type": "text",
                        "content": [{"type": "text", "text": "This is test content 1"}],
                        "data": "Fallback data 1",
                    },
                },
                {
                    "title": "Test Document 2",
                    "url": "https://example.com/doc2",
                    "type": "documentation",
                    "record_type": "test record type 2",
                    "context": "Another context",
                    "source": {
                        "media_type": "text",
                        "content": [{"type": "text", "text": "This is test content 2"}],
                        "data": "Fallback data 2",
                    },
                },
            ]
        }

    @pytest.fixture
    def mock_chat_completion(
        self, sample_rag_response: Dict[str, Any]
    ) -> ChatCompletion:
        """Create a mock ChatCompletion response."""
        return ChatCompletion(
            id="test_id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=json.dumps(sample_rag_response),
                        role="assistant",
                    ),
                )
            ],
            created=1234567890,
            model="inkeep-rag",
            object="chat.completion",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "api_key, env_key, expected_key",
        [
            ("api_key", None, "api_key"),
            (None, "env_key", "env_key"),
            (None, None, None),
        ],
    )
    async def test_init_with_api_key(
        self,
        api_key: Optional[str],
        env_key: Optional[str],
        expected_key: Optional[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test initialization with different API key scenarios."""
        # Given
        if env_key:
            monkeypatch.setenv(INKEEP_API_KEY_ENV_VAR, env_key)
        elif env_key is None and api_key is None:
            monkeypatch.delenv(INKEEP_API_KEY_ENV_VAR, raising=False)

        # When
        retrieval = InKeepDocumentRetrieval(api_key=api_key)

        # Then
        assert retrieval._api_key == expected_key

    @pytest.mark.parametrize(
        "base_url_arg, env_url, config_url, expected_base_url",
        [
            (
                "https://custom.inkeep.com/v1",
                None,
                "https://api.inkeep.com/v1",
                "https://custom.inkeep.com/v1",
            ),
            (
                None,
                "https://env.inkeep.com/v1",
                "https://api.inkeep.com/v1",
                "https://env.inkeep.com/v1",
            ),
            (
                None,
                None,
                "https://config.inkeep.com/v1",
                "https://config.inkeep.com/v1",
            ),
        ],
    )
    def test_base_url_resolution(
        self,
        base_url_arg: Optional[str],
        env_url: Optional[str],
        config_url: str,
        expected_base_url: str,
        mock_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that base_url is resolved from constructor, env, or config."""
        if env_url is not None:
            monkeypatch.setenv(INKEEP_BASE_URL_ENV_VAR, env_url)
        else:
            monkeypatch.delenv(INKEEP_BASE_URL_ENV_VAR, raising=False)

        with patch.object(config, "INKEEP_BASE_URL", config_url):
            retrieval = InKeepDocumentRetrieval(
                api_key=mock_api_key,
                base_url=base_url_arg,
            )
            assert retrieval.base_url == expected_base_url

    def test_base_url_resolution_with_proxy(
        self,
        mock_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv(INKEEP_BASE_URL_ENV_VAR, raising=False)
        proxy_base = "https://llm-proxy.example.com"
        proxy_documentation_url = f"{proxy_base}/documentation"

        with (
            patch.object(config, "HELLO_LLM_PROXY_BASE_URL", proxy_base),
            patch.object(config, "INKEEP_BASE_URL", proxy_documentation_url),
        ):
            retrieval = InKeepDocumentRetrieval(api_key=mock_api_key, base_url=None)
            assert retrieval.base_url == proxy_documentation_url

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_retrieve_documents_success(
        self,
        mock_openai_class: Mock,
        mock_api_key: str,
        mock_chat_completion: ChatCompletion,
    ):
        """Test successful document retrieval."""
        # Given
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mock_openai_class.return_value = mock_client

        # When
        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)
        documents = await retrieval.retrieve_documents("test query")

        # Then
        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert documents[0].content == "This is test content 1"
        assert documents[0].title == "Test Document 1"
        assert documents[0].url == "https://example.com/doc1"
        assert documents[0].metadata["type"] == "documentation"
        assert documents[0].metadata["record_type"] == "test record type 1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "client_error,expected_error_message",
        [
            (openai.OpenAIError, "InKeep Document Retrieval: API error"),
            (asyncio.TimeoutError, "InKeep AI request timed out"),
            (Exception, "InKeep Document Retrieval: Unexpected error"),
        ],
    )
    @patch("openai.AsyncOpenAI")
    async def test_retrieve_documents_api_errors(
        self,
        mock_openai_class: Mock,
        mock_api_key: str,
        client_error: Type[Exception],
        expected_error_message: Optional[str],
    ):
        """Test API error scenarios when calling retrieve_documents."""
        # Given
        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = client_error("test error")
        mock_openai_class.return_value = mock_client

        # When
        with pytest.raises(DocumentRetrievalError) as exc_info:
            await retrieval.retrieve_documents("test query")

        # Then
        if expected_error_message:
            assert expected_error_message in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "api_response,expected_exception,expected_error_message",
        [
            (
                {"choices": [{"message": {"content": None}}]},
                DocumentRetrievalError,
                "InKeep Document Retrieval: Empty response",
            ),
            (
                {"choices": [{"message": {"content": "invalid json"}}]},
                None,
                None,
            ),
        ],
    )
    @patch("openai.AsyncOpenAI")
    async def test_retrieve_documents_parsing_errors(
        self,
        mock_openai_class: Mock,
        mock_api_key: str,
        api_response: Optional[Dict[str, Any]],
        expected_exception: Optional[Type[Exception]],
        expected_error_message: Optional[str],
    ):
        """Test parsing error scenarios when calling retrieve_documents."""
        # Given
        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        if api_response is None:
            mock_message.content = None
        else:
            mock_message.content = api_response["choices"][0]["message"]["content"]

        mock_choice.message = mock_message
        mock_response.choices = (
            [mock_choice] if api_response and "choices" in api_response else []
        )

        # Explicitly set usage to None to avoid Mock object issues
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # When
        if expected_exception:
            with pytest.raises(expected_exception) as exc_info:
                await retrieval.retrieve_documents("test query")

            # Then
            if expected_error_message:
                assert expected_error_message in str(exc_info.value)
        else:
            # When
            documents = await retrieval.retrieve_documents("test query")
            assert isinstance(documents, list)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "api_response,expected_exception,expected_error_message",
        [
            # Empty content array
            (
                {"choices": [{"message": {"content": '{"content": []}'}}]},
                None,
                None,
            ),
            # Missing choices
            (
                {"other_field": "value"},
                DocumentRetrievalError,
                "InKeep Document Retrieval: Unexpected error",
            ),
            # Empty choices
            (
                {"choices": []},
                DocumentRetrievalError,
                "InKeep Document Retrieval: Unexpected error",
            ),
        ],
    )
    @patch("openai.AsyncOpenAI")
    async def test_retrieve_documents_edge_cases(
        self,
        mock_openai_class: Mock,
        mock_api_key: str,
        api_response: Optional[Dict[str, Any]],
        expected_exception: Optional[Type[Exception]],
        expected_error_message: Optional[str],
    ):
        """Test edge case scenarios when calling retrieve_documents."""
        # Given
        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        if api_response is None:
            mock_message.content = None
            mock_choice.message = mock_message
            mock_response.choices = []
        elif api_response.get("choices"):
            # Valid response with choices
            mock_message.content = api_response["choices"][0]["message"]["content"]
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
        else:
            # Missing choices or empty choices
            mock_message.content = None
            mock_choice.message = mock_message
            mock_response.choices = []

        # Explicitly set usage to None to avoid Mock object issues
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # When
        if expected_exception:
            with pytest.raises(expected_exception) as exc_info:
                await retrieval.retrieve_documents("test query")

            # Then
            if expected_error_message:
                assert expected_error_message in str(exc_info.value)
        else:
            # When
            documents = await retrieval.retrieve_documents("test query")
            assert isinstance(documents, list)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response_content,expected_documents",
        [
            # Empty response
            ({"content": []}, []),
            # None response
            (None, []),
            # Invalid JSON
            ("invalid json", []),
        ],
    )
    async def test_parse_documents_edge_cases(
        self,
        response_content: Optional[Union[str, Dict[str, Any]]],
        expected_documents: List[Document],
        mock_api_key: str,
    ):
        """Test parsing documents with edge cases."""
        # Given
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        if response_content is None:
            mock_message.content = None
        elif isinstance(response_content, str):
            mock_message.content = response_content
        else:
            mock_message.content = json.dumps(response_content)

        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # When
        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)
        documents = retrieval._parse_documents_from_response(mock_response)

        # Then
        assert len(documents) == len(expected_documents)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_type,expected_error_message",
        [
            (openai.OpenAIError, "InKeep Document Retrieval: API error"),
            (asyncio.TimeoutError, "InKeep AI request timed out"),
            (Exception, "InKeep Document Retrieval: Unexpected error"),
        ],
    )
    @patch("openai.AsyncOpenAI")
    async def test_call_inkeep_rag_api_exceptions(
        self,
        mock_openai_class: Mock,
        exception_type: Type[Exception],
        expected_error_message: str,
        mock_api_key: str,
    ):
        """Test API call exception handling."""
        # Given
        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = exception_type("test error")
        mock_openai_class.return_value = mock_client

        # When
        with pytest.raises(DocumentRetrievalError) as exc_info:
            await retrieval._call_inkeep_rag_api("test query", 0.0, 30.0)

        # Then
        assert expected_error_message in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_call_inkeep_rag_api_empty_response(
        self,
        mock_openai_class: Mock,
        mock_api_key: str,
    ):
        """Test handling of empty response from API."""
        # Given
        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)
        expected_error_message = "InKeep Document Retrieval: Empty response"

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # When
        with pytest.raises(DocumentRetrievalError) as exc_info:
            await retrieval._call_inkeep_rag_api("test query", 0.0, 30.0)

        # Then
        assert expected_error_message in str(exc_info.value)

    @pytest.mark.asyncio
    @patch(
        "rasa.builder.document_retrieval.inkeep_document_retrieval.config.DEPLOYMENT_STACK",
        "unit-test",
    )
    @patch(
        "rasa.builder.document_retrieval.inkeep_document_retrieval.config.DEPLOYMENT_STACK_HEADER_NAME",
        "deployment-stack",
    )
    @patch(
        "rasa.builder.document_retrieval.inkeep_document_retrieval.config.HELLO_LLM_PROXY_BASE_URL",
        None,
    )
    @patch("openai.AsyncOpenAI")
    async def test_get_client_creation(
        self,
        mock_openai_class: Mock,
        mock_api_key: str,
    ):
        """Test client creation functionality."""
        # Given
        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # When - First call - should create client
        async with retrieval._get_client() as client1:
            assert client1 == mock_client
            # Implementation adds trailing slash to match client expectations
            expected_base_url = f"{config.INKEEP_BASE_URL.rstrip('/')}/"
            mock_openai_class.assert_called_once_with(
                api_key=mock_api_key,
                base_url=expected_base_url,
                default_headers={"deployment-stack": "unit-test"},
            )

        # When - Second call - should create a new client
        async with retrieval._get_client() as client2:
            assert client2 == mock_client

        # Then - Should be called twice (new client each time)
        assert mock_openai_class.call_count == 2

    async def test_inkeep_client_proxy_base_url(
        self, monkeypatch: pytest.MonkeyPatch, mock_api_key: str
    ):
        proxy = "https://hello-llm-proxy.example"
        license_token = "rasa-license-jwt"
        monkeypatch.setenv("HELLO_LLM_PROXY_BASE_URL", proxy)
        monkeypatch.setenv("RASA_PRO_LICENSE", license_token)

        # Reload config to re-evaluate INKEEP_BASE_URL (computed at import)
        importlib.reload(config)

        # Patch config values after reload to ensure they take effect
        monkeypatch.setattr(config, "DEPLOYMENT_STACK", "unit-test")
        monkeypatch.setattr(config, "DEPLOYMENT_STACK_HEADER_NAME", "deployment-stack")

        mock_client = AsyncMock()
        async_openai_mock = MagicMock(return_value=mock_client)
        monkeypatch.setattr("openai.AsyncOpenAI", async_openai_mock)

        retrieval = InKeepDocumentRetrieval(api_key=mock_api_key)

        async with retrieval._get_client() as client:
            assert client == mock_client

        # Expect proxy auth (license), not provider key
        assert (
            async_openai_mock.call_args.kwargs.get("api_key") == config.RASA_PRO_LICENSE
        )

        # Expect the dynamically computed proxy base url from config with trailing slash
        expected_base_url = getattr(config, "INKEEP_BASE_URL", None)
        assert expected_base_url
        # Implementation adds trailing slash to match client expectations
        expected_base_url_with_slash = f"{expected_base_url.rstrip('/')}/"
        assert (
            async_openai_mock.call_args.kwargs.get("base_url")
            == expected_base_url_with_slash
        )

        # Verify default_headers are set correctly
        assert async_openai_mock.call_args.kwargs.get("default_headers") == {
            "deployment-stack": "unit-test"
        }
