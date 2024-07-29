import asyncio
import json
import random
from typing import Dict, Text
from unittest.mock import AsyncMock

import pytest
import structlog
from aiohttp import ClientResponse
from pytest import MonkeyPatch

from rasa.shared.providers.openai.clients import (
    AioHTTPSessionAzureChatOpenAI,
    AioHTTPSessionOpenAIChat,
    AioHTTPSessionOpenAIEmbeddings,
)

structlogger = structlog.get_logger()

# file deepcode ignore HardcodedNonCryptoSecret/test: Test Credentials

@pytest.fixture
def mock_request(monkeypatch: MonkeyPatch) -> AsyncMock:
    mock_request = AsyncMock()
    monkeypatch.setattr("aiohttp.client.ClientSession.request", mock_request)
    return mock_request


def create_response(json_response: Dict) -> AsyncMock:
    response = AsyncMock(spec=ClientResponse)
    response.status = 200
    response.json = AsyncMock(return_value=json_response)
    response.text = AsyncMock(return_value=json.dumps(json_response))
    response.read = AsyncMock(return_value=json.dumps(json_response).encode())
    return response


def create_completion_response(content: Text) -> AsyncMock:
    json_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-llm-model",
        "system_fingerprint": "fingerprint123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,  # make sure the same content is in response
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return create_response(json_response)


def create_embedding_response() -> AsyncMock:
    json_response = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [1.0 for _ in range(1536)], "index": 0}
        ],
        "model": "test-embedding-model",
        "usage": {"prompt_tokens": 100, "total_tokens": 100},
    }
    return create_response(json_response)


async def completion_request_side_effect(*args, **kwargs) -> AsyncMock:
    sent_data = json.loads(kwargs["data"].decode("utf-8"))
    content = sent_data["messages"][0]["content"]
    # random delay to simulate network or processing latency
    await asyncio.sleep(random.uniform(0.1, 1.0))
    structlogger.info("Handling message", content=content)
    return create_completion_response(content)


async def embedding_request_side_effect(*args, **kwargs) -> AsyncMock:
    sent_data = json.loads(kwargs["data"].decode("utf-8"))
    content = sent_data["input"]
    await asyncio.sleep(random.uniform(0.1, 1.0))
    structlogger.info("Handling embedding request", content=content)
    return create_embedding_response()


@pytest.mark.asyncio
async def test_azure_chat_client(
    mock_request: AsyncMock,
) -> None:
    # Given
    mock_request.side_effect = completion_request_side_effect
    messages = [f"azure-client-message-{i}" for i in range(10)]
    azure_client = AioHTTPSessionAzureChatOpenAI(
        openai_api_key="fake_api_key",
        openai_api_base="https://fakeapi.openai.com",
        openai_api_version="fake_api_version",
    )

    # When: run all tasks concurrently
    tasks = [azure_client.apredict(text=message) for message in messages]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Then
    assert len(responses) == len(messages)
    for message in messages:
        assert message in responses


@pytest.mark.asyncio
async def test_openai_chat_client(
    mock_request: AsyncMock,
) -> None:
    # Given
    mock_request.side_effect = completion_request_side_effect
    messages = [f"openai-client-message-{i}" for i in range(10)]
    client = AioHTTPSessionOpenAIChat(openai_api_key="fake_api_key")

    # When: run all tasks concurrently
    tasks = [client.apredict(text=message) for message in messages]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Then
    assert len(responses) == len(messages)
    for message in messages:
        assert message in responses


@pytest.mark.asyncio
async def test_embedding_client(
    mock_request: AsyncMock,
) -> None:
    # Given
    mock_request.side_effect = embedding_request_side_effect
    documents = [f"embedding-client-message-{i}" for i in range(10)]
    client = AioHTTPSessionOpenAIEmbeddings(
        openai_api_key="fake_api_key",
    )

    # When: run all tasks concurrently
    tasks = [client.aembed_documents([document]) for document in documents]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Then
    assert len(responses) == len(documents)
    assert all(
        not isinstance(response, Exception) for response in responses
    ), "All tasks should complete without exceptions"
