from dataclasses import is_dataclass

from rasa.shared.providers.embedding.embedding_response import (
    EmbeddingResponse,
    EmbeddingUsage,
)


def test_embedding_usage_is_dataclass() -> None:
    assert is_dataclass(EmbeddingUsage)


def test_embedding_response_is_dataclass() -> None:
    assert is_dataclass(EmbeddingResponse)


def test_embedding_usage_initialization() -> None:
    usage = EmbeddingUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_embedding_response_initialization() -> None:
    data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    embedding_response = EmbeddingResponse(data=data)
    assert embedding_response.data == data


def test_embedding_response_with_usage_initialization() -> None:
    data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    usage = EmbeddingUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    embedding_response = EmbeddingResponse(data=data, usage=usage)
    assert embedding_response.data == data
    assert embedding_response.usage == usage


def test_empty_initialization() -> None:
    embedding_response = EmbeddingResponse(data=[], usage=None)
    assert embedding_response.data == []
    assert embedding_response.usage is None
