from unittest.mock import Mock, patch

import pytest

from rasa.core.information_retrieval.information_retrieval import (
    create_from_endpoint_config,
)
from rasa.core.information_retrieval.milvus import Milvus, Milvus_Store
from rasa.core.information_retrieval.qdrant import Qdrant, Qdrant_Store


@pytest.fixture
def embeddings() -> Mock:
    return Mock()


def test_create_from_endpoint_config_with_milvus(embeddings: Mock) -> None:
    config_type = "milvus"

    # Mock the Milvus class to avoid connecting to a Milvus DB
    with patch.object(Milvus, "__init__", return_value=None):
        result = create_from_endpoint_config(config_type, embeddings)
        assert isinstance(result, Milvus_Store)


def test_create_from_endpoint_config_with_qdrant(embeddings: Mock) -> None:
    config_type = "qdrant"

    # Mock the Qdrant class to avoid connecting to a Qdrant DB
    with patch.object(Qdrant, "__init__", return_value=None):
        result = create_from_endpoint_config(config_type, embeddings)
        assert isinstance(result, Qdrant_Store)


def test_create_from_endpoint_config_with_unknown_type(embeddings: Mock) -> None:
    config_type = "unknown"

    with pytest.raises(ValueError) as e:
        create_from_endpoint_config(config_type, embeddings)
    assert str(e.value) == "Unknown vector store type 'unknown'."
