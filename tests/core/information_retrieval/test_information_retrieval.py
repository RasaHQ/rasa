from unittest.mock import Mock, patch
from typing import TYPE_CHECKING
import pytest

from rasa.core.information_retrieval import create_from_endpoint_config
from rasa.core.information_retrieval.milvus import Milvus, Milvus_Store
from rasa.core.information_retrieval.qdrant import Qdrant, Qdrant_Store
from data.test_information_retrieval.custom_store import CustomStore

if TYPE_CHECKING:
    from langchain.schema.embeddings import Embeddings


def test_create_from_endpoint_config_with_milvus(embeddings: "Embeddings") -> None:
    config_type = "milvus"

    # Mock the Milvus class to avoid connecting to a Milvus DB
    with patch.object(Milvus, "__init__", return_value=None):
        result = create_from_endpoint_config(config_type, embeddings)
        assert isinstance(result, Milvus_Store)


def test_create_from_endpoint_config_with_qdrant(embeddings: "Embeddings") -> None:
    config_type = "qdrant"

    # Mock the Qdrant class to avoid connecting to a Qdrant DB
    with patch.object(Qdrant, "__init__", return_value=None):
        result = create_from_endpoint_config(config_type, embeddings)
        assert isinstance(result, Qdrant_Store)


@patch("importlib.import_module")
def test_custom_information_retrieval(
    mock_import_module: Mock, embeddings: "Embeddings"
) -> None:
    # Given: A config_type that is neither "milvus" nor "qdrant"
    config_type = "data.test_information_retrieval.custom_store.CustomStore"

    # Mock the module and class creation
    mock_module = Mock()
    mock_module.CustomStore = CustomStore
    mock_import_module.return_value = mock_module

    # When: Calling create_from_endpoint_config with the
    #   unknown config_type and mock embeddings
    result = create_from_endpoint_config(config_type, embeddings)

    # Then: Ensure that the dynamic import is attempted
    #   and the external class is instantiated correctly
    mock_import_module.assert_called_once_with(
        "data.test_information_retrieval.custom_store"
    )
    assert isinstance(result, CustomStore)


def test_create_from_endpoint_config_with_unknown_type(
    embeddings: "Embeddings",
) -> None:
    config_type = "unknown"

    with pytest.raises(ValueError) as e:
        create_from_endpoint_config(config_type, embeddings)
    assert (
        str(e.value) == f"Invalid configuration for vector store: '{config_type}'."
        " Expected a module path and a class name separated by a dot."
    )


def test_create_from_endpoint_config_invalid_config(embeddings: "Embeddings") -> None:
    config_type = "invalid.import.path"

    with pytest.raises(ImportError) as e:
        create_from_endpoint_config(config_type, embeddings)
    assert str(e.value) == f"Cannot retrieve class from path {config_type}."
