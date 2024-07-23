from langchain_community.embeddings import FakeEmbeddings
from langchain.schema.embeddings import Embeddings
import pytest
from unittest.mock import Mock


@pytest.fixture
def embeddings() -> Embeddings:
    return FakeEmbeddings(size=768)


@pytest.fixture
def mock_import_module(mocker) -> Mock:
    mock = mocker.patch("my_module.importlib.import_module")
    return mock
