from unittest.mock import Mock

import pytest
from langchain.schema.embeddings import Embeddings
from langchain_community.embeddings import FakeEmbeddings


@pytest.fixture
def embeddings() -> Embeddings:
    return FakeEmbeddings(size=768)


@pytest.fixture
def mock_import_module(mocker) -> Mock:
    mock = mocker.patch("my_module.importlib.import_module")
    return mock
