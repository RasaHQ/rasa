from langchain.embeddings import FakeEmbeddings
from langchain.schema.embeddings import Embeddings
import pytest


@pytest.fixture
def embeddings() -> Embeddings:
    return FakeEmbeddings(size=768)
