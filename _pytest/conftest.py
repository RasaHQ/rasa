import pytest
import spacy


@pytest.fixture
def spacy_nlp_en():
    return spacy.load("en", parser=False)
