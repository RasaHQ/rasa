import pytest
import spacy


@pytest.fixture(scope="session")
def spacy_nlp_en():
    return spacy.load("en", parser=False)
