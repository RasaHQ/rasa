import pytest
import spacy
import mitie


@pytest.fixture(scope="session")
def spacy_nlp_en():
    return spacy.load("en", parser=False)


@pytest.fixture(scope="session")
def mitie_feature_extractor():
    mitie_file = "data/total_word_feature_Extractor.dat"
    return mitie.total_word_feature_extractor(mitie_file)
