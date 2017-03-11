import logging
import pytest
import spacy
from mitie import total_word_feature_extractor

from rasa_nlu.data_router import InterpreterBuilder


logging.basicConfig(level="DEBUG")


@pytest.fixture(scope="session")
def spacy_nlp_en():
    return spacy.load("en", parser=False)


@pytest.fixture(scope="session")
def mitie_feature_extractor():
    mitie_file = "data/total_word_feature_Extractor.dat"
    return total_word_feature_extractor(mitie_file)


@pytest.fixture(scope="session")
def interpreter_builder():
    return InterpreterBuilder(use_cache=True)
