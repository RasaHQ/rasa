import pytest
import asyncio
from rasa.core.validator import Validator
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH,
    DEFAULT_STORIES_FILE,
    DEFAULT_NLU_DATA,
)
from rasa.core.domain import Domain
from rasa.nlu.training_data import load_data, TrainingData
from rasa.core.training.dsl import StoryFileReader


@pytest.fixture
def validator():
    domain = Domain.load(DEFAULT_DOMAIN_PATH)
    stories = asyncio.run(
        StoryFileReader.read_from_folder(DEFAULT_STORIES_FILE, domain)
    )
    intents = load_data(DEFAULT_NLU_DATA)

    return Validator(domain=domain, intents=intents, stories=stories)


def test_validator_creation(validator):
    assert isinstance(validator.domain, Domain)
    assert isinstance(validator.intents, TrainingData)
    assert isinstance(validator.stories, list)


def test_search(validator):
    vec = ["a", "b", "c", "d", "e"]
    assert validator._search(vector=vec, searched_value="c")


def test_verify_intents(validator):
    valid_intents = ["greet", "goodbye", "affirm"]
    validator.verify_intents()
    assert validator.valid_intents == valid_intents


def test_verify_utters(validator):
    valid_utterances = ["utter_greet", "utter_goodbye", "utter_default"]
    validator.verify_utterances()
    assert validator.valid_utterances == valid_utterances
