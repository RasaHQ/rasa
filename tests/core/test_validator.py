import pytest
from rasa.core.validator import Validator
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH,
    DEFAULT_STORIES_FILE,
    DEFAULT_NLU_DATA,
)
from rasa.core.domain import Domain
from rasa.nlu.training_data import TrainingData


@pytest.fixture
def validator():
    return Validator.from_files(
        domain_file=DEFAULT_DOMAIN_PATH,
        nlu_data=DEFAULT_NLU_DATA,
        story_data=DEFAULT_STORIES_FILE,
    )


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
    assert set(validator.valid_intents) == set(valid_intents)


def test_verify_utterances(validator):
    valid_utterances = ["utter_greet", "utter_goodbye", "utter_default"]
    validator.verify_utterances()
    assert set(validator.valid_utterances) == set(valid_utterances)
