import pytest
from rasa.core.validator import Validator
from rasa.importers.rasa import RasaFileImporter
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH,
    DEFAULT_STORIES_FILE,
    DEFAULT_NLU_DATA,
)
from rasa.core.domain import Domain
from rasa.nlu.training_data import TrainingData


@pytest.fixture
async def validator():
    importer = RasaFileImporter(
        domain_path=DEFAULT_DOMAIN_PATH,
        training_data_paths=[DEFAULT_NLU_DATA, DEFAULT_STORIES_FILE],
    )
    return await Validator.from_importer(importer)


def test_validator_creation(validator: Validator):
    assert isinstance(validator.domain, Domain)
    assert isinstance(validator.intents, TrainingData)
    assert isinstance(validator.stories, list)


def test_verify_intents(validator: Validator):
    valid_intents = [intent for intent in validator.domain.intents]
    verified_intents = validator.verify_intents()
    assert set(verified_intents) == set(valid_intents)


def test_verify_utterances(validator: Validator):
    valid_utterances = ["utter_greet", "utter_goodbye", "utter_default"]
    verified_utterances = validator.verify_utterances()
    assert set(verified_utterances) == set(valid_utterances)
