import pytest
from rasa.core.validator import Validator
from rasa.importers.rasa import RasaFileImporter
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_STORIES_FILE,
    DEFAULT_NLU_DATA,
)
from rasa.core.domain import Domain
from rasa.nlu.training_data import TrainingData
import rasa.utils.io as io_utils


async def test_verify_intents_does_not_fail_on_valid_data():
    importer = RasaFileImporter(
        domain_path="examples/moodbot/domain.yml",
        training_data_paths=["examples/moodbot/data/nlu.md"],
    )
    validator = await Validator.from_importer(importer)
    assert validator.verify_intents()


async def test_verify_intents_does_fail_on_invalid_data():
    # domain and nlu data are from different domain and should produce warnings
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=["examples/moodbot/data/nlu.md"],
    )
    validator = await Validator.from_importer(importer)
    assert not validator.verify_intents()


async def test_verify_valid_utterances():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=[DEFAULT_NLU_DATA, DEFAULT_STORIES_FILE],
    )
    validator = await Validator.from_importer(importer)
    assert validator.verify_utterances()


async def test_fail_on_invalid_utterances(tmpdir):
    # domain and stories are from different domain and should produce warnings
    invalid_domain = str(tmpdir / "invalid_domain.yml")
    io_utils.write_yaml_file(
        {
            "templates": {"utter_greet": {"text": "hello"}},
            "actions": [
                "utter_greet",
                "utter_non_existent",  # error: utter template odes not exist
            ],
        },
        invalid_domain,
    )
    importer = RasaFileImporter(domain_path=invalid_domain)
    validator = await Validator.from_importer(importer)
    assert not validator.verify_utterances()
