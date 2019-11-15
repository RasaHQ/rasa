import pytest
import logging
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


async def test_verify_there_is_example_repetition_in_intents():
    # moodbot nlu data already has duplicated example 'good afternoon'
    # for intents greet and goodbye
    importer = RasaFileImporter(
        domain_path="examples/moodbot/domain.yml",
        training_data_paths=["examples/moodbot/data/nlu.md"],
    )
    validator = await Validator.from_importer(importer)
    assert not validator.verify_example_repetition_in_intents(False)


async def test_verify_logging_message_for_repetition_in_intents(caplog):
    # moodbot nlu data already has duplicated example 'good afternoon'
    # for intents greet and goodbye
    importer = RasaFileImporter(
        domain_path="examples/moodbot/domain.yml",
        training_data_paths=["examples/moodbot/data/nlu.md"],
    )
    validator = await Validator.from_importer(importer)
    caplog.clear()  # clear caplog to avoid counting earlier debug messages
    with pytest.warns(UserWarning) as record:
        validator.verify_example_repetition_in_intents(False)
    assert len(record) == 1
    assert (
        "The example 'good afternoon' was found in these "
        "multiples intents: goodbye, greet" in record[0].message.args[0]
    )


async def test_early_exit_on_invalid_domain():
    domain_path = "data/test_domains/duplicate_intents.yml"

    importer = RasaFileImporter(domain_path=domain_path)
    with pytest.warns(UserWarning) as record:
        validator = await Validator.from_importer(importer)
    validator.verify_domain_validity()
    assert len(record) == 2
    assert (
        f"Loading domain from '{domain_path}' failed. Using empty domain. "
        "Error: 'Intents are not unique! Found two intents with name "
        "'default'. Either rename or remove one of them.'" in record[0].message.args[0]
    )
    assert (
        f"Loading domain from '{domain_path}' failed. Using empty domain. "
        "Error: 'Intents are not unique! Found two intents with name "
        "'default'. Either rename or remove one of them.'" in record[1].message.args[0]
    )


async def test_verify_there_is_not_example_repetition_in_intents():
    importer = RasaFileImporter(
        domain_path="examples/moodbot/domain.yml",
        training_data_paths=["examples/knowledgebasebot/data/nlu.md"],
    )
    validator = await Validator.from_importer(importer)
    assert validator.verify_example_repetition_in_intents(False)
