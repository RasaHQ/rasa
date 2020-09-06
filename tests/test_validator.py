from pep440_version_utils import Version

import pytest

from rasa.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.validator import Validator, KEY_TRAINING_DATA_FORMAT_VERSION
from rasa.importers.rasa import RasaFileImporter
from tests.conftest import DEFAULT_NLU_DATA
from tests.core.conftest import DEFAULT_STORIES_FILE
import rasa.utils.io as io_utils


async def test_verify_intents_does_not_fail_on_valid_data():
    importer = RasaFileImporter(
        domain_path="examples/moodbot/domain.yml",
        training_data_paths=[DEFAULT_NLU_DATA],
    )
    validator = await Validator.from_importer(importer)
    assert validator.verify_intents()


async def test_verify_intents_does_fail_on_invalid_data():
    # domain and nlu data are from different domain and should produce warnings
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=[DEFAULT_NLU_DATA],
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


async def test_verify_story_structure():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=[DEFAULT_STORIES_FILE],
    )
    validator = await Validator.from_importer(importer)
    assert validator.verify_story_structure(ignore_warnings=False)


async def test_verify_bad_story_structure():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=["data/test_stories/stories_conflicting_2.md"],
    )
    validator = await Validator.from_importer(importer)
    assert not validator.verify_story_structure(ignore_warnings=False)


async def test_verify_bad_story_structure_ignore_warnings():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=["data/test_stories/stories_conflicting_2.md"],
    )
    validator = await Validator.from_importer(importer)
    assert validator.verify_story_structure(ignore_warnings=True)


async def test_fail_on_invalid_utterances(tmpdir):
    # domain and stories are from different domain and should produce warnings
    invalid_domain = str(tmpdir / "invalid_domain.yml")
    io_utils.write_yaml(
        {
            "responses": {"utter_greet": [{"text": "hello"}]},
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
        training_data_paths=[DEFAULT_NLU_DATA],
    )
    validator = await Validator.from_importer(importer)
    assert not validator.verify_example_repetition_in_intents(False)


async def test_verify_logging_message_for_repetition_in_intents(caplog):
    # moodbot nlu data already has duplicated example 'good afternoon'
    # for intents greet and goodbye
    importer = RasaFileImporter(
        domain_path="examples/moodbot/domain.yml",
        training_data_paths=[DEFAULT_NLU_DATA],
    )
    validator = await Validator.from_importer(importer)
    caplog.clear()  # clear caplog to avoid counting earlier debug messages
    with pytest.warns(UserWarning) as record:
        validator.verify_example_repetition_in_intents(False)
    assert len(record) == 1
    assert "You should fix that conflict " in record[0].message.args[0]


async def test_early_exit_on_invalid_domain():
    domain_path = "data/test_domains/duplicate_intents.yml"

    importer = RasaFileImporter(domain_path=domain_path)
    with pytest.warns(UserWarning) as record:
        validator = await Validator.from_importer(importer)
    validator.verify_domain_validity()

    # two for non-unique domains
    assert len(record) == 2
    assert (
        f"Loading domain from '{domain_path}' failed. Using empty domain. "
        "Error: 'Intents are not unique! Found multiple intents with name(s) "
        "['default', 'goodbye']. Either rename or remove the duplicate ones.'"
        in record[0].message.args[0]
    )
    assert record[0].message.args[0] == record[1].message.args[0]


async def test_verify_there_is_not_example_repetition_in_intents():
    importer = RasaFileImporter(
        domain_path="examples/moodbot/domain.yml",
        training_data_paths=["examples/knowledgebasebot/data/nlu.md"],
    )
    validator = await Validator.from_importer(importer)
    assert validator.verify_example_repetition_in_intents(False)


async def test_future_training_data_format_version_not_compatible():

    next_minor = str(Version(LATEST_TRAINING_DATA_FORMAT_VERSION).next_minor())

    incompatible_version = {KEY_TRAINING_DATA_FORMAT_VERSION: next_minor}

    with pytest.warns(UserWarning):
        assert not Validator.validate_training_data_format_version(
            incompatible_version, ""
        )


async def test_compatible_training_data_format_version():

    prev_major = str(Version("1.0"))

    compatible_version_1 = {KEY_TRAINING_DATA_FORMAT_VERSION: prev_major}
    compatible_version_2 = {
        KEY_TRAINING_DATA_FORMAT_VERSION: LATEST_TRAINING_DATA_FORMAT_VERSION
    }

    for version in [compatible_version_1, compatible_version_2]:
        with pytest.warns(None):
            assert Validator.validate_training_data_format_version(version, "")


async def test_invalid_training_data_format_version_warns():

    invalid_version_1 = {KEY_TRAINING_DATA_FORMAT_VERSION: 2.0}
    invalid_version_2 = {KEY_TRAINING_DATA_FORMAT_VERSION: "Rasa"}

    for version in [invalid_version_1, invalid_version_2]:
        with pytest.warns(UserWarning):
            assert Validator.validate_training_data_format_version(version, "")
