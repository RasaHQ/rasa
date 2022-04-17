from pathlib import Path
from typing import Text
import os

from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_CONVERSATION_TEST_PATH,
)
from rasa.shared.core.constants import DEFAULT_INTENTS, SESSION_START_METADATA_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import AnySlot
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.importers.rasa import RasaFileImporter


def test_rasa_file_importer(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)

    importer = RasaFileImporter(config_path, domain_path, [default_data_path])

    domain = importer.get_domain()
    assert len(domain.intents) == 7 + len(DEFAULT_INTENTS)
    assert domain.slots == [AnySlot(SESSION_START_METADATA_SLOT, mappings=[{}])]
    assert domain.entities == []
    assert len(domain.action_names_or_texts) == 19
    assert len(domain.responses) == 6

    stories = importer.get_stories()
    assert len(stories.story_steps) == 5

    test_stories = importer.get_conversation_tests()
    assert len(test_stories.story_steps) == 0

    nlu_data = importer.get_nlu_data("en")
    assert len(nlu_data.intents) == 7
    assert len(nlu_data.intent_examples) == 68


def test_read_conversation_tests(project: Text):
    importer = RasaFileImporter(
        training_data_paths=[str(Path(project) / DEFAULT_CONVERSATION_TEST_PATH)]
    )

    test_stories = importer.get_conversation_tests()
    assert len(test_stories.story_steps) == 7


def test_rasa_file_importer_with_invalid_config():
    importer = RasaFileImporter(config_file="invalid path")
    actual = importer.get_config()

    assert actual == {}


def test_rasa_file_importer_with_invalid_domain(tmp_path: Path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("")
    importer = TrainingDataImporter.load_from_dict({}, str(config_file), None, [])

    actual = importer.get_domain()
    assert actual.as_dict() == Domain.empty().as_dict()
