from pathlib import Path
from typing import Text
import os

from rasa.constants import DEFAULT_CONFIG_PATH, DEFAULT_DOMAIN_PATH, DEFAULT_DATA_PATH
from rasa.core.domain import Domain
from rasa.importers.importer import TrainingDataImporter
from rasa.importers.rasa import RasaFileImporter

# noinspection PyUnresolvedReferences
from tests.core.conftest import project


async def test_rasa_file_importer(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)

    importer = RasaFileImporter(config_path, domain_path, [default_data_path])

    domain = await importer.get_domain()
    assert len(domain.intents) == 7
    assert domain.slots == []
    assert domain.entities == []
    assert len(domain.action_names) == 14
    assert len(domain.templates) == 6

    stories = await importer.get_stories()
    assert len(stories.story_steps) == 5

    nlu_data = await importer.get_nlu_data("en")
    assert len(nlu_data.intents) == 7
    assert len(nlu_data.intent_examples) == 43


async def test_rasa_file_importer_with_invalid_config():
    importer = RasaFileImporter(config_file="invalid path")
    actual = await importer.get_config()

    assert actual == {}


async def test_rasa_file_importer_with_invalid_domain(tmp_path: Path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("")
    importer = TrainingDataImporter.load_from_dict({}, str(config_file), None, [])

    actual = await importer.get_domain()
    assert actual.as_dict() == Domain.empty().as_dict()
