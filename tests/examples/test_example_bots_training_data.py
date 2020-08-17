from pathlib import Path
from typing import Text, List

import pytest

from rasa.importers.importer import TrainingDataImporter
from rasa.importers.utils import story_graph_from_paths

EXAMPLES_FOLDER_ROOT = "examples"
EXAMPLES_FOLDER_SUFFIX = "data"
CONFIG_FILE = "config.yml"
DOMAIN_FILE = "domain.yml"


def get_training_data_folder(bot_name: Text) -> List[Text]:
    return [str(Path(EXAMPLES_FOLDER_ROOT) / bot_name / EXAMPLES_FOLDER_SUFFIX)]


def get_config_file(bot_name: Text) -> Text:
    return str(Path(EXAMPLES_FOLDER_ROOT) / bot_name / CONFIG_FILE)


def get_domain_file(bot_name: Text) -> Text:
    return str(Path(EXAMPLES_FOLDER_ROOT) / bot_name / DOMAIN_FILE)


@pytest.mark.parametrize(
    "bot_name", ["concertbot", "knowledgebasebot", "moodbot", "reminderbot", "rules"]
)
async def test_example_bot_training_data_not_raises(bot_name: Text):

    importer = TrainingDataImporter.load_from_config(
        get_config_file(bot_name),
        get_domain_file(bot_name),
        get_training_data_folder(bot_name),
    )

    with pytest.warns(None) as record:
        await importer.get_nlu_data()
        await importer.get_stories()

    assert not len(record)
