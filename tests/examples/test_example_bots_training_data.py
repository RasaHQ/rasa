from typing import Text

import pytest

from rasa.importers.importer import TrainingDataImporter


@pytest.mark.parametrize(
    "config_file, domain_file, data_folder",
    [
        (
            "examples/concertbot/config.yml",
            "examples/concertbot/domain.yml",
            "examples/concertbot/data",
        ),
        (
            "examples/knowledgebasebot/config.yml",
            "examples/knowledgebasebot/domain.yml",
            "examples/knowledgebasebot/data",
        ),
        (
            "examples/moodbot/config.yml",
            "examples/moodbot/domain.yml",
            "examples/moodbot/data",
        ),
        (
            "examples/reminderbot/config.yml",
            "examples/reminderbot/domain.yml",
            "examples/reminderbot/data",
        ),
        (
            "examples/rules/config.yml",
            "examples/rules/domain.yml",
            "examples/rules/data",
        ),
        (
            "rasa/cli/initial_project/config.yml",
            "rasa/cli/initial_project/domain.yml",
            "rasa/cli/initial_project/data",
        ),
    ],
)
async def test_example_bot_training_data_not_raises(
    config_file: Text, domain_file: Text, data_folder: Text
):

    importer = TrainingDataImporter.load_from_config(
        config_file, domain_file, [data_folder]
    )

    with pytest.warns(None) as record:
        await importer.get_nlu_data()
        await importer.get_stories()

    assert not len(record)
