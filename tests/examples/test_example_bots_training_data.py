from pathlib import Path
from typing import Text

import pytest

from rasa.cli import scaffold
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


async def test_example_bot_training_on_initial_project(tmp_path: Path):
    # we need to test this one separately, as we can't test it in place
    # configuration suggestions would otherwise change the initial file
    scaffold.create_initial_project(str(tmp_path))

    importer = TrainingDataImporter.load_from_config(
        str(tmp_path / "config.yml"),
        str(tmp_path / "domain.yml"),
        str(tmp_path / "data"),
    )

    with pytest.warns(None) as record:
        await importer.get_nlu_data()
        await importer.get_stories()

    assert not len(record)
