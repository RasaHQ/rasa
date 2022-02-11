from pathlib import Path
from typing import Text

import pytest

from rasa.cli import scaffold
from rasa.shared.importers.importer import TrainingDataImporter


@pytest.mark.parametrize(
    "config_file, domain_file, data_folder",
    [
        (
            "examples/concertbot/config.yml",
            "examples/concertbot/domain.yml",
            "examples/concertbot/data",
        ),
        (
            "examples/formbot/config.yml",
            "examples/formbot/domain.yml",
            "examples/formbot/data",
        ),
        (
            "examples/knowledgebasebot/config.yml",
            "examples/knowledgebasebot/domain.yml",
            "examples/knowledgebasebot/data",
        ),
        (
            "data/test_moodbot/config.yml",
            "data/test_moodbot/domain.yml",
            "data/test_moodbot/data",
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
def test_example_bot_training_data_raises_only_auto_fill_warning(
    config_file: Text, domain_file: Text, data_folder: Text
):

    importer = TrainingDataImporter.load_from_config(
        config_file, domain_file, [data_folder]
    )

    with pytest.warns(UserWarning) as record:
        importer.get_nlu_data()
        importer.get_stories()

    # 5 for slot auto-fill removal warning raised in `Domain.from_dict`
    # this method is called because `importer.get_nlu_data`
    # calls `CombinedDataImporter.get_domain()` which uses `Domain.from_dict`
    assert len(record) == 5
    assert all(
        [
            "Slot auto-fill has been removed in 3.0 and replaced with "
            "a new explicit mechanism to set slots." in r.message.args[0]
            for r in record
        ]
    )


def test_example_bot_training_on_initial_project(tmp_path: Path):
    # we need to test this one separately, as we can't test it in place
    # configuration suggestions would otherwise change the initial file
    scaffold.create_initial_project(str(tmp_path))

    importer = TrainingDataImporter.load_from_config(
        str(tmp_path / "config.yml"),
        str(tmp_path / "domain.yml"),
        str(tmp_path / "data"),
    )

    with pytest.warns(UserWarning) as record:
        importer.get_nlu_data()
        importer.get_stories()

    # 5 for slot auto-fill removal warning raised in `Domain.from_dict`
    # this method is called because `importer.get_nlu_data`
    # calls `CombinedDataImporter.get_domain()` which uses `Domain.from_dict`
    assert len(record) == 5
    assert all(
        [
            "Slot auto-fill has been removed in 3.0 and replaced with "
            "a new explicit mechanism to set slots." in r.message.args[0]
            for r in record
        ]
    )
