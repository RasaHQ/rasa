import warnings
from pathlib import Path

import pytest

from rasa.cli import scaffold
from rasa.shared.importers.importer import TrainingDataImporter
from tests.conftest import filter_expected_warnings


@pytest.mark.flaky
def test_example_bot_training_data_does_not_raise_warnings() -> None:
    importer = TrainingDataImporter.load_from_config(
        "data/test_moodbot/config.yml",
        "data/test_moodbot/domain.yml",
        ["data/test_moodbot/data"],
    )

    with warnings.catch_warnings() as record:
        importer.get_nlu_data()
        importer.get_stories()

    assert record is None


def test_example_bot_training_on_initial_project(tmp_path: Path):
    # we need to test this one separately, as we can't test it in place
    # configuration suggestions would otherwise change the initial file
    scaffold.create_initial_project(str(tmp_path))

    importer = TrainingDataImporter.load_from_config(
        str(tmp_path / "config.yml"),
        str(tmp_path / "domain.yml"),
        str(tmp_path / "data"),
    )

    with warnings.catch_warnings() as record:
        importer.get_nlu_data()
        importer.get_stories()

    if record is not None:
        records = filter_expected_warnings(record)
        assert len(records) == 0
