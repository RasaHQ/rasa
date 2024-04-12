import warnings
from pathlib import Path
from typing import Optional, Text

import pytest

from rasa.cli import scaffold
from rasa.shared.importers.importer import TrainingDataImporter
from tests.conftest import filter_expected_warnings


@pytest.mark.flaky
@pytest.mark.parametrize(
    "config_file, domain_file, data_folder, raise_slot_warning, msg",
    [
        (
            "examples/nlu_based/concertbot/config.yml",
            "examples/nlu_based/concertbot/domain.yml",
            "examples/nlu_based/concertbot/data",
            True,
            None,
        ),
        (
            "examples/nlu_based/formbot/config.yml",
            "examples/nlu_based/formbot/domain.yml",
            "examples/nlu_based/formbot/data",
            True,
            None,
        ),
        (
            "examples/nlu_based/knowledgebasebot/config.yml",
            "examples/nlu_based/knowledgebasebot/domain.yml",
            "examples/nlu_based/knowledgebasebot/data",
            True,
            "You are using an experimental feature: "
            "Action 'action_query_knowledge_base'!",
        ),
        (
            "data/test_moodbot/config.yml",
            "data/test_moodbot/domain.yml",
            "data/test_moodbot/data",
            False,
            None,
        ),
        (
            "examples/nlu_based/reminderbot/config.yml",
            "examples/nlu_based/reminderbot/domain.yml",
            "examples/nlu_based/reminderbot/data",
            True,
            None,
        ),
        (
            "examples/nlu_based/rules/config.yml",
            "examples/nlu_based/rules/domain.yml",
            "examples/nlu_based/rules/data",
            True,
            None,
        ),
    ],
)
def test_example_bot_training_data_raises_only_auto_fill_warning(
    config_file: Text,
    domain_file: Text,
    data_folder: Text,
    raise_slot_warning: bool,
    msg: Optional[Text],
):
    importer = TrainingDataImporter.load_from_config(
        config_file, domain_file, [data_folder]
    )

    if raise_slot_warning:
        with pytest.warns() as record:
            warnings.simplefilter(action="ignore", category=DeprecationWarning)

            if msg is not None:
                warnings.filterwarnings(action="ignore", message=msg)

            importer.get_nlu_data()
            importer.get_stories()

        assert len(record) == 2
        assert all(
            [
                "Slot auto-fill has been removed in 3.0 and replaced with "
                "a new explicit mechanism to set slots." in r.message.args[0]
                for r in record
            ]
        )
    else:
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
