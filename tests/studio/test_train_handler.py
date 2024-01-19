import os
from pathlib import Path

import pytest
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.io import read_yaml_file

from rasa.studio.config import StudioConfig
from rasa.studio.data_handler import StudioDataHandler
from rasa.studio.train import _create_temp_file, make_training_file


@pytest.fixture
def handler() -> StudioDataHandler:
    return StudioDataHandler(
        StudioConfig(
            authentication_server_url="http://studio.amazonaws.com",
            studio_url="http://studio.amazonaws.com",
            realm_name="rasa-test",
            client_id="rasa-cli",
        ),
        "test",
    )


def test_make_training_file(handler: StudioDataHandler) -> None:
    domain_path = "tests/data/train/domain.yml"
    nlu_dir = "tests/data/train/data"
    data1 = TrainingDataImporter.load_from_dict(
        domain_path=domain_path, training_data_paths=[nlu_dir]
    )
    handler.nlu_assistant = True
    file = make_training_file(handler, data1, data1)
    assert Path(file).exists()

    data_test = TrainingDataImporter.load_from_dict(
        domain_path=domain_path, training_data_paths=[str(file)]
    )
    # assert the data is the same as the original data
    # since the data is the same, the training data file is the same
    assert data_test.get_nlu_data().intents == data1.get_nlu_data().intents
    assert (
        data_test.get_nlu_data().intent_examples == data1.get_nlu_data().intent_examples
    )


def test_create_temp_file() -> None:
    temp_file_path = _create_temp_file(
        """test:
            foo: bar""",
        "test.yml",
    )
    assert os.path.exists(temp_file_path)
    assert (
        read_yaml_file(temp_file_path)
        == """test:
            foo: bar"""
    )
