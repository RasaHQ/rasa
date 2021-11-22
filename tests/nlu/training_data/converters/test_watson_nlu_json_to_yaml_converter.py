import os
from pathlib import Path

import pytest

from rasa.nlu.training_data.converters.watson_nlu_json_to_yaml_converter import (
    WatsonTrainingDataConverter,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
import yaml


def test_filter():
    source = Path("data/examples/watson/demo_watson_v2.json")
    filter = WatsonTrainingDataConverter().filter(source)
    if filter:
        assert True


def test_not_filter():
    source = Path("data/examples/luis/demo-restaurants_v7.json")
    filter = WatsonTrainingDataConverter().filter(source)
    if not filter:
        assert True


async def test_convert_and_write(tmp_path: Path):
    source = Path("data/examples/watson/demo_watson_v2.json")
    converted_data_folder = tmp_path / "converted_data"
    converted_data_folder.mkdir()

    with pytest.warns(None) as warnings:
        await WatsonTrainingDataConverter().convert_and_write(
            source, converted_data_folder
        )
    assert not warnings
    assert len(os.listdir(converted_data_folder)) == 1
    converted_file = converted_data_folder / "demo_watson_v2_converted.yml"
    content = converted_file.read_text()
    training_data = yaml.safe_load(content)
    assert training_data["version"] == LATEST_TRAINING_DATA_FORMAT_VERSION
    assert training_data["nlu"] is not None
