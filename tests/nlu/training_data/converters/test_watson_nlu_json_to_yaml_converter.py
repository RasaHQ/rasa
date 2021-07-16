import os
from pathlib import Path
from typing import Text

import pytest

from rasa.nlu.training_data.converters.watson_nlu_json_to_yaml_converter import (
    WatsonTrainingDataConverter,
)
from rasa.shared.nlu.training_data.loading import load_data


def test_watson_data():
    td = load_data("data/examples/watson/demo_watson_v2.json")
    assert not td.is_empty()
    assert len(td.entity_examples) == 117
    assert len(td.intent_examples) == 309


def test_filter():
    source = Path("data/examples/watson/demo_watson_v2.json")
    filter = WatsonTrainingDataConverter().filter(source)
    assert filter == True


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
