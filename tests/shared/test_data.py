import os
import tempfile
from pathlib import Path

import pytest
import rasa.shared

import rasa.shared.data
from rasa.shared.nlu.training_data.loading import load_data
from rasa.shared.utils.io import write_text_file, json_to_string
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)


@pytest.mark.parametrize(
    "path,is_yaml",
    [
        ("my_file.yaml", True),
        ("my_file.yml", True),
        ("/a/b/c/my_file.yml", True),
        ("/a/b/c/my_file.ml", False),
        ("my_file.json", False),
    ],
)
def test_is_yaml_file(path, is_yaml):
    assert rasa.shared.data.is_likely_yaml_file(path) == is_yaml


@pytest.mark.parametrize(
    "path,is_json",
    [
        ("my_file.json", True),
        ("/a/b/c/my_file.json", True),
        ("/a/b/c/my_file.yml", False),
    ],
)
def test_is_json_file(path, is_json):
    assert rasa.shared.data.is_likely_json_file(path) == is_json


def test_get_core_directory(project):
    data_dir = os.path.join(project, "data")
    core_directory = rasa.shared.data.get_core_directory([data_dir])
    core_files = os.listdir(core_directory)

    assert len(core_files) == 2
    assert any(file.endswith("stories.yml") for file in core_files)
    assert any(file.endswith("rules.yml") for file in core_files)


def test_get_nlu_directory(project):
    data_dir = os.path.join(project, "data")
    nlu_directory = rasa.shared.data.get_nlu_directory([data_dir])

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 1
    assert nlu_files[0].endswith("nlu.yml")


def test_get_nlu_file(project):
    data_file = os.path.join(project, "data/nlu.yml")
    nlu_directory = rasa.shared.data.get_nlu_directory(data_file)

    nlu_files = os.listdir(nlu_directory)

    original = load_data(data_file)
    copied = load_data(nlu_directory)

    assert nlu_files[0].endswith("nlu.yml")
    assert original.intent_examples == copied.intent_examples


def test_get_core_nlu_files(project):
    data_dir = os.path.join(project, "data")
    nlu_files = rasa.shared.data.get_data_files(
        [data_dir], rasa.shared.data.is_nlu_file
    )
    core_files = rasa.shared.data.get_data_files(
        [data_dir], YAMLStoryReader.is_stories_file
    )
    assert len(nlu_files) == 1
    assert list(nlu_files)[0].endswith("nlu.yml")

    assert len(core_files) == 2
    assert any(file.endswith("stories.yml") for file in core_files)
    assert any(file.endswith("rules.yml") for file in core_files)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            "dialogflow",
            [
                "data/examples/dialogflow/agent.json",
                "data/examples/dialogflow/entities/cuisine.json",
                "data/examples/dialogflow/entities/cuisine_entries_en.json",
                "data/examples/dialogflow/entities/cuisine_entries_es.json",
                "data/examples/dialogflow/entities/flightNumber.json",
                "data/examples/dialogflow/entities/flightNumber_entries_en.json",
                "data/examples/dialogflow/entities/location.json",
                "data/examples/dialogflow/entities/location_entries_en.json",
                "data/examples/dialogflow/entities/location_entries_es.json",
                "data/examples/dialogflow/intents/Default Fallback Intent.json",
                "data/examples/dialogflow/intents/affirm.json",
                "data/examples/dialogflow/intents/affirm_usersays_en.json",
                "data/examples/dialogflow/intents/affirm_usersays_es.json",
                "data/examples/dialogflow/intents/goodbye.json",
                "data/examples/dialogflow/intents/goodbye_usersays_en.json",
                "data/examples/dialogflow/intents/goodbye_usersays_es.json",
                "data/examples/dialogflow/intents/hi.json",
                "data/examples/dialogflow/intents/hi_usersays_en.json",
                "data/examples/dialogflow/intents/hi_usersays_es.json",
                "data/examples/dialogflow/intents/inform.json",
                "data/examples/dialogflow/intents/inform_usersays_en.json",
                "data/examples/dialogflow/intents/inform_usersays_es.json",
                "data/examples/dialogflow/package.json",
            ],
        ),
        ("luis", ["data/examples/luis/demo-restaurants_v7.json"]),
        (
            "rasa",
            [
                "data/examples/rasa/demo-rasa-multi-intent.yml",
                "data/examples/rasa/demo-rasa-responses.yml",
                "data/examples/rasa/demo-rasa.json",
                "data/examples/rasa/demo-rasa.yml",
            ],
        ),
        ("wit", ["data/examples/wit/demo-flights.json"]),
    ],
)
def test_find_nlu_files_with_different_formats(test_input, expected):
    examples_dir = "data/examples"
    data_dir = os.path.join(examples_dir, test_input)
    nlu_files = rasa.shared.data.get_data_files(
        [data_dir], rasa.shared.data.is_nlu_file
    )
    assert [Path(f) for f in nlu_files] == [Path(f) for f in expected]


def test_is_nlu_file_with_json():
    test = {
        "rasa_nlu_data": {
            "lookup_tables": [
                {"name": "plates", "elements": ["beans", "rice", "tacos", "cheese"]}
            ]
        }
    }

    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")

    write_text_file(json_to_string(test), file)

    assert rasa.shared.data.is_nlu_file(file)


def test_is_not_nlu_file_with_json():
    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")
    write_text_file('{"test": "a"}', file)

    assert not rasa.shared.data.is_nlu_file(file)


def test_get_story_file_with_yaml():
    examples_dir = "data/test_yaml_stories"
    core_files = rasa.shared.data.get_data_files(
        [examples_dir], YAMLStoryReader.is_stories_file
    )
    assert core_files
