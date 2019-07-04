import json
import os
import shutil
import tempfile

import pytest

import rasa.data as data
from tests.core.conftest import DEFAULT_STORIES_FILE, DEFAULT_NLU_DATA
from rasa.nlu.training_data import load_data
from rasa.nlu.utils import json_to_string


def test_get_core_directory(project):
    data_dir = os.path.join(project, "data")
    core_directory = data.get_core_directory([data_dir])
    stories = os.listdir(core_directory)

    assert len(stories) == 1
    assert stories[0].endswith("stories.md")


def test_get_nlu_directory(project):
    data_dir = os.path.join(project, "data")
    nlu_directory = data.get_nlu_directory([data_dir])

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 1
    assert nlu_files[0].endswith("nlu.md")


def test_get_nlu_file(project):
    data_file = os.path.join(project, "data/nlu.md")
    nlu_directory = data.get_nlu_directory(data_file)

    nlu_files = os.listdir(nlu_directory)

    original = load_data(data_file)
    copied = load_data(nlu_directory)

    assert nlu_files[0].endswith("nlu.md")
    assert original.intent_examples == copied.intent_examples


def test_get_core_nlu_files(project):
    data_dir = os.path.join(project, "data")
    core_files, nlu_files = data.get_core_nlu_files([data_dir])

    assert len(nlu_files) == 1
    assert list(nlu_files)[0].endswith("nlu.md")

    assert len(core_files) == 1
    assert list(core_files)[0].endswith("stories.md")


def test_get_core_nlu_directories(project):
    data_dir = os.path.join(project, "data")
    core_directory, nlu_directory = data.get_core_nlu_directories([data_dir])

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 1
    assert nlu_files[0].endswith("nlu.md")

    stories = os.listdir(core_directory)

    assert len(stories) == 1
    assert stories[0].endswith("stories.md")


def test_get_core_nlu_directories_with_none():
    directories = data.get_core_nlu_directories(None)

    assert all([directory for directory in directories])
    assert all([not os.listdir(directory) for directory in directories])


def test_same_file_names_get_resolved(tmpdir):
    # makes sure the resolution properly handles if there are two files with
    # with the same name in different directories

    tmpdir.join("one").mkdir()
    tmpdir.join("two").mkdir()
    data_dir_one = os.path.join(tmpdir.join("one").join("stories.md").strpath)
    data_dir_two = os.path.join(tmpdir.join("two").join("stories.md").strpath)
    shutil.copy2(DEFAULT_STORIES_FILE, data_dir_one)
    shutil.copy2(DEFAULT_STORIES_FILE, data_dir_two)

    nlu_dir_one = os.path.join(tmpdir.join("one").join("nlu.md").strpath)
    nlu_dir_two = os.path.join(tmpdir.join("two").join("nlu.md").strpath)
    shutil.copy2(DEFAULT_NLU_DATA, nlu_dir_one)
    shutil.copy2(DEFAULT_NLU_DATA, nlu_dir_two)

    core_directory, nlu_directory = data.get_core_nlu_directories([tmpdir.strpath])

    nlu_files = os.listdir(nlu_directory)

    assert len(nlu_files) == 2
    assert all([f.endswith("nlu.md") for f in nlu_files])

    stories = os.listdir(core_directory)

    assert len(stories) == 2
    assert all([f.endswith("stories.md") for f in stories])


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            "dialogflow",
            {
                "data/examples/dialogflow/entities/cuisine.json",
                "data/examples/dialogflow/intents/affirm.json",
                "data/examples/dialogflow/entities/location_entries_es.json",
                "data/examples/dialogflow/intents/affirm_usersays_en.json",
                "data/examples/dialogflow/intents/hi_usersays_es.json",
                "data/examples/dialogflow/entities/cuisine_entries_es.json",
                "data/examples/dialogflow/intents/inform_usersays_en.json",
                "data/examples/dialogflow/intents/hi.json",
                "data/examples/dialogflow/intents/goodbye_usersays_en.json",
                "data/examples/dialogflow/agent.json",
                "data/examples/dialogflow/intents/hi_usersays_en.json",
                "data/examples/dialogflow/entities/location.json",
                "data/examples/dialogflow/intents/affirm_usersays_es.json",
                "data/examples/dialogflow/entities/cuisine_entries_en.json",
                "data/examples/dialogflow/package.json",
                "data/examples/dialogflow/intents/Default Fallback Intent.json",
                "data/examples/dialogflow/intents/goodbye_usersays_es.json",
                "data/examples/dialogflow/intents/goodbye.json",
                "data/examples/dialogflow/entities/location_entries_en.json",
                "data/examples/dialogflow/intents/inform.json",
                "data/examples/dialogflow/intents/inform_usersays_es.json",
            },
        ),
        ("luis", {"data/examples/luis/demo-restaurants.json"}),
        (
            "rasa",
            {"data/examples/rasa/demo-rasa.json", "data/examples/rasa/demo-rasa.md"},
        ),
        ("wit", {"data/examples/wit/demo-flights.json"}),
    ],
)
def test_find_nlu_files_with_different_formats(test_input, expected):
    examples_dir = "data/examples"
    data_dir = os.path.join(examples_dir, test_input)
    core_files, nlu_files = data.get_core_nlu_files([data_dir])
    assert nlu_files == expected


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
    with open(file, "w", encoding="utf-8") as f:
        f.write(json_to_string(test))

    assert data._is_nlu_file(file)


def test_is_not_nlu_file_with_json():
    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")
    with open(file, "w", encoding="utf-8") as f:
        f.write('{"test": "a"}')

    assert not data._is_nlu_file(file)
