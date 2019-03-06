import json
import os
import tempfile

import pytest

import rasa.data as data


def test_get_core_directory(project):
    data_dir = os.path.join(project, "data")
    core_directory = data.get_core_directory(data_dir)

    assert os.listdir(core_directory) == ["stories.md"]


def test_get_nlu_directory(project):
    data_dir = os.path.join(project, "data")
    nlu_directory = data.get_nlu_directory(data_dir)

    assert os.listdir(nlu_directory) == ["nlu.md"]


def test_get_core_nlu_directories(project):
    data_dir = os.path.join(project, "data")
    core_directory, nlu_directory = data.get_core_nlu_directories(data_dir)

    assert os.listdir(core_directory) == ["stories.md"]
    assert os.listdir(nlu_directory) == ["nlu.md"]


@pytest.mark.parametrize("line", [
    "##intent:aintent",
    "##synonym: synonym",
    "##regex:a_regex",
    " ##lookup:additional"])
def test_contains_nlu_pattern(line):
    assert data._contains_nlu_pattern(line)


def test_is_nlu_file_with_json():
    test = {"rasa_nlu_data": {"lookup_tables": [
        {"name": "plates", "elements": ["beans", "rice", "tacos", "cheese"]}]}}

    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")
    with open(file, "w") as f:
        f.write(json.dumps(test))

    assert data._is_nlu_file(file)


def test_is_not_nlu_file_with_json():
    directory = tempfile.mkdtemp()
    file = os.path.join(directory, "test.json")
    with open(file, "w") as f:
        f.write('{"test": "a"}')

    assert not data._is_nlu_file(file)


@pytest.mark.parametrize("line", [
    "- example",
    "## story intent 1 + two"
    "##slots"
    "* entry"])
def test_not_contains_nlu_pattern(line):
    assert not data._contains_nlu_pattern(line)
