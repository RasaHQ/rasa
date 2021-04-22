import os
import pickle
import pytest
import tempfile
import shutil
from typing import Text

import rasa.shared.nlu.training_data.message
import rasa.shared.utils.io
from rasa.nlu import utils


@pytest.fixture(scope="function")
def empty_model_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def fake_model_dir(empty_model_dir):
    metadata_file = "metadata.json"
    metadata_content = {"pipeline": "pretrained_embeddings_spacy", "language": "en"}
    metadata_path = os.path.join(empty_model_dir, metadata_file)
    utils.write_json_to_file(metadata_path, metadata_content)

    fake_obj = {"Fake", "model"}
    fake_obj_path = os.path.join(empty_model_dir, "component.pkl")
    with open(fake_obj_path, "wb") as f:
        pickle.dump(fake_obj, f)
    return empty_model_dir  # not empty anymore ;)


def test_list_files_invalid_resource():
    with pytest.raises(ValueError) as execinfo:
        rasa.shared.utils.io.list_files(None)
    assert "must be a string type" in str(execinfo.value)


def test_list_files_non_existing_dir():
    with pytest.raises(ValueError) as execinfo:
        rasa.shared.utils.io.list_files("my/made_up/path")
    assert "Could not locate the resource" in str(execinfo.value)


def test_list_files_ignores_hidden_files(tmpdir):
    # create a hidden file
    open(os.path.join(tmpdir.strpath, ".hidden"), "a").close()
    # create a normal file
    normal_file = os.path.join(tmpdir.strpath, "normal_file")
    open(normal_file, "a").close()
    assert rasa.shared.utils.io.list_files(tmpdir.strpath) == [normal_file]


def test_creation_of_existing_dir(tmpdir):
    # makes sure there is no exception
    assert rasa.shared.utils.io.create_directory(tmpdir.strpath) is None


@pytest.mark.parametrize(
    "url, result",
    [
        ("a/b/c", False),
        ("a", False),
        ("https://192.168.1.1", True),
        ("http://192.168.1.1", True),
        ("https://google.com", True),
        ("https://www.google.com", True),
        ("http://google.com", True),
        ("http://www.google.com", True),
        ("http://www.google.com?foo=bar", True),
        ("http://a/b/c", True),
        ("http://localhost:5002/api/projects/default/models/tags/production", True),
        ("http://rasa-x:5002/api/projects/default/models/tags/production", True),
        (
            "http://rasa-x:5002/api/projects/default/models/tags/production?foo=bar",
            True,
        ),
        ("file:///some/path/file", True),
    ],
)
def test_is_url(url: Text, result: bool):
    assert result == utils.is_url(url)
