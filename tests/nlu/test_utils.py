import io
import os
import pickle
import pytest
import tempfile
import shutil
from typing import Text

from rasa.shared.exceptions import RasaException
import rasa.shared.nlu.training_data.message
import rasa.shared.utils.io
import rasa.utils.io as io_utils
from rasa.nlu import utils
from pathlib import Path


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


def test_relative_normpath():
    test_file = "/my/test/path/file.txt"
    assert utils.relative_normpath(test_file, "/my/test") == Path("path/file.txt")
    assert utils.relative_normpath(None, "/my/test") is None


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


def test_empty_is_model_dir(empty_model_dir):
    assert utils.is_model_dir(empty_model_dir)


def test_non_existent_folder_is_no_model_dir():
    assert not utils.is_model_dir("nonexistent_for_sure_123/")


def test_data_folder_is_no_model_dir():
    assert not utils.is_model_dir("data/")


def test_model_folder_is_model_dir(fake_model_dir):
    assert utils.is_model_dir(fake_model_dir)


def test_remove_model_empty(empty_model_dir):
    assert utils.remove_model(empty_model_dir)


def test_remove_model_with_files(fake_model_dir):
    assert utils.remove_model(fake_model_dir)


def test_remove_model_invalid(empty_model_dir):
    test_file = "something.else"
    test_content = "Some other stuff"
    test_file_path = os.path.join(empty_model_dir, test_file)
    utils.write_to_file(test_file_path, test_content)

    with pytest.raises(RasaException):
        utils.remove_model(empty_model_dir)

    os.remove(test_file_path)


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
        ("http://a/b/c", False),
    ],
)
def test_is_url(url: Text, result: bool):
    assert result == utils.is_url(url)
