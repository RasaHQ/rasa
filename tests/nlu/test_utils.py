import io
import os
import pickle
import pytest
import tempfile
import shutil
import rasa.utils.io as io_utils
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


def test_relative_normpath():
    test_file = "/my/test/path/file.txt"
    assert utils.relative_normpath(test_file, "/my/test") == "path/file.txt"
    assert utils.relative_normpath(None, "/my/test") is None


def test_list_files_invalid_resource():
    with pytest.raises(ValueError) as execinfo:
        io_utils.list_files(None)
    assert "must be a string type" in str(execinfo.value)


def test_list_files_non_existing_dir():
    with pytest.raises(ValueError) as execinfo:
        io_utils.list_files("my/made_up/path")
    assert "Could not locate the resource" in str(execinfo.value)


def test_list_files_ignores_hidden_files(tmpdir):
    # create a hidden file
    open(os.path.join(tmpdir.strpath, ".hidden"), "a").close()
    # create a normal file
    normal_file = os.path.join(tmpdir.strpath, "normal_file")
    open(normal_file, "a").close()
    assert io_utils.list_files(tmpdir.strpath) == [normal_file]


def test_creation_of_existing_dir(tmpdir):
    # makes sure there is no exception
    assert io_utils.create_directory(tmpdir.strpath) is None


def test_ordered():
    target = {"a": [1, 3, 2], "c": "a", "b": 1}
    assert utils.ordered(target) == [("a", [1, 2, 3]), ("b", 1), ("c", "a")]


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

    with pytest.raises(ValueError):
        utils.remove_model(empty_model_dir)

    os.remove(test_file_path)


def test_is_url():
    assert not utils.is_url("./some/file/path")
    assert utils.is_url("https://rasa.com/")
