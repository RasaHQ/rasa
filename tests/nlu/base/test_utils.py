# -*- coding: utf-8 -*-
import io
import os
import pickle
import pytest
import tempfile

import rasa.utils.io
from rasa.nlu import utils
from rasa.nlu.utils import (
    create_dir,
    is_model_dir,
    is_url,
    ordered,
    relative_normpath,
    remove_model,
    write_json_to_file,
    write_to_file,
)
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def empty_model_dir(scope="function"):
    temp_path = tempfile.mkdtemp()
    yield temp_path
    if os.path.exists(temp_path):
        os.rmdir(temp_path)


def test_relative_normpath():
    test_file = "/my/test/path/file.txt"
    assert relative_normpath(test_file, "/my/test") == "path/file.txt"
    assert relative_normpath(None, "/my/test") is None


def test_list_files_invalid_resource():
    with pytest.raises(ValueError) as execinfo:
        utils.list_files(None)
    assert "must be a string type" in str(execinfo.value)


def test_list_files_non_existing_dir():
    with pytest.raises(ValueError) as execinfo:
        utils.list_files("my/made_up/path")
    assert "Could not locate the resource" in str(execinfo.value)


def test_list_files_ignores_hidden_files(tmpdir):
    # create a hidden file
    open(os.path.join(tmpdir.strpath, ".hidden"), "a").close()
    # create a normal file
    normal_file = os.path.join(tmpdir.strpath, "normal_file")
    open(normal_file, "a").close()
    assert utils.list_files(tmpdir.strpath) == [normal_file]


def test_creation_of_existing_dir(tmpdir):
    # makes sure there is no exception
    assert create_dir(tmpdir.strpath) is None


def test_ordered():
    target = {"a": [1, 3, 2], "c": "a", "b": 1}
    assert ordered(target) == [("a", [1, 2, 3]), ("b", 1), ("c", "a")]


@pytest.mark.parametrize(
    ("model_dir", "expected"),
    [
        ("test_models/test_model_mitie/model_20170628-002704", True),
        ("test_models/test_model_mitie_sklearn/model_20170628-002712", True),
        ("test_models/test_model_spacy_sklearn/model_20170628-002705", True),
        ("test_models/", False),
        ("test_models/nonexistent_for_sure_123", False),
    ],
)
def test_is_model_dir(model_dir, expected):
    assert is_model_dir(model_dir) == expected


def test_is_model_dir_empty(empty_model_dir):
    assert is_model_dir(empty_model_dir)


def test_remove_model_empty(empty_model_dir):
    assert remove_model(empty_model_dir)


def test_remove_model_with_files(empty_model_dir):
    metadata_file = "metadata.json"
    metadata_content = {"pipeline": "pretrained_embeddings_spacy", "language": "en"}
    metadata_path = os.path.join(empty_model_dir, metadata_file)
    write_json_to_file(metadata_path, metadata_content)

    fake_obj = {"Fake", "model"}
    fake_obj_path = os.path.join(empty_model_dir, "component.pkl")
    with io.open(fake_obj_path, "wb") as f:
        pickle.dump(fake_obj, f)

    assert remove_model(empty_model_dir)


def test_remove_model_invalid(empty_model_dir):
    test_file = "something.else"
    test_content = "Some other stuff"
    test_file_path = os.path.join(empty_model_dir, test_file)
    write_to_file(test_file_path, test_content)

    with pytest.raises(ValueError) as e:
        remove_model(empty_model_dir)

    os.remove(test_file_path)


def test_is_url():
    assert not is_url("./some/file/path")
    assert is_url("https://rasa.com/")


def test_environment_variable_not_existing():
    content = "model: \n  test: ${variable}"
    with pytest.raises(ValueError):
        rasa.utils.io.read_yaml(content)


def test_environment_variable_dict_without_prefix_and_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: ${variable}"

    result = rasa.utils.io.read_yaml(content)

    assert result["model"]["test"] == "test"


def test_environment_variable_in_list():
    os.environ["variable"] = "test"
    content = "model: \n  - value\n  - ${variable}"

    result = rasa.utils.io.read_yaml(content)

    assert result["model"][1] == "test"


def test_environment_variable_dict_with_prefix():
    os.environ["variable"] = "test"
    content = "model: \n  test: dir/${variable}"

    result = rasa.utils.io.read_yaml(content)

    assert result["model"]["test"] == "dir/test"


def test_environment_variable_dict_with_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: ${variable}/dir"

    result = rasa.utils.io.read_yaml(content)

    assert result["model"]["test"] == "test/dir"


def test_environment_variable_dict_with_prefix_and_with_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: dir/${variable}/dir"

    result = rasa.utils.io.read_yaml(content)

    assert result["model"]["test"] == "dir/test/dir"


def test_emojis_in_yaml():
    test_data = """
    data:
        - one 😁💯 👩🏿‍💻👨🏿‍💻
        - two £ (?u)\\b\\w+\\b f\u00fcr
    """
    actual = rasa.utils.io.read_yaml(test_data)

    assert actual["data"][0] == "one 😁💯 👩🏿‍💻👨🏿‍💻"
    assert actual["data"][1] == "two £ (?u)\\b\\w+\\b für"


def test_emojis_in_tmp_file():
    test_data = """
        data:
            - one 😁💯 👩🏿‍💻👨🏿‍💻
            - two £ (?u)\\b\\w+\\b f\u00fcr
        """
    test_file = utils.create_temporary_file(test_data)
    with io.open(test_file, mode="r", encoding="utf-8") as f:
        content = f.read()
    actual = rasa.utils.io.read_yaml(content)

    assert actual["data"][0] == "one 😁💯 👩🏿‍💻👨🏿‍💻"
    assert actual["data"][1] == "two £ (?u)\\b\\w+\\b für"


def test_read_emojis_from_json():
    import json

    d = {"text": "hey 😁💯 👩🏿‍💻👨🏿‍💻🧜‍♂️(?u)\\b\\w+\\b} f\u00fcr"}
    json_string = json.dumps(d, indent=2)

    s = rasa.utils.io.read_yaml(json_string)

    expected = "hey 😁💯 👩🏿‍💻👨🏿‍💻🧜‍♂️(?u)\\b\\w+\\b} für"
    assert s.get("text") == expected


def test_bool_str():
    test_data = """
    one: "yes"
    two: "true"
    three: "True"
    """

    actual = rasa.utils.io.read_yaml(test_data)

    assert actual["one"] == "yes"
    assert actual["two"] == "true"
    assert actual["three"] == "True"


def test_default_token_name():
    test_data = {"url": "http://test", "token": "token"}

    actual = EndpointConfig.from_dict(test_data)

    assert actual.token_name == "token"


def test_custom_token_name():
    test_data = {"url": "http://test", "token": "token", "token_name": "test_token"}

    actual = EndpointConfig.from_dict(test_data)

    assert actual.token_name == "test_token"
