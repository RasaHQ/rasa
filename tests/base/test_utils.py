from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import os
import pickle
import tempfile

import pytest
from httpretty import httpretty
from rasa_nlu import utils
from rasa_nlu.utils import (
    relative_normpath, create_dir, is_url, ordered, is_model_dir, remove_model,
    write_json_to_file, write_to_file, EndpointConfig)


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
    open(os.path.join(tmpdir.strpath, ".hidden"), 'a').close()
    # create a normal file
    normal_file = os.path.join(tmpdir.strpath, "normal_file")
    open(normal_file, 'a').close()
    assert utils.list_files(tmpdir.strpath) == [normal_file]


def test_creation_of_existing_dir(tmpdir):
    # makes sure there is no exception
    assert create_dir(tmpdir.strpath) is None


def test_ordered():
    target = {"a": [1, 3, 2], "c": "a", "b": 1}
    assert ordered(target) == [('a', [1, 2, 3]), ('b', 1), ('c', 'a')]


@pytest.mark.parametrize(("model_dir", "expected"),
                         [("test_models/test_model_mitie/model_20170628-002704", True),
                          ("test_models/test_model_mitie_sklearn/model_20170628-002712", True),
                          ("test_models/test_model_spacy_sklearn/model_20170628-002705", True),
                          ("test_models/", False),
                          ("test_models/nonexistent_for_sure_123", False)])
def test_is_model_dir(model_dir, expected):
    assert is_model_dir(model_dir) == expected


def test_is_model_dir_empty(empty_model_dir):
    assert is_model_dir(empty_model_dir)


def test_remove_model_empty(empty_model_dir):
    assert remove_model(empty_model_dir)


def test_remove_model_with_files(empty_model_dir):
    metadata_file = "metadata.json"
    metadata_content = {"pipeline": "spacy_sklearn", "language": "en"}
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
    assert not is_url('./some/file/path')
    assert is_url('https://rasa.com/')


def test_endpoint_config():
    endpoint = EndpointConfig(
            "https://abc.defg/",
            params={"A": "B"},
            headers={"X-Powered-By": "Rasa"},
            basic_auth={"username": "user",
                        "password": "pass"},
            token="mytoken",
            token_name="letoken"
    )

    httpretty.register_uri(
            httpretty.POST,
            'https://abc.defg/test',
            status=500,
            body='')

    httpretty.enable()
    endpoint.request("post", subpath="test",
                     content_type="application/text",
                     json={"c": "d"},
                     params={"P": "1"})
    httpretty.disable()

    r = httpretty.latest_requests[-1]

    assert json.loads(str(r.body.decode("utf-8"))) == {"c": "d"}
    assert r.headers.get("X-Powered-By") == "Rasa"
    assert r.headers.get("Authorization") == "Basic dXNlcjpwYXNz"
    assert r.querystring.get("A") == ["B"]
    assert r.querystring.get("P") == ["1"]
    assert r.querystring.get("letoken") == ["mytoken"]


def test_environment_variable_not_existing():
    content = "model: \n  test: ${variable}"
    with pytest.raises(KeyError):
        utils.read_yaml(content)


def test_environment_variable_dict_without_prefix_and_postfix():
    os.environ['variable'] = 'test'
    content = "model: \n  test: ${variable}"

    result = utils.read_yaml(content)

    assert result['model']['test'] == 'test'


def test_environment_variable_in_list():
    os.environ['variable'] = 'test'
    content = "model: \n  - value\n  - ${variable}"

    result = utils.read_yaml(content)

    assert result['model'][1] == 'test'


def test_environment_variable_dict_with_prefix():
    os.environ['variable'] = 'test'
    content = "model: \n  test: dir/${variable}"

    result = utils.read_yaml(content)

    assert result['model']['test'] == 'dir/test'


def test_environment_variable_dict_with_postfix():
    os.environ['variable'] = 'test'
    content = "model: \n  test: ${variable}/dir"

    result = utils.read_yaml(content)

    assert result['model']['test'] == 'test/dir'


def test_environment_variable_dict_with_prefix_and_with_postfix():
    os.environ['variable'] = 'test'
    content = "model: \n  test: dir/${variable}/dir"

    result = utils.read_yaml(content)

    assert result['model']['test'] == 'dir/test/dir'
