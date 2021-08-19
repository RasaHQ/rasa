import copy
from pathlib import Path

import numpy as np
import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError

import rasa.shared.utils.io
import rasa.utils.io as io_utils

from rasa.shared.nlu.training_data.features import Features


@pytest.mark.parametrize("file, parents", [("A/test.md", "A"), ("A", "A")])
def test_file_in_path(file, parents):
    assert rasa.shared.utils.io.is_subdirectory(file, parents)


@pytest.mark.parametrize(
    "file, parents", [("A", "A/B"), ("B", "A"), ("A/test.md", "A/B"), (None, "A")]
)
def test_file_not_in_path(file, parents):
    assert not rasa.shared.utils.io.is_subdirectory(file, parents)


@pytest.mark.parametrize("actual_path", ["", "file.md", "file"])
def test_file_path_validator_with_invalid_paths(actual_path):
    test_error_message = actual_path

    validator = io_utils.file_type_validator([".yml"], test_error_message)

    document = Document(actual_path)
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == test_error_message


@pytest.mark.parametrize("actual_path", ["domain.yml", "lala.yaml"])
def test_file_path_validator_with_valid_paths(actual_path):
    validator = io_utils.file_type_validator([".yml", ".yaml"], "error message")

    document = Document(actual_path)
    # If the path is valid there shouldn't be an exception
    assert validator.validate(document) is None


@pytest.mark.parametrize("user_input", ["", "   ", "\t", "\n"])
def test_non_empty_text_validator_with_empty_input(user_input):
    test_error_message = "enter something"

    validator = io_utils.not_empty_validator(test_error_message)

    document = Document(user_input)
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == test_error_message


@pytest.mark.parametrize("user_input", ["utter_greet", "greet", "Hi there!"])
def test_non_empty_text_validator_with_valid_input(user_input):
    validator = io_utils.not_empty_validator("error message")

    document = Document(user_input)
    # If there is input there shouldn't be an exception
    assert validator.validate(document) is None


def test_create_validator_from_callable():
    def is_valid(user_input) -> None:
        return user_input == "this passes"

    error_message = "try again"

    validator = io_utils.create_validator(is_valid, error_message)

    document = Document("this passes")
    assert validator.validate(document) is None

    document = Document("this doesn't")
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == error_message


def test_write_utf_8_yaml_file(tmp_path: Path):
    """This test makes sure that dumping a yaml doesn't result in Uxxxx sequences
    but rather directly dumps the unicode character."""

    file_path = str(tmp_path / "test.yml")
    data = {"data": "amazing 🌈"}

    rasa.shared.utils.io.write_yaml(data, file_path)
    assert rasa.shared.utils.io.read_file(file_path) == "data: amazing 🌈\n"


@pytest.mark.parametrize(
    "container",
    [
        {},
        {"hello": "world"},
        {1: 2},
        {"foo": ["bar"]},
        {"a": []},
        [],
        ["a"],
        [{}],
        [None],
    ],
)
def test_fingerprint_containers(container):
    assert rasa.shared.utils.io.deep_container_fingerprint(
        container
    ) == rasa.shared.utils.io.deep_container_fingerprint(copy.deepcopy(container))


def test_deep_container_fingerprint_can_use_instance_fingerprint():
    m1 = np.asarray([[0.5, 3.1, 3.0], [1.1, 1.2, 1.3], [4.7, 0.3, 2.7]])
    f = Features(m1, "sentence", "text", "CountVectorsFeaturizer")
    assert rasa.shared.utils.io.deep_container_fingerprint(f) == f.fingerprint()
