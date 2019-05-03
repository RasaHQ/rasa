import sys
import tempfile

import pytest

from rasa.cli.utils import (
    parse_last_positional_argument_as_model_path,
    get_validated_path,
)


@pytest.mark.parametrize(
    "argv",
    [
        ["rasa", "run"],
        ["rasa", "run", "core"],
        ["rasa", "interactive", "nlu", "--param", "xy"],
    ],
)
def test_parse_last_positional_argument_as_model_path(argv):
    test_model_dir = tempfile.gettempdir()
    argv.append(test_model_dir)

    sys.argv = argv.copy()
    parse_last_positional_argument_as_model_path()

    assert sys.argv[-2] == "--model"
    assert sys.argv[-1] == test_model_dir


@pytest.mark.parametrize(
    "argv",
    [
        ["rasa", "run"],
        ["rasa", "run", "core"],
        ["rasa", "test", "nlu", "--param", "xy", "--model", "test"],
    ],
)
def test_parse_no_positional_model_path_argument(argv):
    sys.argv = argv.copy()

    parse_last_positional_argument_as_model_path()

    assert sys.argv == argv


def test_validate_invalid_path():
    with pytest.raises(SystemExit):
        get_validated_path("test test test", "out", "default")


def test_validate_valid_path():
    tempdir = tempfile.mkdtemp()

    assert get_validated_path(tempdir, "out", "default") == tempdir


def test_validate_if_none_is_valid():
    assert get_validated_path(None, "out", "default", True) is None


def test_validate_if_default_is_valid():
    tempdir = tempfile.mkdtemp()

    assert get_validated_path(None, "out", tempdir) == tempdir
