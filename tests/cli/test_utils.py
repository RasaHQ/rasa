import sys
import tempfile

import pytest

from rasa.cli.utils import (parse_last_positional_argument_as_model_path,
                            validate)


@pytest.mark.parametrize("argv",
                         [["rasa", "run"],
                          ["rasa", "run", "core"],
                          ["rasa", "test", "nlu", "--param", "xy"]])
def test_parse_last_positional_argument_as_model_path(argv):
    test_model_dir = tempfile.gettempdir()
    argv.append(test_model_dir)

    sys.argv = argv.copy()
    parse_last_positional_argument_as_model_path()

    assert sys.argv[-2] == "--model"
    assert sys.argv[-1] == test_model_dir


@pytest.mark.parametrize("argv",
                         [["rasa", "run"],
                          ["rasa", "run", "core"],
                          ["rasa", "test", "nlu", "--param", "xy", "--model",
                           "test"]])
def test_parse_no_positional_model_path_argument(argv):
    sys.argv = argv.copy()

    parse_last_positional_argument_as_model_path()

    print(sys.argv)
    assert sys.argv == argv


def test_validate_invalid_path():
    arguments = type('', (), {})()

    arguments.out = "test test test"

    with pytest.raises(SystemExit):
        validate(arguments, [("out", "default")])


def test_validate_valid_path():
    tempdir = tempfile.mkdtemp()
    arguments = type('', (), {})()
    arguments.out = tempdir

    validate(arguments, [("out", "default")])

    assert arguments.out == tempdir


def test_validate_if_none_is_valid():
    arguments = type('', (), {})()
    arguments.out = None

    validate(arguments, [("out", "default", True)])

    assert arguments.out is None


def test_validate_if_default_is_valid():
    tempdir = tempfile.mkdtemp()
    arguments = type('', (), {})()
    arguments.out = None

    validate(arguments, [("out", tempdir)])

    assert arguments.out == tempdir
