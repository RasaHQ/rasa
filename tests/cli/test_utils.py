import contextlib
import logging
import os
import pathlib
import sys
import tempfile

import pytest
from _pytest.logging import LogCaptureFixture

import rasa.cli.utils
from rasa.cli.utils import (
    parse_last_positional_argument_as_model_path,
    get_validated_path,
)


@contextlib.contextmanager
def make_actions_subdir():
    """Create a subdir called actions to test model argument handling."""
    with tempfile.TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        try:
            (pathlib.Path(tempdir) / "actions").mkdir()
            yield
        finally:
            os.chdir(cwd)


@pytest.mark.parametrize(
    "argv",
    [
        ["rasa", "run"],
        ["rasa", "run", "actions"],
        ["rasa", "run", "core"],
        ["rasa", "interactive", "nlu", "--param", "xy"],
    ],
)
def test_parse_last_positional_argument_as_model_path(argv):
    with make_actions_subdir():
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
        ["rasa", "run", "actions"],
        ["rasa", "run", "core"],
        ["rasa", "test", "nlu", "--param", "xy", "--model", "test"],
    ],
)
def test_parse_no_positional_model_path_argument(argv):
    with make_actions_subdir():
        sys.argv = argv.copy()

        parse_last_positional_argument_as_model_path()

        assert sys.argv == argv


def test_validate_invalid_path():
    with pytest.raises(SystemExit):
        get_validated_path("test test test", "out", "default")


def test_validate_valid_path(tmp_path: pathlib.Path):
    assert get_validated_path(str(tmp_path), "out", "default") == str(tmp_path)


def test_validate_if_none_is_valid():
    assert get_validated_path(None, "out", "default", True) is None


def test_validate_with_none_if_default_is_valid(
    caplog: LogCaptureFixture, tmp_path: pathlib.Path
):
    with caplog.at_level(logging.WARNING, rasa.cli.utils.logger.name):
        assert get_validated_path(None, "out", str(tmp_path)) == str(tmp_path)

    assert caplog.records == []


def test_validate_with_invalid_directory_if_default_is_valid(tmp_path: pathlib.Path):
    invalid_directory = "gcfhvjkb"
    with pytest.warns(UserWarning) as record:
        assert get_validated_path(invalid_directory, "out", str(tmp_path)) == str(
            tmp_path
        )
    assert len(record) == 1
    assert "does not seem to exist" in record[0].message.args[0]


def test_print_error_and_exit():
    with pytest.raises(SystemExit):
        rasa.cli.utils.print_error_and_exit("")
