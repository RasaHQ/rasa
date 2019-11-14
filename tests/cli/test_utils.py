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
from tests.conftest import assert_log_emitted


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


def test_validate_valid_path():
    tempdir = tempfile.mkdtemp()

    assert get_validated_path(tempdir, "out", "default") == tempdir


def test_validate_if_none_is_valid():
    assert get_validated_path(None, "out", "default", True) is None


def test_validate_with_none_if_default_is_valid(caplog: LogCaptureFixture):
    tempdir = tempfile.mkdtemp()

    with caplog.at_level(logging.WARNING, rasa.cli.utils.logger.name):
        assert get_validated_path(None, "out", tempdir) == tempdir

    assert caplog.records == []


def test_validate_with_invalid_directory_if_default_is_valid(caplog: LogCaptureFixture):
    tempdir = tempfile.mkdtemp()
    invalid_directory = "gcfhvjkb"
    with pytest.warns(UserWarning) as record:
        assert get_validated_path(invalid_directory, "out", tempdir) == tempdir
    assert len(record) == 1
    assert "does not exist" in record[0].message.args[0]


def test_print_error_and_exit():
    with pytest.raises(SystemExit):
        rasa.cli.utils.print_error_and_exit("")


def test_logging_capture(caplog: LogCaptureFixture):
    logger = logging.getLogger(__name__)

    # make a random INFO log and ensure it passes decorator
    info_text = "SOME INFO"
    logger.info(info_text)
    with assert_log_emitted(caplog, logger.name, logging.INFO, info_text):
        pass


def test_logging_capture_failure(caplog: LogCaptureFixture):
    logger = logging.getLogger(__name__)

    # make a random INFO log
    logger.info("SOME INFO")

    # test for string in log that wasn't emitted
    with pytest.raises(AssertionError):
        with assert_log_emitted(caplog, logger.name, logging.INFO, "NONONO"):
            pass
