import logging
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


def test_validate_with_none_if_default_is_valid(caplog: LogCaptureFixture):
    tempdir = tempfile.mkdtemp()

    with caplog.at_level(logging.WARNING, rasa.cli.utils.logger.name):
        assert get_validated_path(None, "out", tempdir) == tempdir

    assert caplog.records == []


def test_validate_with_invalid_directory_if_default_is_valid(caplog: LogCaptureFixture):
    tempdir = tempfile.mkdtemp()
    invalid_directory = "gcfhvjkb"

    with caplog.at_level(logging.WARNING, rasa.cli.utils.logger.name):
        assert get_validated_path(invalid_directory, "out", tempdir) == tempdir

    assert "'{}' does not exist".format(invalid_directory) in caplog.text


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
