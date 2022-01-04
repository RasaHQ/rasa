import os
import logging
from pathlib import Path
from typing import Any, Text, Type
from unittest import mock

import pytest

from rasa.core.agent import Agent
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
import rasa.utils.common
from rasa.utils.common import (
    RepeatedLogFilter,
    find_unavailable_packages,
    configure_logging_and_warnings,
)
from rasa.shared.constants import DEFAULT_LOG_LEVEL
from rasa.constants import DEFAULT_LOG_LEVEL_LIBRARIES
import tests.conftest


def test_repeated_log_filter():
    log_filter = RepeatedLogFilter()
    record1 = logging.LogRecord(
        "rasa", logging.INFO, "/some/path.py", 42, "Super msg: %s", ("yes",), None
    )
    record1_same = logging.LogRecord(
        "rasa", logging.INFO, "/some/path.py", 42, "Super msg: %s", ("yes",), None
    )
    record2_other_args = logging.LogRecord(
        "rasa", logging.INFO, "/some/path.py", 42, "Super msg: %s", ("no",), None
    )
    record3_other = logging.LogRecord(
        "rasa", logging.INFO, "/some/path.py", 42, "Other msg", (), None
    )
    assert log_filter.filter(record1) is True
    assert log_filter.filter(record1_same) is False  # same log
    assert log_filter.filter(record2_other_args) is True
    assert log_filter.filter(record3_other) is True
    assert log_filter.filter(record1) is True  # same as before, but not repeated


async def test_call_maybe_coroutine_with_async() -> Any:
    expected = 5

    async def my_function():
        return expected

    actual = await rasa.utils.common.call_potential_coroutine(my_function())

    assert actual == expected


async def test_call_maybe_coroutine_with_sync() -> Any:
    expected = 5

    def my_function():
        return expected

    actual = await rasa.utils.common.call_potential_coroutine(my_function())

    assert actual == expected


def test_dir_size_empty(tmp_path: Path):
    assert rasa.utils.common.directory_size_in_mb(tmp_path) == 0


def test_dir_size_with_single_file(tmp_path: Path):
    tests.conftest.create_test_file_with_size(tmp_path, 5)
    assert rasa.utils.common.directory_size_in_mb(tmp_path) == pytest.approx(5)


def test_dir_size_with_multiple_files(tmp_path: Path):
    tests.conftest.create_test_file_with_size(tmp_path, 2)
    tests.conftest.create_test_file_with_size(tmp_path, 3)
    assert rasa.utils.common.directory_size_in_mb(tmp_path) == pytest.approx(5)


def test_dir_size_with_sub_directory(tmp_path: Path):
    subdir = tmp_path / "sub"
    subdir.mkdir()

    tests.conftest.create_test_file_with_size(tmp_path, 2)
    tests.conftest.create_test_file_with_size(subdir, 3)

    assert rasa.utils.common.directory_size_in_mb(tmp_path) == pytest.approx(5)


@pytest.mark.parametrize("create_destination", [True, False])
def test_copy_directory_with_created_destination(
    tmp_path: Path, create_destination: bool
):
    source = tmp_path / "source"
    source.mkdir()

    sub_dir_name = "sub"
    sub_dir = source / sub_dir_name
    sub_dir.mkdir()

    file_in_sub_dir_name = "file.txt"
    file_in_sub_dir = sub_dir / file_in_sub_dir_name
    file_in_sub_dir.touch()

    test_file_name = "some other file.txt"
    test_file = source / test_file_name
    test_file.touch()

    destination = tmp_path / "destination"
    if create_destination:
        destination.mkdir()

    rasa.utils.common.copy_directory(source, destination)

    assert destination.is_dir()
    assert (destination / test_file_name).is_file()
    assert (destination / sub_dir_name).is_dir()
    assert (destination / sub_dir_name / file_in_sub_dir_name).is_file()


def test_copy_directory_with_non_empty_destination(tmp_path: Path):
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "some_file.json").touch()

    with pytest.raises(ValueError):
        rasa.utils.common.copy_directory(tmp_path, destination)


def test_find_unavailable_packages():
    unavailable = find_unavailable_packages(
        ["my_made_up_package_name", "io", "foo_bar", "foo_bar"]
    )
    assert unavailable == {"my_made_up_package_name", "foo_bar"}


@pytest.mark.parametrize(
    "clazz,module_path",
    [
        (Path, "pathlib.Path"),
        (Agent, "rasa.core.agent.Agent"),
        (DIETClassifier, "rasa.nlu.classifiers.diet_classifier.DIETClassifier",),
    ],
)
def test_module_path_from_class(clazz: Type, module_path: Text):
    assert rasa.utils.common.module_path_from_class(clazz) == module_path


def test_override_defaults():
    defaults = {"nested-dict": {"key1": "value1", "key2": "value2"}}
    custom = {"nested-dict": {"key2": "override-value2"}}

    updated_config = rasa.utils.common.override_defaults(defaults, custom)

    expected_config = {"nested-dict": {"key1": "value1", "key2": "override-value2"}}
    assert updated_config == expected_config


def test_cli_missing_log_level_default_used():
    """Test CLI without log level parameter or env var uses default."""
    configure_logging_and_warnings()
    rasa_logger = logging.getLogger("rasa")
    # Default log level is currently INFO
    rasa_logger.level == logging.INFO
    matplotlib_logger = logging.getLogger("matplotlib")
    # Default log level for libraries is currently ERROR
    matplotlib_logger.level == logging.ERROR


def test_cli_log_level_debug_used():
    """Test CLI with log level uses for rasa logger whereas libraries stay default."""
    configure_logging_and_warnings(logging.DEBUG)
    rasa_logger = logging.getLogger("rasa")
    rasa_logger.level == logging.DEBUG
    matplotlib_logger = logging.getLogger("matplotlib")
    # Default log level for libraries is currently ERROR
    matplotlib_logger.level == logging.ERROR


@mock.patch.dict(os.environ, {"LOG_LEVEL": "WARNING"})
def test_cli_log_level_overrides_env_var_used():
    """Test CLI log level has precedence over env var."""
    configure_logging_and_warnings(logging.DEBUG)
    rasa_logger = logging.getLogger("rasa")
    rasa_logger.level == logging.DEBUG
    matplotlib_logger = logging.getLogger("matplotlib")
    # Default log level for libraries is currently ERROR
    matplotlib_logger.level == logging.ERROR


@mock.patch.dict(os.environ, {"LOG_LEVEL": "WARNING", "LOG_LEVEL_MATPLOTLIB": "INFO"})
def test_cli_missing_log_level_env_var_used():
    """Test CLI without log level uses env var for both rasa and libraries."""
    configure_logging_and_warnings()
    rasa_logger = logging.getLogger("rasa")
    rasa_logger.level == logging.WARNING
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.level == logging.INFO
