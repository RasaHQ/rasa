import logging
from pathlib import Path
from typing import Any

import pytest

import rasa.utils.common
from rasa.utils.common import RepeatedLogFilter
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
