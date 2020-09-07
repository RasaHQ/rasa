from pathlib import Path
from typing import Callable, Text, List, Set

import pytest

import rasa.shared
from rasa.shared.utils import io as io_utils


def test_raise_user_warning():
    with pytest.warns(UserWarning) as record:
        io_utils.raise_warning("My warning.")

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."


def test_raise_future_warning():
    with pytest.warns(FutureWarning) as record:
        io_utils.raise_warning("My future warning.", FutureWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My future warning."


def test_raise_deprecation():
    with pytest.warns(DeprecationWarning) as record:
        io_utils.raise_warning("My warning.", DeprecationWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."
    assert isinstance(record[0].message, DeprecationWarning)


def test_read_file_with_not_existing_path():
    with pytest.raises(ValueError):
        rasa.shared.utils.io.read_file("some path")


@pytest.mark.parametrize(
    "list_function, expected",
    [
        (
            io_utils.list_directory,
            {"subdirectory", "subdirectory/sub_file.txt", "file.txt"},
        ),
        (io_utils.list_files, {"subdirectory/sub_file.txt", "file.txt"}),
        (io_utils.list_subdirectories, {"subdirectory"}),
    ],
)
def test_list_directory(
    tmpdir: Path, list_function: Callable[[Text], List[Text]], expected: Set[Text]
):
    subdirectory = tmpdir / "subdirectory"
    subdirectory.mkdir()

    sub_sub_directory = subdirectory / "subdirectory"
    sub_sub_directory.mkdir()

    sub_sub_file = sub_sub_directory / "sub_file.txt"
    sub_sub_file.write_text("", encoding=io_utils.DEFAULT_ENCODING)

    file1 = subdirectory / "file.txt"
    file1.write_text("", encoding="utf-8")

    hidden_directory = subdirectory / ".hidden"
    hidden_directory.mkdir()

    hidden_file = subdirectory / ".test.text"
    hidden_file.write_text("", encoding="utf-8")

    expected = {str(subdirectory / entry) for entry in expected}

    assert set(list_function(str(subdirectory))) == expected
