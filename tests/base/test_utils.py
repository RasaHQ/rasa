from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pytest

from rasa_nlu.utils import relative_normpath, recursively_find_files, create_dir, ordered


def test_relative_normpath():
    assert relative_normpath("/my/test/path/file.txt", "/my/test") == "path/file.txt"
    assert relative_normpath(None, "/my/test") is None


def test_recursively_find_files_invalid_resource():
    with pytest.raises(ValueError) as execinfo:
        recursively_find_files(None)
    assert "must be an existing directory" in str(execinfo.value)


def test_recursively_find_files_non_existing_dir():
    with pytest.raises(ValueError) as execinfo:
        recursively_find_files("my/made_up/path")
    assert "Could not locate the resource" in str(execinfo.value)


def test_creation_of_existing_dir(tmpdir):
    assert create_dir(tmpdir.strpath) is None   # makes sure there is no exception


def test_ordered():
    target = {"a": [1, 3, 2], "c": "a", "b": 1}
    assert ordered(target) == [('a', [1, 2, 3]), ('b', 1), ('c', 'a')]
