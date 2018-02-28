from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from rasa_nlu.utils import (
    relative_normpath,
    recursively_find_files, create_dir, ordered)


def test_relative_normpath():
    test_file = "/my/test/path/file.txt"
    assert relative_normpath(test_file, "/my/test") == "path/file.txt"
    assert relative_normpath(None, "/my/test") is None


def test_recursively_find_files_invalid_resource():
    with pytest.raises(ValueError) as execinfo:
        recursively_find_files(None)
    assert "must be a string type" in str(execinfo.value)


def test_recursively_find_files_non_existing_dir():
    with pytest.raises(ValueError) as execinfo:
        recursively_find_files("my/made_up/path")
    assert "Could not locate the resource" in str(execinfo.value)


def test_creation_of_existing_dir(tmpdir):
    # makes sure there is no exception
    assert create_dir(tmpdir.strpath) is None


def test_ordered():
    target = {"a": [1, 3, 2], "c": "a", "b": 1}
    assert ordered(target) == [('a', [1, 2, 3]), ('b', 1), ('c', 'a')]
