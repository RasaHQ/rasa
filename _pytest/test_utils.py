from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pytest

from rasa_nlu.utils import relative_normpath, recursively_find_files


def test_relative_normpath():
    assert relative_normpath("/my/test/path/file.txt", "/my/test") == "path/file.txt"


def test_recursively_find_files_invalid_resource():
    with pytest.raises(ValueError) as execinfo:
        recursively_find_files(None)
    assert "must be an existing directory" in str(execinfo.value)


def test_recursively_find_files_non_existing_dir():
    with pytest.raises(ValueError) as execinfo:
        recursively_find_files("my/made_up/path")
    assert "Could not locate the resource" in str(execinfo.value)
