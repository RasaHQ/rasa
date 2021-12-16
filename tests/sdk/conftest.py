import pytest
import shutil
import os

from pathlib import Path
from rasa.cli import scaffold
from subprocess import check_call
from _pytest.tmpdir import TempdirFactory
from typing import Text


@pytest.fixture(scope="session")
def simple_project(tmpdir_factory: TempdirFactory):
    path = tmpdir_factory.mktemp("simple")
    scaffold.create_initial_project(str(path))
    return path
