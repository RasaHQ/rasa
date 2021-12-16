import pytest
import shutil
from rasa.cli import scaffold
from _pytest.tmpdir import TempdirFactory


@pytest.fixture(scope="session")
def simple_project(tmpdir_factory: TempdirFactory):
    path = tmpdir_factory.mktemp("simple")
    scaffold.create_initial_project(str(path))
    return path


@pytest.fixture(scope="session")
def formbot_project(tmpdir_factory: TempdirFactory):
    path = tmpdir_factory.mktemp("formbot") / "content"
    shutil.copytree("examples/formbot", path)
    return path
