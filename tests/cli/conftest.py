import shutil

from subprocess import check_call

from _pytest.tmpdir import TempdirFactory
from typing import Callable
import pytest
import os
from _pytest.pytester import Testdir, RunResult


@pytest.fixture
def run(testdir: Testdir) -> Callable[..., RunResult]:
    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_with_stdin(testdir: Testdir) -> Callable[..., RunResult]:
    def do_run(*args, stdin):
        args = ["rasa"] + list(args)
        return testdir.run(*args, stdin=stdin)

    return do_run


@pytest.fixture(scope="session")
def init_default_project(tmpdir_factory: TempdirFactory) -> str:
    path = tmpdir_factory.mktemp("agent").strpath
    os.environ["LOG_LEVEL"] = "ERROR"

    check_call(["rasa", "init", "--no-prompt"], cwd=path)
    return path


@pytest.fixture
def run_in_default_project(
    testdir: Testdir, init_default_project: str
) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "ERROR"

    # makes sure we do not always retrain an initial model for every "new" project
    for file_name in os.listdir(init_default_project):
        full_file_name = os.path.join(init_default_project, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, str(testdir.tmpdir))
        else:
            shutil.copytree(full_file_name, str(testdir.tmpdir / file_name))

    def do_run(*args):
        args = ["rasa"] + list(args)
        result = testdir.run(*args)
        os.environ["LOG_LEVEL"] = "INFO"
        return result

    return do_run
