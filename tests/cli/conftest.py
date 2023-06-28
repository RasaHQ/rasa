from pathlib import Path

from subprocess import check_call

from typing import Callable, Text
import pytest
import shutil
import os

from pytest import TempdirFactory, Testdir
from _pytest.pytester import RunResult

from rasa.cli import scaffold
from rasa.shared.utils.io import write_yaml

RASA_EXE = os.environ.get("RASA_EXECUTABLE", "rasa")


@pytest.fixture
def run(testdir: Testdir) -> Callable[..., RunResult]:
    def do_run(*args):
        args = [shutil.which(RASA_EXE)] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_with_stdin(testdir: Testdir) -> Callable[..., RunResult]:
    def do_run(*args, stdin):
        args = [shutil.which(RASA_EXE)] + list(args)
        return testdir.run(*args, stdin=stdin)

    return do_run


def create_simple_project(path: Path):
    scaffold.create_initial_project(str(path))

    # create a config file
    # for the cli test the resulting model is not important, use components that are
    # fast to train
    write_yaml(
        {
            "assistant_id": "placeholder_default",
            "language": "en",
            "pipeline": [{"name": "KeywordIntentClassifier"}],
            "policies": [
                {"name": "RulePolicy"},
                {"name": "MemoizationPolicy", "max_history": 3},
            ],
        },
        path / "config.yml",
    )
    return path


def create_simple_project_with_missing_assistant_id(path: Path):
    scaffold.create_initial_project(str(path))

    write_yaml(
        {
            "language": "en",
            "pipeline": [{"name": "KeywordIntentClassifier"}],
            "policies": [
                {"name": "RulePolicy"},
                {"name": "MemoizationPolicy", "max_history": 3},
            ],
        },
        path / "config.yml",
    )
    return path


@pytest.fixture(scope="session")
def trained_simple_project(tmpdir_factory: TempdirFactory) -> Text:
    path = tmpdir_factory.mktemp("simple")
    create_simple_project(path)

    os.environ["LOG_LEVEL"] = "ERROR"

    check_call([shutil.which(RASA_EXE), "train"], cwd=path.strpath)

    return path.strpath


@pytest.fixture
def run_in_simple_project(testdir: Testdir) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "ERROR"

    create_simple_project(testdir.tmpdir)

    def do_run(*args):
        args = [shutil.which(RASA_EXE)] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_in_simple_project_with_warnings(testdir: Testdir) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "WARNING"

    create_simple_project(testdir.tmpdir)

    def do_run(*args):
        args = [shutil.which(RASA_EXE)] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_in_simple_project_with_no_domain(testdir: Testdir) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "WARNING"

    create_simple_project(testdir.tmpdir)
    Path(testdir.tmpdir / "domain.yml").unlink()

    def do_run(*args):
        args = [shutil.which(RASA_EXE)] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_in_simple_project_with_model(
    testdir: Testdir, trained_simple_project: Text
) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "ERROR"

    # makes sure we do not always retrain an initial model for every "new" project
    for file_name in os.listdir(trained_simple_project):
        full_file_name = os.path.join(trained_simple_project, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, str(testdir.tmpdir))
        else:
            shutil.copytree(full_file_name, str(testdir.tmpdir / file_name))

    def do_run(*args):
        args = [shutil.which(RASA_EXE)] + list(args)
        result = testdir.run(*args)
        os.environ["LOG_LEVEL"] = "INFO"
        return result

    return do_run
