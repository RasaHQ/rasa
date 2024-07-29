import pathlib
from pathlib import Path

from subprocess import check_call

from typing import Callable, Text
import pytest
import shutil
import os

from pytest import TempPathFactory, Testdir
from _pytest.pytester import RunResult

from rasa.cli import scaffold
from rasa.shared.utils.yaml import write_yaml
from tests.conftest import create_simple_project

RASA_EXE = os.environ.get("RASA_EXECUTABLE", "rasa")


@pytest.fixture
def run(testdir: Testdir) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "DEBUG"

    def do_run(*args):
        args = [shutil.which(RASA_EXE)] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_with_stdin(testdir: Testdir) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "DEBUG"

    def do_run(*args, stdin):
        args = [shutil.which(RASA_EXE)] + list(args)
        return testdir.run(*args, stdin=stdin)

    return do_run


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
def trained_simple_project(tmp_path_factory: TempPathFactory) -> Text:
    path = tmp_path_factory.mktemp("simple")
    create_simple_project(path)

    os.environ["LOG_LEVEL"] = "DEBUG"

    # deepcode ignore CommandInjection/test: Very low risk - an exploit would first need to compromise the host running the tests and maliciously edit the environment variables.
    check_call([shutil.which(RASA_EXE), "train"], cwd=path)

    return str(path)


@pytest.fixture
def run_in_simple_project_with_warnings(testdir: Testdir) -> Callable[..., RunResult]:
    # Check for all the logs by default, logs can always be filtered out
    os.environ["LOG_LEVEL"] = "DEBUG"

    create_simple_project(testdir.tmpdir)

    def do_run(*args):
        args = [shutil.which(RASA_EXE)] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_in_simple_project_with_no_domain(testdir: Testdir) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "DEBUG"

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
    os.environ["LOG_LEVEL"] = "DEBUG"

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
        return result

    return do_run


@pytest.fixture
def e2e_input_folder() -> pathlib.Path:
    return (
        pathlib.Path(__file__).parent.parent.parent
        / "data"
        / "end_to_end_testing_input_files"
    )
