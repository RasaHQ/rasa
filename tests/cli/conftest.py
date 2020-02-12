from typing import Callable
import pytest
import shutil
import os
from _pytest.pytester import Testdir, RunResult

from rasa.utils.io import write_yaml_file


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


@pytest.fixture
def run_in_default_project_without_models(testdir: Testdir) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "ERROR"

    _set_up_initial_project(testdir)

    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_in_default_project(testdir: Testdir) -> Callable[..., RunResult]:
    os.environ["LOG_LEVEL"] = "ERROR"

    _set_up_initial_project(testdir)

    testdir.run("rasa", "train")

    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir.run(*args)

    return do_run


def _set_up_initial_project(testdir: Testdir):
    # copy initial project files
    testdir.copy_example("rasa/cli/initial_project/actions.py")
    testdir.copy_example("rasa/cli/initial_project/credentials.yml")
    testdir.copy_example("rasa/cli/initial_project/domain.yml")
    testdir.copy_example("rasa/cli/initial_project/endpoints.yml")
    testdir.mkdir("data")
    testdir.copy_example("rasa/cli/initial_project/data")
    testdir.run("mv", "nlu.md", "data/nlu.md")
    testdir.run("mv", "stories.md", "data/stories.md")

    # create a config file
    # for the cli test the resulting model is not important, use components that are
    # fast to train
    write_yaml_file(
        {
            "language": "en",
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {"name": "KeywordIntentClassifier"},
            ],
            "policies": [
                {"name": "MappingPolicy"},
                {"name": "MemoizationPolicy", "max_history": 5},
            ],
        },
        "config.yml",
    )
