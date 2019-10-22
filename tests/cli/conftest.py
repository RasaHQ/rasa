from typing import Callable, Any, Tuple
import pytest
import os
from _pytest.pytester import Testdir, RunResult


@pytest.fixture
def run(testdir: Testdir) -> Callable[[Tuple[Any]], RunResult]:
    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_in_default_project(testdir: Testdir) -> Callable[[Tuple[Any]], RunResult]:
    os.environ["LOG_LEVEL"] = "ERROR"
    testdir.run("rasa", "init", "--no-prompt")

    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir.run(*args)

    return do_run
