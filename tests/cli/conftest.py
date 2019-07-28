import pytest
import os


@pytest.fixture
def run(testdir):
    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir.run(*args)

    return do_run


@pytest.fixture
def run_in_default_project(testdir):
    os.environ["LOG_LEVEL"] = "ERROR"
    testdir.run("rasa", "init", "--no-prompt")

    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir.run(*args)

    return do_run
