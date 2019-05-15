import pytest
import os

# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.
pytest_plugins = ["pytester"]


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
