from typing import Callable
import pytest
import shutil

from _pytest.pytester import Testdir, RunResult


@pytest.fixture
def run(testdir: Testdir) -> Callable[..., RunResult]:
    def do_run(*args):
        args = [shutil.which("rasa")] + list(args)
        return testdir.run(*args)

    return do_run
