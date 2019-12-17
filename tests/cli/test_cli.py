import pytest
from typing import Callable
from _pytest.pytester import RunResult


@pytest.mark.repeat(3)
def test_cli_start(run: Callable[..., RunResult]):
    """
    Startup of cli should not take longer than n seconds
    """
    import time

    start = time.time()
    run("--help")
    end = time.time()

    duration = end - start

    # When run in parallel, it takes a little longer
    assert duration <= 5


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("--help")

    help_text = """usage: rasa [-h] [--version]
            {init,run,shell,train,interactive,test,visualize,data,x} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line
