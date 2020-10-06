from typing import Callable
from _pytest.pytester import RunResult
import pytest
import sys


def test_cli_start(run: Callable[..., RunResult]):
    """
    Checks that a call to ``rasa --help`` does not take longer than 7 seconds
    (10 seconds on Windows).
    """
    import time

    start = time.time()
    run("--help")
    end = time.time()

    duration = end - start

    # it sometimes takes a bit more time to start it on Windows
    assert duration <= 20 if sys.platform == "win32" else 10


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("--help")

    help_text = """usage: rasa [-h] [--version]
            {init,run,shell,train,interactive,telemetry,test,visualize,data,export,x}
            ..."""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


@pytest.mark.xfail(
    sys.platform == "win32", reason="--version doesn't print anything on Windows"
)
def test_version_print_lines(run: Callable[..., RunResult]):
    output = run("--version")
    output_text = "".join(output.outlines)
    assert "Rasa Version" in output_text
    assert "Python Version" in output_text
    assert "Operating System" in output_text
    assert "Python Path" in output_text
