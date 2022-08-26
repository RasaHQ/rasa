from pathlib import Path
from typing import Callable

from pytest import Testdir, RunResult
import pytest
import sys

from tests.cli.conftest import RASA_EXE


def test_cli_start_is_fast(testdir: Testdir):
    """
    Checks that a call to ``rasa --help`` does not import any slow imports.

    If this is failing this means, that a simple "rasa --help" commands imports
    `tensorflow` which makes our CLI extremely slow. In case this test is failing
    you've very likely added a global import of "tensorflow" which should be
    avoided. Consider making this import (or the import of its parent module)
    a local import.

    If you are clueless where that import happens, you can run
    ```
    python -X importtime -m rasa.__main__ --help  2> import.log
    tuna import.log
    ```
    to get the import chain.
    (make sure to run with python >= 3.7, and install tune (pip install tuna))
    """

    rasa_path = str(
        (Path(__file__).parent / ".." / ".." / "rasa" / "__main__.py").absolute()
    )
    args = [sys.executable, "-X", "importtime", rasa_path, "--help"]
    result = testdir.run(*args)

    assert result.ret == 0

    # tensorflow is slow -> can't get imported when running basic CLI commands
    result.stderr.no_fnmatch_line("*tensorflow.python.eager")


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("--help")

    help_text = f"""usage: {RASA_EXE} [-h] [--version]
            {{init,run,shell,train,interactive,telemetry,test,visualize,data,export,x,evaluate}}
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
