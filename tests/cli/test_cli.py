from typing import Callable
from _pytest.pytester import RunResult


def test_cli_start(run: Callable[..., RunResult]):
    """
    Checks that a call to ``rasa --help`` does not take longer than 7 seconds.
    """
    import time

    start = time.time()
    run("--help")
    end = time.time()

    duration = end - start

    assert duration <= 7


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("--help")

    help_text = """usage: rasa [-h] [--version]
            {init,run,shell,train,interactive,test,visualize,data,export,x}
            ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_version_print_lines(run: Callable[..., RunResult]):
    output = run("--version")
    output_text = "".join(output.outlines)
    assert "Rasa Version" in output_text
    assert "Python Version" in output_text
    assert "Operating System" in output_text
    assert "Python Path" in output_text
