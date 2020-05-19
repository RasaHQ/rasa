from typing import Callable
from _pytest.pytester import RunResult


def test_cli_start(run: Callable[..., RunResult]):
    """
    Measures an average startup time and checks that it
    does not deviate more than x seconds from 5.
    """
    import time

    durations = []

    for i in range(5):
        start = time.time()
        run("--help")
        end = time.time()

        durations.append(end - start)

    avg_duration = sum(durations) / len(durations)

    # When run in parallel, it takes a little longer
    assert avg_duration - 5 <= 2


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
