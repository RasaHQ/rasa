import pytest


@pytest.mark.repeat(3)
def test_cli_start(run):
    import time

    start = time.time()
    run("--help")
    end = time.time()

    duration = end - start

    assert duration < 3  # startup of cli should not take longer than 3 seconds


def test_data_convert_help(run):
    output = run("--help")

    help_text = """usage: rasa [-h] [--version]
            {init,run,shell,train,interactive,test,visualize,data,x} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line
