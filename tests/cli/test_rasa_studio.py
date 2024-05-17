from typing import Callable

from pytest import RunResult


def test_studio_config_help(run: Callable[..., RunResult]):
    output = run("studio", "config", "--help")

    help_text = """usage: rasa studio config [-h] [-v] [-vv] [--quiet]
                 [--logging-config-file LOGGING_CONFIG_FILE]
                 [--advanced]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help
