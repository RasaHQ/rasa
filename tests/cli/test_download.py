from typing import Callable

from pytest import RunResult


def test_rasa_download_help(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa studio download [-h] [-v] [-vv] [--quiet]
                [--logging-config-file LOGGING_CONFIG_FILE]
                [-d DOMAIN] [--data DATA [DATA ...]] [-c CONFIG]
                [--endpoints ENDPOINTS] [--overwrite]
                assistant_name"""
    lines = help_text.split("\n")

    output = run("studio", "download", "--help")

    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help
