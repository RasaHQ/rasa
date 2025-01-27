from typing import Callable

from pytest import RunResult


def test_rasa_test_dialogue_understanding_help(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa test du [-h] [-v] [-vv] [--quiet]
                    [--logging-config-file LOGGING_CONFIG_FILE]
                    [--output-file OUTPUT_FILE] [--no-output]
                    [-m MODEL] [--endpoints ENDPOINTS]
                    [--output-prompt]
                    [--remote-storage REMOTE_STORAGE]
                    [path-to-test-cases]
                    [--remove-default-commands [REMOVE_DEFAULT_COMMANDS ...]]
                    [--additional-commands [ADDITIONAL_COMMANDS ...]]

Runs dialogue understanding testing."""
    lines = help_text.split("\n")

    output = run("test", "du", "--help")

    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help
