from typing import Callable

from pytest import RunResult


def test_rasa_upload_help(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa studio upload [-h] [-v] [-vv] [--quiet]
                [--logging-config-file LOGGING_CONFIG_FILE]
                [--data DATA [DATA ...]] [-d DOMAIN]
                [--flows FLOWS [FLOWS ...]] [-c CONFIG]
                [--entities ENTITIES [ENTITIES ...]]
                [--intents INTENTS [INTENTS ...]] [--calm]
                """
    lines = help_text.split("\n")

    output = run("studio", "upload", "--help")

    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help
