from typing import Callable

from pytest import RunResult


def test_rasa_license_help(run: Callable[..., RunResult]) -> None:
    help_text = (
        "usage: rasa license [-h] [-v] [-vv] [--quiet]",
        "Display licensing information.",
    )
    output = run("license", "--help")

    printed_help = {line.strip() for line in output.outlines}
    for line in help_text:
        assert line in printed_help


def test_rasa_license(run: Callable[..., RunResult]) -> None:
    output = run("license")
    printed_license = "\n".join(line.strip() for line in output.outlines)
    assert "psycopg2 & psycopg2-binary" in printed_license
    assert "/developer-terms" in printed_license
