from typing import Callable
from _pytest.pytester import RunResult


def test_visualize_help(run: Callable[..., RunResult]):
    output = run("visualize", "--help")

    help_text = """usage: rasa visualize [-h] [-v] [-vv] [--quiet] [-d DOMAIN] [-s STORIES]
                      [-c CONFIG] [--out OUT] [--max-history MAX_HISTORY]
                      [-u NLU]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help
