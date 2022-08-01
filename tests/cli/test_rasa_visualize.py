from typing import Callable
from _pytest.pytester import RunResult

from tests.cli.conftest import RASA_EXE


def test_visualize_help(run: Callable[..., RunResult]):
    output = run("visualize", "--help")

    help_text = f"""usage: {RASA_EXE} visualize [-h] [-v] [-vv] [--quiet] [-d DOMAIN] [-s STORIES]
                      [--out OUT] [--max-history MAX_HISTORY] [-u NLU]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help
