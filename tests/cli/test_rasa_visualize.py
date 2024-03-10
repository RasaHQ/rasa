from typing import Callable
from _pytest.pytester import RunResult

from tests.cli.conftest import RASA_EXE


def test_visualize_help(run: Callable[..., RunResult]):
    output = run("visualize", "--help")

    help_text = f"""usage: {RASA_EXE} visualize [-h] [-v] [-vv] [--quiet]
                      [--logging-config-file LOGGING_CONFIG_FILE] [-d DOMAIN]
                      [-s STORIES] [--out OUT] [--max-history MAX_HISTORY]
                      [-u NLU]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help
