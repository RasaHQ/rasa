def test_visualize_help(run):
    output = run("visualize", "--help")

    help_text = """usage: rasa visualize [-h] [-v] [-vv] [--quiet] [-d DOMAIN] [-s STORIES]
                      [-c CONFIG] [--out OUT] [--max-history MAX_HISTORY]
                      [-u NLU]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line
