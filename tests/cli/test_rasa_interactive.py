def test_interactive_help(run):
    output = run("interactive", "--help")

    help_text = """usage: rasa interactive [-h] [-v] [-vv] [--quiet] [-m MODEL]
                        [--data DATA [DATA ...]] [--skip-visualization]
                        [--endpoints ENDPOINTS] [-c CONFIG] [-d DOMAIN]
                        [--out OUT] [--augmentation AUGMENTATION]
                        [--debug-plots] [--dump-stories] [--force]
                        {core} ... [model-as-positional-argument]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_interactive_core_help(run):
    output = run("interactive", "core", "--help")

    help_text = """usage: rasa interactive core [-h] [-v] [-vv] [--quiet] [-m MODEL] [-s STORIES]
                             [--skip-visualization] [--endpoints ENDPOINTS]
                             [-c CONFIG] [-d DOMAIN] [--out OUT]
                             [--augmentation AUGMENTATION] [--debug-plots]
                             [--dump-stories]
                             [model-as-positional-argument]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line
