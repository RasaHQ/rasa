import os


def test_test(run_in_default_project):
    run_in_default_project("test")

    assert os.path.exists("results")


def test_test_core(run_in_default_project):
    run_in_default_project("test", "core", "--stories", "data")

    assert os.path.exists("results")


def test_test_help(run):
    help = run("test", "--help")

    help_text = """usage: rasa test [-h] [-v] [-vv] [--quiet] [-m MODEL]
                 [--max-stories MAX_STORIES] [--core CORE] [--output OUTPUT]
                 [--e2e] [--endpoints ENDPOINTS] [--fail-on-prediction-errors]
                 [-s STORIES] [--url URL] [-u NLU] [-c CONFIG] [-f FOLDS]
                 [--report [REPORT]] [--successes [SUCCESSES]]
                 [--errors ERRORS] [--histogram HISTOGRAM] [--confmat CONFMAT]
                 {core,nlu} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line


def test_test_nlu_help(run):
    help = run("test", "nlu", "--help")

    help_text = """usage: rasa test nlu [-h] [-v] [-vv] [--quiet] [-m MODEL] [-u NLU] [-c CONFIG]
                     [-f FOLDS] [--report [REPORT]] [--successes [SUCCESSES]]
                     [--errors ERRORS] [--histogram HISTOGRAM]
                     [--confmat CONFMAT]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line


def test_test_core_help(run):
    help = run("test", "core", "--help")

    help_text = """usage: rasa test core [-h] [-v] [-vv] [--quiet] [--max-stories MAX_STORIES]
                      [--core CORE] [-u NLU] [--output OUTPUT] [--e2e]
                      [--endpoints ENDPOINTS] [--fail-on-prediction-errors]
                      [-s STORIES] [--url URL] [-m MODEL [MODEL ...]]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line
