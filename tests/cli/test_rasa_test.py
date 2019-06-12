import os


def test_test_core(run_in_default_project):
    run_in_default_project("test", "core", "--stories", "data")

    assert os.path.exists("results")


def test_test(run_in_default_project):
    run_in_default_project("test", "--report", "report")

    assert os.path.exists("report")
    assert os.path.exists("results")
    assert os.path.exists("hist.png")
    assert os.path.exists("confmat.png")


def test_test_nlu(run_in_default_project):
    run_in_default_project("test", "nlu", "--nlu", "data", "--success", "success.json")

    assert os.path.exists("hist.png")
    assert os.path.exists("confmat.png")
    assert os.path.exists("success.json")


def test_test_nlu_cross_validation(run_in_default_project):
    run_in_default_project(
        "test", "nlu", "--cross-validation", "-c", "config.yml", "-f", "2"
    )

    assert os.path.exists("hist.png")
    assert os.path.exists("confmat.png")


def test_test_nlu_comparison(run_in_default_project):
    from shutil import copyfile

    copyfile("config.yml", "nlu-config.yml")

    run_in_default_project(
        "test",
        "nlu",
        "-c",
        "config.yml",
        "nlu-config.yml",
        "--report",
        "nlu-report",
        "--run",
        "2",
    )

    assert os.path.exists("nlu-report")


def test_test_help(run):
    output = run("test", "--help")

    help_text = """usage: rasa test [-h] [-v] [-vv] [--quiet] [-m MODEL] [-s STORIES]
                 [--max-stories MAX_STORIES] [--out OUT] [--e2e]
                 [--endpoints ENDPOINTS] [--fail-on-prediction-errors]
                 [--url URL] [-u NLU] [--report [REPORT]]
                 [--successes [SUCCESSES]] [--errors ERRORS]
                 [--histogram HISTOGRAM] [--confmat CONFMAT]
                 [-c CONFIG [CONFIG ...]] [--cross-validation] [-f FOLDS]
                 [-r RUNS] [-p PERCENTAGES [PERCENTAGES ...]]
                 {core,nlu} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_test_nlu_help(run):
    output = run("test", "nlu", "--help")

    help_text = """usage: rasa test nlu [-h] [-v] [-vv] [--quiet] [-m MODEL] [-u NLU]
                     [--report [REPORT]] [--successes [SUCCESSES]]
                     [--errors ERRORS] [--histogram HISTOGRAM]
                     [--confmat CONFMAT] [-c CONFIG [CONFIG ...]]
                     [--cross-validation] [-f FOLDS] [-r RUNS]
                     [-p PERCENTAGES [PERCENTAGES ...]]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_test_core_help(run):
    output = run("test", "core", "--help")

    help_text = """usage: rasa test core [-h] [-v] [-vv] [--quiet] [-m MODEL [MODEL ...]]
                      [-s STORIES] [--max-stories MAX_STORIES] [--out OUT]
                      [--e2e] [--endpoints ENDPOINTS]
                      [--fail-on-prediction-errors] [--url URL]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line
