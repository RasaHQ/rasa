import os


def test_data_split_nlu(run_in_default_project):
    run_in_default_project(
        "data", "split", "nlu", "-u", "data/nlu.md", "--training-fraction", "0.75"
    )

    assert os.path.exists("train_test_split")
    assert os.path.exists(os.path.join("train_test_split", "test_data.md"))
    assert os.path.exists(os.path.join("train_test_split", "training_data.md"))


def test_data_convert_nlu(run_in_default_project):
    run_in_default_project(
        "data",
        "convert",
        "nlu",
        "--data",
        "data/nlu.md",
        "--out",
        "out_nlu_data.json",
        "-f",
        "json",
    )

    assert os.path.exists("out_nlu_data.json")


def test_data_split_help(run):
    output = run("data", "split", "nlu", "--help")

    help_text = """usage: rasa data split nlu [-h] [-v] [-vv] [--quiet] [-u NLU]
                           [--training-fraction TRAINING_FRACTION] [--out OUT]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_data_convert_help(run):
    output = run("data", "convert", "nlu", "--help")

    help_text = """usage: rasa data convert nlu [-h] [-v] [-vv] [--quiet] --data DATA --out OUT
                             [-l LANGUAGE] -f {json,md}"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_data_validate_help(run):
    output = run("data", "validate", "--help")

    help_text = """usage: rasa data validate [-h] [-v] [-vv] [--quiet] [--fail-on-warnings]
                          [-d DOMAIN] [--data DATA]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line
