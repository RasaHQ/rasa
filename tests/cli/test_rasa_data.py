import os
import pytest
from collections import namedtuple
from typing import Callable
from _pytest.pytester import RunResult
from rasa.cli import data


def test_data_split_nlu(run_in_default_project: Callable[..., RunResult]):
    run_in_default_project(
        "data", "split", "nlu", "-u", "data/nlu.md", "--training-fraction", "0.75"
    )

    assert os.path.exists("train_test_split")
    assert os.path.exists(os.path.join("train_test_split", "test_data.md"))
    assert os.path.exists(os.path.join("train_test_split", "training_data.md"))


def test_data_convert_nlu(run_in_default_project: Callable[..., RunResult]):
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


def test_data_split_help(run: Callable[..., RunResult]):
    output = run("data", "split", "nlu", "--help")

    help_text = """usage: rasa data split nlu [-h] [-v] [-vv] [--quiet] [-u NLU]
                           [--training-fraction TRAINING_FRACTION] [--out OUT]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("data", "convert", "nlu", "--help")

    help_text = """usage: rasa data convert nlu [-h] [-v] [-vv] [--quiet] --data DATA --out OUT
                             [-l LANGUAGE] -f {json,md}"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_data_validate_help(run: Callable[..., RunResult]):
    output = run("data", "validate", "--help")

    help_text = """usage: rasa data validate [-h] [-v] [-vv] [--quiet] [--fail-on-warnings]
                          [-d DOMAIN] [--data DATA]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_validate_files_exit_early():
    with pytest.raises(SystemExit) as pytest_e:
        args = {"domain": "data/test_domains/duplicate_intents.yml", "data": None}
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))

    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == 1


def test_clean_init(run_in_default_project: Callable[..., RunResult]):
    # Nothing to be cleaned in the init project
    output = run_in_default_project("data", "clean")
    assert output.outlines == []
    assert output.errlines == []


def test_clean(run: Callable[..., RunResult]):
    os.mkdir("data")
    with open("data/stories.md", "w+") as file:
        file.write(
            "## story\n"
            "* greet\n"
            "  - utter_greet\n"
            "\n"
            "## story\n"
            "* bye\n"
            "  - utter_bye\n"
        )

    with open("domain.yml", "w+") as file:
        file.write(
            "actions:\n"
            "- utter_greet\n"
            "- utter_bye\n"
            "intents:\n"
            "- greet\n"
            "- bye\n"
            "templates:\n"
            "  utter_greet:\n"
            '  - text: "hi"\n'
            "  utter_bye:\n"
            '  - text: "bye"\n'
        )

    output = run("data", "clean")
    # One replacement + headline
    assert len(output.errlines) == 2

    output = run("data", "validate", "stories", "--max-history", "3")
    # No errors:
    assert "No story structure conflicts found" in output.errlines[-1]
