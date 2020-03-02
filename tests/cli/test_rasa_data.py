import argparse
import os
from unittest.mock import Mock
import pytest
from collections import namedtuple
from typing import Callable, Text

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult
from rasa.cli import data
from rasa.importers.importer import TrainingDataImporter
from rasa.validator import Validator


def test_data_split_nlu(
    run_in_default_project_without_models: Callable[..., RunResult]
):
    run_in_default_project_without_models(
        "data", "split", "nlu", "-u", "data/nlu.md", "--training-fraction", "0.75"
    )

    assert os.path.exists("train_test_split")
    assert os.path.exists(os.path.join("train_test_split", "test_data.md"))
    assert os.path.exists(os.path.join("train_test_split", "training_data.md"))


def test_data_convert_nlu(
    run_in_default_project_without_models: Callable[..., RunResult]
):
    run_in_default_project_without_models(
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
                           [--training-fraction TRAINING_FRACTION]
                           [--random-seed RANDOM_SEED] [--out OUT]"""

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

    help_text = """usage: rasa data validate [-h] [-v] [-vv] [--quiet]
                          [--max-history MAX_HISTORY] [--fail-on-warnings]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def _text_is_part_of_output_error(text: Text, output: RunResult) -> bool:
    found_info_string = False
    for line in output.errlines:
        if text in line:
            found_info_string = True
    return found_info_string


def test_data_validate_stories_with_max_history_zero(monkeypatch: MonkeyPatch):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Rasa commands")
    data.add_subparser(subparsers, parents=[])

    args = parser.parse_args(["data", "validate", "stories", "--max-history", 0])

    async def mock_from_importer(importer: TrainingDataImporter) -> Validator:
        return Mock()

    monkeypatch.setattr("rasa.validator.Validator.from_importer", mock_from_importer)

    with pytest.raises(argparse.ArgumentTypeError):
        data.validate_files(args)


def test_validate_files_exit_early():
    with pytest.raises(SystemExit) as pytest_e:
        args = {
            "domain": "data/test_domains/duplicate_intents.yml",
            "data": None,
            "max_history": None,
        }
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))

    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == 1
