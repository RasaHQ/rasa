import argparse
import filecmp
import os
from pathlib import Path
from unittest.mock import Mock
import pytest
from collections import namedtuple
from typing import Callable, Text, Dict, List

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult, Testdir
from rasa.cli import data
from rasa.core.constants import (
    DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_CORE_FALLBACK_THRESHOLD,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION, DEFAULT_DATA_PATH
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.validator import Validator
import rasa.shared.utils.io


def test_data_split_nlu(run_in_simple_project: Callable[..., RunResult]):
    responses_yml = (
        "responses:\n"
        "  chitchat/ask_name:\n"
        "  - text: my name is Sara, Rasa's documentation bot!\n"
        "  chitchat/ask_weather:\n"
        "  - text: the weather is great!\n"
    )

    with open("data/responses.yml", "w") as f:
        f.write(responses_yml)

    run_in_simple_project(
        "data",
        "split",
        "nlu",
        "-u",
        "data/nlu.yml",
        "--training-fraction",
        "0.75",
        "--random-seed",
        "12345",
    )

    folder = Path("train_test_split")
    assert folder.exists()

    nlu_files = [folder / "test_data.yml", folder / "training_data.yml"]
    nlg_files = [folder / "nlg_test_data.yml", folder / "nlg_training_data.yml"]
    for yml_file in nlu_files:
        assert yml_file.exists(), f"{yml_file} file does not exist"
        nlu_data = rasa.shared.utils.io.read_yaml_file(yml_file)
        assert "version" in nlu_data
        assert nlu_data.get("nlu")

    for yml_file in nlg_files:
        assert yml_file.exists(), f"{yml_file} file does not exist"


def test_data_convert_nlu(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "--data",
        "data/nlu.yml",
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
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("data", "convert", "nlu", "--help")

    help_text = """usage: rasa data convert nlu [-h] [-v] [-vv] [--quiet] [-f {json,yaml}]
                             [--data DATA [DATA ...]] [--out OUT]
                             [-l LANGUAGE]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_validate_help(run: Callable[..., RunResult]):
    output = run("data", "validate", "--help")

    help_text = """usage: rasa data validate [-h] [-v] [-vv] [--quiet]
                          [--max-history MAX_HISTORY] [-c CONFIG]
                          [--fail-on-warnings] [-d DOMAIN]
                          [--data DATA [DATA ...]]
                          {stories} ..."""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_validate_stories_with_max_history_zero(monkeypatch: MonkeyPatch):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Rasa commands")
    data.add_subparser(subparsers, parents=[])

    args = parser.parse_args(
        [
            "data",
            "validate",
            "stories",
            "--data",
            "data/test_moodbot/data",
            "--max-history",
            0,
        ]
    )

    def mock_from_importer(importer: TrainingDataImporter) -> Validator:
        return Mock()

    monkeypatch.setattr("rasa.validator.Validator.from_importer", mock_from_importer)

    with pytest.raises(argparse.ArgumentTypeError):
        data.validate_files(args)


@pytest.mark.parametrize(
    ("file_type", "data_type"), [("stories", "story"), ("rules", "rule")]
)
def test_validate_files_action_not_found_invalid_domain(
    file_type: Text, data_type: Text, tmp_path: Path
):
    file_name = tmp_path / f"{file_type}.yml"
    file_name.write_text(
        f"""
        version: "2.0"
        {file_type}:
        - {data_type}: test path
          steps:
          - intent: goodbye
          - action: action_test
        """
    )
    args = {
        "domain": "data/test_moodbot/domain.yml",
        "data": [file_name],
        "max_history": None,
        "config": None,
    }
    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))


@pytest.mark.parametrize(
    ("file_type", "data_type"), [("stories", "story"), ("rules", "rule")]
)
def test_validate_files_form_not_found_invalid_domain(
    file_type: Text, data_type: Text, tmp_path: Path
):
    file_name = tmp_path / f"{file_type}.yml"
    file_name.write_text(
        f"""
        version: "2.0"
        {file_type}:
        - {data_type}: test path
          steps:
            - intent: request_restaurant
            - action: restaurant_form
            - active_loop: restaurant_form
        """
    )
    args = {
        "domain": "data/test_restaurantbot/domain.yml",
        "data": [file_name],
        "max_history": None,
        "config": None,
    }
    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))


def test_validate_files_form_slots_not_matching(tmp_path: Path):
    domain_file_name = tmp_path / "domain.yml"
    domain_file_name.write_text(
        """
        version: "2.0"
        forms:
          name_form:
            required_slots:
              first_name:
              - type: from_text
              last_name:
              - type: from_text
        slots:
             first_name:
                type: text
             last_nam:
                type: text
        """
    )
    args = {
        "domain": domain_file_name,
        "data": None,
        "max_history": None,
        "config": None,
    }
    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))


def test_validate_files_exit_early():
    with pytest.raises(SystemExit) as pytest_e:
        args = {
            "domain": "data/test_domains/duplicate_intents.yml",
            "data": None,
            "max_history": None,
            "config": None,
        }
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))

    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == 1


def test_validate_files_invalid_domain():
    args = {
        "domain": "data/test_domains/default_with_mapping.yml",
        "data": None,
        "max_history": None,
        "config": None,
    }

    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))
        with pytest.warns(UserWarning) as w:
            assert "Please migrate to RulePolicy." in str(w[0].message)


def test_rasa_data_convert_responses(
    run_in_simple_project: Callable[..., RunResult], run: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)
    expected_data_folder = "expected_data"
    os.mkdir(expected_data_folder)

    domain = (
        "version: '2.0'\n"
        "session_config:\n"
        "  session_expiration_time: 60\n"
        "  carry_over_slots_to_new_session: true\n"
        "actions:\n"
        "- respond_chitchat\n"
        "- utter_greet\n"
        "- utter_cheer_up"
    )

    expected_domain = (
        "version: '2.0'\n"
        "session_config:\n"
        "  session_expiration_time: 60\n"
        "  carry_over_slots_to_new_session: true\n"
        "actions:\n"
        "- utter_chitchat\n"
        "- utter_greet\n"
        "- utter_cheer_up\n"
    )

    with open(f"{expected_data_folder}/domain.yml", "w") as f:
        f.write(expected_domain)

    with open("domain.yml", "w") as f:
        f.write(domain)

    stories = (
        "stories:\n"
        "- story: sad path\n"
        "  steps:\n"
        "  - intent: greet\n"
        "  - action: utter_greet\n"
        "  - intent: mood_unhappy\n"
        "- story: chitchat\n"
        "  steps:\n"
        "  - intent: chitchat\n"
        "  - action: respond_chitchat\n"
    )

    expected_stories = (
        'version: "2.0"\n'
        "stories:\n"
        "- story: sad path\n"
        "  steps:\n"
        "  - intent: greet\n"
        "  - action: utter_greet\n"
        "  - intent: mood_unhappy\n"
        "- story: chitchat\n"
        "  steps:\n"
        "  - intent: chitchat\n"
        "  - action: utter_chitchat\n"
    )

    with open(f"{expected_data_folder}/stories.yml", "w") as f:
        f.write(expected_stories)

    with open("data/stories.yml", "w") as f:
        f.write(stories)

    run_in_simple_project(
        "data",
        "convert",
        "responses",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "domain_converted.yml",
        Path(expected_data_folder) / "domain.yml",
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "stories_converted.yml",
        Path(expected_data_folder) / "stories.yml",
    )
