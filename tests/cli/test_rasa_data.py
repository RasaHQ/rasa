import argparse
import os
from pathlib import Path
from unittest.mock import Mock
import pytest
from collections import namedtuple
from typing import Callable, Text

from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult
from rasa.cli import data
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa.validator import Validator
import rasa.shared.utils.io

from tests.cli.conftest import RASA_EXE


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


def test_data_convert_nlu_json(run_in_simple_project: Callable[..., RunResult]):

    result = run_in_simple_project(
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

    assert "NLU data in Rasa JSON format is deprecated" in str(result.stderr)
    assert os.path.exists("out_nlu_data.json")


def test_data_convert_nlu_yml(
    run: Callable[..., RunResult], tmp_path: Path, request: FixtureRequest
):

    target_file = tmp_path / "out.yml"

    # The request rootdir is required as the `testdir` fixture in `run` changes the
    # working directory
    test_data_dir = Path(request.config.rootdir, "data", "examples", "rasa")
    source_file = (test_data_dir / "demo-rasa.json").absolute()
    result = run(
        "data",
        "convert",
        "nlu",
        "--data",
        str(source_file),
        "--out",
        str(target_file),
        "-f",
        "yaml",
    )

    assert result.ret == 0
    assert target_file.exists()

    actual_data = RasaYAMLReader().read(target_file)
    expected = RasaYAMLReader().read(test_data_dir / "demo-rasa.yml")

    assert len(actual_data.training_examples) == len(expected.training_examples)
    assert len(actual_data.entity_synonyms) == len(expected.entity_synonyms)
    assert len(actual_data.regex_features) == len(expected.regex_features)
    assert len(actual_data.lookup_tables) == len(expected.lookup_tables)
    assert actual_data.entities == expected.entities


def test_data_split_help(run: Callable[..., RunResult]):
    output = run("data", "split", "nlu", "--help")

    help_text = f"""usage: {RASA_EXE} data split nlu [-h] [-v] [-vv] [--quiet] [-u NLU]
                           [--training-fraction TRAINING_FRACTION]
                           [--random-seed RANDOM_SEED] [--out OUT]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("data", "convert", "nlu", "--help")

    help_text = (
        f"""usage: {RASA_EXE} data convert nlu [-h] [-v] [-vv]"""
        """ [--quiet] [-f {json,yaml}]"""
    )

    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    assert help_text in printed_help


def test_data_validate_help(run: Callable[..., RunResult]):
    output = run("data", "validate", "--help")

    help_text = f"""usage: {RASA_EXE} data validate [-h] [-v] [-vv] [--quiet]
                          [--max-history MAX_HISTORY] [-c CONFIG]
                          [--fail-on-warnings] [-d DOMAIN]
                          [--data DATA [DATA ...]]
                          {{stories}} ..."""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_migrate_help(run: Callable[..., RunResult]):
    output = run("data", "migrate", "--help")
    printed_help = set(output.outlines)

    help_text = f"""usage: {RASA_EXE} data migrate [-h] [-v] [-vv] [--quiet] [-d DOMAIN] [--out OUT]"""  # noqa: E501
    assert help_text in printed_help


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
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
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


@pytest.mark.parametrize(
    ("file_type", "data_type"), [("stories", "story"), ("rules", "rule")]
)
def test_validate_files_with_active_loop_null(
    file_type: Text, data_type: Text, tmp_path: Path
):
    file_name = tmp_path / f"{file_type}.yml"
    file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        {file_type}:
        - {data_type}: test path
          steps:
            - intent: request_restaurant
            - action: restaurant_form
            - active_loop: restaurant_form
            - active_loop: null
            - action: action_search_restaurants
        """
    )
    args = {
        "domain": "data/test_domains/restaurant_form.yml",
        "data": [file_name],
        "max_history": None,
        "config": None,
        "fail_on_warnings": False,
    }
    with pytest.warns(None):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))


def test_validate_files_form_slots_not_matching(tmp_path: Path):
    domain_file_name = tmp_path / "domain.yml"
    domain_file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        forms:
          name_form:
            required_slots:
            - first_name
            - last_name
        slots:
             first_name:
                type: text
                mappings:
                - type: from_text
             last_nam:
                type: text
                mappings:
                - type: from_text
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


def test_validate_files_invalid_slot_mappings(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    slot_name = "started_booking_form"
    domain.write_text(
        f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intents:
            - activate_booking
            entities:
            - city
            slots:
              {slot_name}:
                type: bool
                influence_conversation: false
                mappings:
                - type: from_trigger_intent
                  intent: activate_booking
                  value: true
              location:
                type: text
                mappings:
                - type: from_entity
                  entity: city
            forms:
              booking_form:
                required_slots:
                - location
                """
    )
    args = {"domain": str(domain), "data": None, "max_history": None, "config": None}
    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))
