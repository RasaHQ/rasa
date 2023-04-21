import contextlib
import copy
import re
import argparse
import logging
import os
import pathlib
import sys
import tempfile
from typing import Any, Dict, Text
from rasa.cli import utils
from unittest.mock import Mock
from collections import namedtuple
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.validator import Validator

import pytest
from _pytest.logging import LogCaptureFixture

import rasa.cli.utils
from rasa.shared.constants import (
    ASSISTANT_ID_DEFAULT_VALUE,
    ASSISTANT_ID_KEY,
    CONFIG_MANDATORY_KEYS,
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS_NLU,
    DEFAULT_CONFIG_PATH,
)
import rasa.shared.utils.io
from tests.cli.conftest import RASA_EXE


@contextlib.contextmanager
def make_actions_subdir():
    """Create a subdir called actions to test model argument handling."""
    with tempfile.TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        try:
            (pathlib.Path(tempdir) / "actions").mkdir()
            yield
        finally:
            os.chdir(cwd)


@pytest.mark.parametrize(
    "argv",
    [
        [RASA_EXE, "run"],
        [RASA_EXE, "run", "actions"],
        [RASA_EXE, "run", "core"],
        [RASA_EXE, "interactive", "nlu", "--param", "xy"],
    ],
)
def test_parse_last_positional_argument_as_model_path(argv):
    with make_actions_subdir():
        test_model_dir = tempfile.gettempdir()
        argv.append(test_model_dir)

        sys.argv = argv.copy()
        rasa.cli.utils.parse_last_positional_argument_as_model_path()

        assert sys.argv[-2] == "--model"
        assert sys.argv[-1] == test_model_dir


@pytest.mark.parametrize(
    "argv",
    [
        [RASA_EXE, "run"],
        [RASA_EXE, "run", "actions"],
        [RASA_EXE, "run", "core"],
        [RASA_EXE, "test", "nlu", "--param", "xy", "--model", "test"],
    ],
)
def test_parse_no_positional_model_path_argument(argv):
    with make_actions_subdir():
        sys.argv = argv.copy()

        rasa.cli.utils.parse_last_positional_argument_as_model_path()

        assert sys.argv == argv


def test_validate_invalid_path():
    with pytest.raises(SystemExit):
        rasa.cli.utils.get_validated_path("test test test", "out", "default")


def test_validate_valid_path(tmp_path: pathlib.Path):
    assert rasa.cli.utils.get_validated_path(str(tmp_path), "out", "default") == str(
        tmp_path
    )


def test_validate_if_none_is_valid():
    assert rasa.cli.utils.get_validated_path(None, "out", "default", True) is None


def test_validate_with_none_if_default_is_valid(
    caplog: LogCaptureFixture, tmp_path: pathlib.Path
):
    with caplog.at_level(logging.WARNING, rasa.cli.utils.logger.name):
        assert rasa.cli.utils.get_validated_path(None, "out", str(tmp_path)) == str(
            tmp_path
        )

        caplog_records = [
            record for record in caplog.records if "ddtrace.internal" not in record.name
        ]

        assert caplog_records == []


def test_validate_with_invalid_directory_if_default_is_valid(tmp_path: pathlib.Path):
    invalid_directory = "gcfhvjkb"
    with pytest.warns(UserWarning) as record:
        assert rasa.cli.utils.get_validated_path(
            invalid_directory, "out", str(tmp_path)
        ) == str(tmp_path)
    assert len(record) == 1
    assert "does not seem to exist" in record[0].message.args[0]


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "config_data": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
            },
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
        },
        {
            "config_data": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
                "policies": None,
            },
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
        },
    ],
)
def test_get_validated_config_with_valid_input(parameters: Dict[Text, Any]) -> None:
    config_path = os.path.join(tempfile.mkdtemp(), "config.yml")
    rasa.shared.utils.io.write_yaml(parameters["config_data"], config_path)

    default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
    rasa.shared.utils.io.write_yaml(parameters["default_config"], default_config_path)

    config_path = rasa.cli.utils.get_validated_config(
        config_path, parameters["mandatory_keys"], default_config_path
    )

    config_data = rasa.shared.utils.io.read_yaml_file(config_path)

    for k in parameters["mandatory_keys"]:
        assert k in config_data


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
        },
        {
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
        },
    ],
)
def test_get_validated_config_with_default_config(parameters: Dict[Text, Any]) -> None:
    config_path = None

    default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
    rasa.shared.utils.io.write_yaml(parameters["default_config"], default_config_path)

    config_path = rasa.cli.utils.get_validated_config(
        config_path, parameters["mandatory_keys"], default_config_path
    )

    config_data = rasa.shared.utils.io.read_yaml_file(config_path)

    for k in parameters["mandatory_keys"]:
        assert k in config_data


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "config_data": {
                "assistant_id": "placeholder_default",
            },
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
        },
        {
            "config_data": {
                "assistant_id": "placeholder_default",
                "policies": ["TEDPolicy", "FallbackPolicy"],
                "imports": "other-folder",
            },
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_NLU,
        },
    ],
)
def test_get_validated_config_with_invalid_input(parameters: Dict[Text, Any]) -> None:
    config_path = os.path.join(tempfile.mkdtemp(), "config.yml")
    rasa.shared.utils.io.write_yaml(parameters["config_data"], config_path)

    default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
    rasa.shared.utils.io.write_yaml(parameters["default_config"], default_config_path)

    with pytest.raises(SystemExit):
        rasa.cli.utils.get_validated_config(
            config_path, parameters["mandatory_keys"], default_config_path
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "config_data": None,
            "default_config": {
                "assistant_id": "placeholder_default",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_NLU,
        },
    ],
)
def test_get_validated_config_with_default_and_no_config(
    parameters: Dict[Text, Any]
) -> None:
    config_path = None
    default_config_content = {
        "assistant_id": "placeholder_default",
        "pipeline": "supervised",
        "policies": ["TEDPolicy", "FallbackPolicy"],
    }
    mandatory_keys = CONFIG_MANDATORY_KEYS_NLU

    default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
    rasa.shared.utils.io.write_yaml(default_config_content, default_config_path)

    with pytest.raises(SystemExit):
        rasa.cli.utils.get_validated_config(
            config_path, mandatory_keys, default_config_path
        )


def test_get_validated_config_with_no_content() -> None:
    config_path = None
    default_config_path = os.path.join(tempfile.mkdtemp(), DEFAULT_CONFIG_PATH)
    mandatory_keys = CONFIG_MANDATORY_KEYS

    with pytest.raises(SystemExit):
        rasa.cli.utils.get_validated_config(
            config_path, mandatory_keys, default_config_path
        )


def test_validate_config_path_with_non_existing_file():
    with pytest.raises(SystemExit):
        rasa.cli.utils.validate_config_path("non-existing-file.yml")


@pytest.mark.parametrize(
    "config_file",
    [
        "data/test_config/config_no_assistant_id.yml",
        "data/test_config/config_default_assistant_id_value.yml",
    ],
)
def test_validate_assistant_id_in_config(config_file: Text) -> None:
    copy_config_data = copy.deepcopy(rasa.shared.utils.io.read_yaml_file(config_file))

    warning_message = (
        f"The config file '{str(config_file)}' is missing a "
        f"unique value for the '{ASSISTANT_ID_KEY}' mandatory key."
    )
    with pytest.warns(UserWarning, match=warning_message):
        rasa.cli.utils.validate_assistant_id_in_config(config_file)

    config_data = rasa.shared.utils.io.read_yaml_file(config_file)
    assistant_name = config_data.get(ASSISTANT_ID_KEY)

    assert assistant_name is not None
    assert assistant_name != ASSISTANT_ID_DEFAULT_VALUE

    # reset input files to original state
    rasa.shared.utils.io.write_yaml(copy_config_data, config_file, True)


def test_data_validate_stories_with_max_history_zero(monkeypatch: pytest.MonkeyPatch):
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
            "--config",
            "data/test_moodbot/config.yml",
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
        "config": "data/test_config/config_defaults.yml",
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
        "config": "data/test_config/config_defaults.yml",
    }
    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))


@pytest.mark.parametrize(
    ("file_type", "data_type"), [("stories", "story"), ("rules", "rule")]
)
def test_validate_files_with_active_loop_null(
    file_type: Text, data_type: Text, tmp_path: Path
):
    domain_file = (
        "data/test_domains/minimal_domain_validate_files_with_active_loop_null.yml"
    )
    nlu_file = "data/test_nlu/test_nlu_validate_files_with_active_loop_null.yml"
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
        "domain": domain_file,
        "data": [file_name, nlu_file],
        "max_history": None,
        "config": "data/test_config/config_unique_assistant_id.yml",
        "fail_on_warnings": False,
    }
    with pytest.warns() as warning_recorder:
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))

    assert not [
        warning.message
        for warning in warning_recorder.list
        if not any(
            type(warning.message) == warning_type
            and re.search(warning_message, str(warning.message))
            for warning_type, warning_message in EXPECTED_WARNINGS
        )
    ]


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
        "config": "data/test_config/config_defaults.yml",
    }
    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))


def test_validate_files_exit_early():
    with pytest.raises(SystemExit) as pytest_e:
        args = {
            "domain": "data/test_domains/duplicate_intents.yml",
            "data": None,
            "max_history": None,
            "config": "data/test_config/config_defaults.yml",
        }
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))

    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == 1


def test_validate_files_invalid_domain():
    args = {
        "domain": "data/test_domains/default_with_mapping.yml",
        "data": None,
        "max_history": None,
        "config": "data/test_config/config_defaults.yml",
    }

    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))
        with pytest.warns(UserWarning) as w:
            assert "Please migrate to RulePolicy." in str(w[0].message)


def test_validate_files_invalid_slot_mappings(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    tested_slot = "duration"
    form_name = "booking_form"
    # form required_slots does not include the tested_slot
    domain.write_text(
        f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intents:
            - state_length_of_time
            entities:
            - city
            slots:
              {tested_slot}:
                type: text
                influence_conversation: false
                mappings:
                - type: from_text
                  intent: state_length_of_time
                  conditions:
                  - active_loop: {form_name}
              location:
                type: text
                mappings:
                - type: from_entity
                  entity: city
            forms:
              {form_name}:
                required_slots:
                - location
                """
    )
    args = {
        "domain": str(domain),
        "data": None,
        "max_history": None,
        "config": "data/test_config/config_defaults.yml",
        "fail_on_warnings": False,
    }
    with pytest.raises(SystemExit):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))


def test_validate_files_config_default_assistant_id():
    args = {
        "domain": "data/test_moodbot/domain.yml",
        "data": None,
        "max_history": None,
        "config": "data/test_config/config_defaults.yml",
        "fail_on_warnings": False,
    }
    msg = (
        f"The config file is missing a unique value for the "
        f"'{ASSISTANT_ID_KEY}' mandatory key. Please replace the default "
        f"placeholder value with a unique identifier."
    )
    with pytest.warns(UserWarning, match=msg):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))


def test_validate_files_config_missing_assistant_id():
    args = {
        "domain": "data/test_moodbot/domain.yml",
        "data": None,
        "max_history": None,
        "config": "data/test_config/config_no_assistant_id.yml",
        "fail_on_warnings": False,
    }
    msg = f"The config file is missing the '{ASSISTANT_ID_KEY}' mandatory key."
    with pytest.warns(UserWarning, match=msg):
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))