import argparse
import contextlib
import copy
import io
import os
import pathlib
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Text, Callable

import pytest
import structlog
from _pytest.pytester import RunResult
from ruamel.yaml import YAML

import rasa.cli.utils
import rasa.shared.utils.io
from rasa.shared.constants import (
    ASSISTANT_ID_DEFAULT_VALUE,
    ASSISTANT_ID_KEY,
    CONFIG_MANDATORY_KEYS,
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS_NLU,
    DEFAULT_CONFIG_PATH,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import read_yaml_file, write_yaml
from rasa.utils.common import EXPECTED_WARNINGS
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name
from tests.cli.conftest import RASA_EXE
from tests.conftest import AsyncMock
from tests.utilities import filter_logs


@contextlib.contextmanager
def make_actions_subdir():
    """Create a subdir called actions to test model argument handling."""
    with TempDirectoryPath(get_temp_dir_name()) as tempdir:
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
        test_model_dir = get_temp_dir_name()
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


def test_validate_with_multiple_default_options(tmp_path: pathlib.Path):
    chosen_option = rasa.cli.utils.get_validated_path(
        None, "out", ["xyz", str(tmp_path)]
    )
    assert chosen_option == str(tmp_path)


def test_validate_with_none_if_default_is_valid(tmp_path: pathlib.Path):
    expected_event = "cli.get_validated_path.path_does_not_exists"
    expected_log_level = "warning"

    with structlog.testing.capture_logs() as caplog:
        assert rasa.cli.utils.get_validated_path(None, "out", str(tmp_path)) == str(
            tmp_path
        )
        logs = filter_logs(caplog, expected_event, expected_log_level)
        assert len(logs) == 0


def test_validate_with_invalid_directory_if_default_is_valid(tmp_path: pathlib.Path):
    invalid_directory = "gcfhvjkb"

    expected_event = "cli.get_validated_path.path_does_not_exists"
    expected_log_level = "warning"
    expected_log_message = "does not seem to exist"

    with structlog.testing.capture_logs() as caplog:
        assert rasa.cli.utils.get_validated_path(
            invalid_directory, "out", str(tmp_path)
        ) == str(tmp_path)

        logs = filter_logs(
            caplog, expected_event, expected_log_level, [expected_log_message]
        )
        assert len(logs) == 1


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
    write_yaml(parameters["config_data"], config_path)

    default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
    write_yaml(parameters["default_config"], default_config_path)

    config_path = rasa.cli.utils.get_validated_config(
        config_path, parameters["mandatory_keys"], default_config_path
    )

    config_data = read_yaml_file(config_path)

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
    write_yaml(parameters["default_config"], default_config_path)

    config_path = rasa.cli.utils.get_validated_config(
        config_path, parameters["mandatory_keys"], default_config_path
    )

    config_data = read_yaml_file(config_path)

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
    write_yaml(parameters["config_data"], config_path)

    default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
    write_yaml(parameters["default_config"], default_config_path)

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
    parameters: Dict[Text, Any],
) -> None:
    config_path = None
    default_config_content = {
        "assistant_id": "placeholder_default",
        "pipeline": "supervised",
        "policies": ["TEDPolicy", "FallbackPolicy"],
    }
    mandatory_keys = CONFIG_MANDATORY_KEYS_NLU

    default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
    write_yaml(default_config_content, default_config_path)

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
    "config_file, expected_log_event, expected_log_message",
    [
        (
            "data/test_config/config_no_assistant_id.yml",
            "cli.validate_assistant_id_in_config.missing_unique_assistant_id_key",
            f"is missing a unique value for the '{ASSISTANT_ID_KEY}' mandatory key. ",
        ),
        (
            "data/test_config/config_default_assistant_id_value.yml",
            "cli.validate_assistant_id_in_config.missing_unique_assistant_id_key",
            f"is missing a unique value for the '{ASSISTANT_ID_KEY}' mandatory key.",
        ),
    ],
)
def test_validate_assistant_id_in_config(
    config_file: Text,
    expected_log_event: Text,
    expected_log_message: Text,
) -> None:
    copy_config_data = copy.deepcopy(read_yaml_file(config_file))

    expected_log_level = "warning"

    with structlog.testing.capture_logs() as caplog:
        rasa.cli.utils.validate_assistant_id_in_config(config_file)
        logs = filter_logs(
            caplog, expected_log_event, expected_log_level, [expected_log_message]
        )
        assert len(logs) == 1

    # Calling the wrapped function to avoid getting cached result
    config_data = read_yaml_file.__wrapped__(config_file)
    assistant_name = config_data.get(ASSISTANT_ID_KEY)
    assert assistant_name is not None
    assert assistant_name != ASSISTANT_ID_DEFAULT_VALUE

    # reset input files to original state
    write_yaml(copy_config_data, config_file, True)


def test_data_validate_stories_with_max_history_zero():
    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_defaults.yml",
        "data/test_moodbot/domain.yml",
        "data/test_moodbot/data",
    )

    with pytest.raises(argparse.ArgumentTypeError):
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=0,
            importer=importer,
        )


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

    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_defaults.yml",
        "data/test_moodbot/domain.yml",
        [file_name],
    )

    with pytest.raises(SystemExit):
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=None,
            importer=importer,
        )


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

    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_defaults.yml",
        "data/test_restaurantbot/domain.yml",
        [file_name],
    )
    with pytest.raises(SystemExit):
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=None,
            importer=importer,
        )


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

    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_unique_assistant_id.yml",
        domain_file,
        [file_name, nlu_file],
    )
    with pytest.warns() as warning_recorder:
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=None,
            importer=importer,
        )

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

    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_defaults.yml",
        domain_file_name,
        "data/test_moodbot/data",
    )
    with pytest.raises(SystemExit):
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=None,
            importer=importer,
        )


def test_validate_files_exit_early():
    with pytest.raises(SystemExit) as pytest_e:
        importer = TrainingDataImporter.load_from_config(
            "data/test_config/config_defaults.yml",
            "data/test_domains/duplicate_intents.yml",
            "data/test_moodbot/data",
        )
        rasa.cli.utils.validate_files(
            fail_on_warnings=True,
            max_history=None,
            importer=importer,
        )

    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == 1


def test_validate_files_invalid_domain():
    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_defaults.yml",
        "data/test_domains/default_with_mapping.yml",
        None,
    )

    with pytest.raises(SystemExit):
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=None,
            importer=importer,
        )
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
    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_defaults.yml", str(domain), None
    )
    with pytest.raises(SystemExit):
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=None,
            importer=importer,
        )


def test_validate_files_config_default_assistant_id():
    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_defaults.yml", "data/test_moodbot/domain.yml", None
    )
    expected_event = "validator.config_missing_unique_mandatory_key_value"
    expected_log_level = "warning"
    expected_log_message = (
        f"The config file is missing a unique value for the "
        f"'{ASSISTANT_ID_KEY}' mandatory key. Please replace the default "
        f"placeholder value with a unique identifier."
    )

    with structlog.testing.capture_logs() as caplog:
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=None,
            importer=importer,
        )
        logs = filter_logs(
            caplog, expected_event, expected_log_level, [expected_log_message]
        )
        assert len(logs) == 1


def test_validate_files_config_missing_assistant_id():
    importer = TrainingDataImporter.load_from_config(
        "data/test_config/config_no_assistant_id.yml",
        "data/test_moodbot/domain.yml",
        None,
    )
    expected_event = "validator.config_missing_mandatory_key"
    expected_log_level = "warning"
    expected_log_message = (
        f"The config file is missing the '{ASSISTANT_ID_KEY}' mandatory key."
    )

    with structlog.testing.capture_logs() as caplog:
        rasa.cli.utils.validate_files(
            fail_on_warnings=False,
            max_history=None,
            importer=importer,
        )
        logs = filter_logs(
            caplog, expected_event, expected_log_level, [expected_log_message]
        )
        assert len(logs) == 1


def test_validate_assistant_id_in_config_preserves_comment() -> None:
    config_file = "data/test_config/config_no_assistant_id_with_comments.yml"
    reader_type = "rt"
    original_config_data = copy.deepcopy(
        read_yaml_file(config_file, reader_type=reader_type)
    )

    # append assistant_id to the config file
    rasa.cli.utils.validate_assistant_id_in_config(config_file)

    # Calling the wrapped function to avoid getting cached result
    config_data = read_yaml_file.__wrapped__(config_file, reader_type=reader_type)

    assert "assistant_id" in config_data

    # get all content of config file including comments
    yaml = YAML()
    buffer = io.StringIO()
    yaml.dump(config_data, buffer)
    config_file_content = buffer.getvalue()

    comment = "# Random comments line {}"
    for i in range(1, 6):
        assert comment.format(i) in config_file_content

    # reset input files to original state
    write_yaml(original_config_data, config_file, True)


@pytest.mark.parametrize(
    "text_input, button",
    [
        ("hi this is test text\n", "hi this is test text"),
        ("hi this is test text (/button_one)", "/button_one"),
        ("hi this is test text (and something) (/button_one)", "/button_one"),
    ],
)
async def test_payload_from_button_question(text_input: str, button: str) -> None:
    question = AsyncMock()
    question.ask_async.return_value = text_input
    result = await rasa.cli.utils.payload_from_button_question(question)
    assert result == button


def test_rasa_version_raises_no_warnings(
    run_in_simple_project: Callable[..., RunResult],
):
    # Run the CLI command "rasa --version"
    result = run_in_simple_project("--version")

    # Get the standard output and error
    stderr = "\n".join(result.stderr.lines)

    # Check if there are any warnings in the output
    assert "warning" not in stderr.lower()


@pytest.mark.parametrize("results_type", ["passed", "failed"])
def test_get_e2e_results_file_name_path_is_dir(
    tmp_path: Path, results_type: str
) -> None:
    results_path = tmp_path / "results"
    results_path.mkdir(exist_ok=True)

    results_file = rasa.cli.utils.get_e2e_results_file_name(results_path, results_type)
    assert results_file == str(results_path / f"e2e_results_{results_type}.yml")


@pytest.mark.parametrize("results_type", ["passed", "failed"])
def test_get_e2e_results_file_name_path_is_file(
    tmp_path: Path, results_type: str
) -> None:
    results_path = tmp_path / "results" / "e2e_test_results.yml"
    results_file = rasa.cli.utils.get_e2e_results_file_name(results_path, results_type)
    assert results_file == str(
        results_path.parent / f"e2e_test_results_{results_type}.yml"
    )
