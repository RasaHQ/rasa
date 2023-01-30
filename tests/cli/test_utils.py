import contextlib
import logging
import os
import pathlib
import sys
import tempfile
import pytest
from _pytest.logging import LogCaptureFixture

import rasa.cli.utils
from rasa.shared.constants import (
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
            "error": False,
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
            "error": False,
        },
        {
            "config_data": {},
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
            "error": True,
        },
        {
            "config_data": {
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
            "error": True,
        },
        {
            "config_data": None,
            "default_config": {
                "assistant_id": "placeholder_default",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_NLU,
            "error": True,
        },
        {
            "config_data": None,
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
            "error": False,
        },
        {
            "config_data": None,
            "default_config": {
                "assistant_id": "placeholder_default",
                "language": "en",
                "pipeline": "supervised",
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
            "error": False,
        },
        {
            "config_data": None,
            "default_config": None,
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
            "error": True,
        },
    ],
)
def test_get_valid_config(parameters):
    config_path = None

    if parameters["config_data"] is not None:
        config_path = os.path.join(tempfile.mkdtemp(), "config.yml")
        rasa.shared.utils.io.write_yaml(parameters["config_data"], config_path)

    default_config_path = os.path.join(tempfile.mkdtemp(), DEFAULT_CONFIG_PATH)

    if parameters["default_config"] is not None:
        default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
        rasa.shared.utils.io.write_yaml(
            parameters["default_config"], default_config_path
        )

    if parameters["error"]:
        with pytest.raises(SystemExit):
            config_path = rasa.cli.utils.validate_config_path(
                config_path, default_config_path
            )
            rasa.cli.utils.get_valid_config(config_path, parameters["mandatory_keys"])

    else:
        config_path = rasa.cli.utils.validate_config_path(
            config_path, default_config_path
        )

        config_path = rasa.cli.utils.get_valid_config(
            config_path,
            parameters["mandatory_keys"],
        )

        config_data = rasa.shared.utils.io.read_yaml_file(config_path)

        for k in parameters["mandatory_keys"]:
            assert k in config_data


def test_validate_config_path_with_non_existing_file():
    with pytest.raises(SystemExit):
        rasa.cli.utils.validate_config_path("non-existing-file.yml")
