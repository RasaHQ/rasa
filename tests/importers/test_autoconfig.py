import shutil
from pathlib import Path
from typing import List, Text, Set
from unittest.mock import Mock

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from rasa.constants import CONFIG_AUTOCONFIGURABLE_KEYS
from rasa.importers import autoconfig
from rasa.utils import io as io_utils

CONFIG_FOLDER = "data/test_config"

SOME_CONFIG = CONFIG_FOLDER + "/arbitrary_example_config.yml"
DEFAULT_CONFIG = "rasa/importers/default_config.yml"


@pytest.mark.parametrize(
    "config_path, autoconfig_keys",
    [
        ("rasa/cli/initial_project/config.yml", ["pipeline", "policies"]),
        (CONFIG_FOLDER + "/config_pipeline_only.yml", ["policies"]),
        (CONFIG_FOLDER + "/config_policies_only.yml", ["pipeline"]),
        (SOME_CONFIG, []),
    ],
)
def test_get_configuration(
    config_path: Text, autoconfig_keys: Set, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(autoconfig, "_dump_config", Mock())
    actual = autoconfig.get_configuration(config_path)

    default = io_utils.read_config_file(DEFAULT_CONFIG)

    expected = io_utils.read_config_file(config_path)
    for key in autoconfig_keys:
        expected[key] = default[key]

    assert actual == expected


@pytest.mark.parametrize("keys", (["policies"], ["pipeline"], ["policies", "pipeline"]))
def test_auto_configure(keys: List[Text]):
    default_config = io_utils.read_config_file(DEFAULT_CONFIG)

    config = autoconfig._auto_configure({}, keys)

    for k in keys:
        assert config[k] == default_config[k]  # given keys are configured correctly

    assert len(config) == len(keys)  # no other keys are configured


def test_dump_config_missing_file(tmp_path: Path, capsys: CaptureFixture):

    config_path = tmp_path / "non_existent_config.yml"

    config = io_utils.read_config_file(SOME_CONFIG)

    autoconfig._dump_config(config, str(config_path))

    actual = io_utils.read_file(config_path)
    expected = io_utils.read_file(
        CONFIG_FOLDER + "/arbitrary_example_config_after_dumping.yml"
    )

    assert actual == expected

    captured = capsys.readouterr()
    assert "does not exist or is empty" in captured.out

    for k in CONFIG_AUTOCONFIGURABLE_KEYS:
        assert k in captured.out


# Test a few cases that are known to be potentially tricky (have failed in the past)
@pytest.mark.parametrize(
    "input_file, expected_file, autoconfig_keys",
    [
        (
            "config_with_comments.yml",
            "config_with_comments_after_dumping.yml",
            ["policies"],
        ),  # comments in various positions
        (
            "config_empty.yml",
            "config_empty_after_dumping.yml",
            ["policies", "pipeline"],
        ),  # no empty lines
        (
            "config_with_comments_after_dumping.yml",
            "config_with_comments_after_dumping.yml",
            ["policies"],
        ),  # with previous auto config that needs to be overwritten
    ],
)
def test_dump_config(
    tmp_path: Path,
    input_file: Text,
    expected_file: Text,
    capsys: CaptureFixture,
    autoconfig_keys: List[Text],
):
    config_path = str(tmp_path / "config.yml")
    shutil.copyfile(CONFIG_FOLDER + "/" + input_file, config_path)

    autoconfig.get_configuration(config_path)

    actual = io_utils.read_file(config_path)
    expected = io_utils.read_file(CONFIG_FOLDER + "/" + expected_file)

    assert actual == expected

    captured = capsys.readouterr()
    assert "does not exist or is empty" not in captured.out

    for k in CONFIG_AUTOCONFIGURABLE_KEYS:
        if k in autoconfig_keys:
            assert k in captured.out
        else:
            assert k not in captured.out
