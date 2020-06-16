from pathlib import Path
from typing import List, Text, Set, Any, Dict
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.importers import autoconfig
from rasa.utils import io as io_utils

EMPTY_CONFIG = "rasa/cli/initial_project/config.yml"
CONFIG_PIPELINE_ONLY = "data/test_config/config_pipeline_only.yml"
CONFIG_POLICIES_ONLY = "data/test_config/config_policies_only.yml"
SOME_CONFIG = "data/test_config/arbitrary_example_config.yml"
DEFAULT_CONFIG = "rasa/importers/default_config.yml"


@pytest.mark.parametrize(
    "config_path, autoconfig_keys",
    [
        (EMPTY_CONFIG, ["pipeline", "policies"]),
        (CONFIG_PIPELINE_ONLY, ["policies"]),
        (CONFIG_POLICIES_ONLY, ["pipeline"]),
        (SOME_CONFIG, []),
    ],
)
def test_get_auto_configuration(
    config_path: Text, autoconfig_keys: Set, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(autoconfig, "dump_config", Mock())
    actual = autoconfig.get_auto_configuration(config_path)

    default = io_utils.read_config_file(DEFAULT_CONFIG)

    expected = io_utils.read_config_file(config_path)
    for key in autoconfig_keys:
        expected[key] = default[key]

    assert actual == expected


@pytest.mark.parametrize("keys", (["policies"], ["pipeline"], ["policies", "pipeline"]))
def test_create_config_for_keys(keys: List[Text]):
    default_config = io_utils.read_config_file(DEFAULT_CONFIG)

    config = autoconfig._create_config_for_keys({}, keys)

    for k in keys:
        assert config[k] == default_config[k]  # given keys are configured correctly

    assert len(config) == len(keys)  # no other keys are configured


def test_dump_config_missing_file(tmp_path: Path):

    config_path = tmp_path / "non_existent_config.yml"

    config = io_utils.read_config_file(SOME_CONFIG)

    autoconfig.dump_config(config, str(config_path))

    actual = io_utils.read_file(config_path)
    expected = io_utils.read_file(
        "data/test_config/arbitrary_example_config_after_dumping.yml"
    )

    assert actual == expected
    # TODO: add assertion for the printed warning


def test_dump_config(tmp_path: Path):
    # things that are done while dumping:
    # - auto configured keys are dumped with commented out configuration
    # - existing comments are kept
    # - the other sections are kept
    # - it works with and without empty lines between the keys
    # - info printed if keys were configured automatically
    pass
