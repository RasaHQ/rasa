from typing import List, Text, Dict, Any, Set

import pytest

from rasa.configurations import autoconfig
from rasa.utils import io as io_utils

EMPTY_CONFIG = "rasa/cli/initial_project/config.yml"
CONFIG_PIPELINE_ONLY = "data/test_config/config_default_pipeline.yml"
CONFIG_POLICIES_ONLY = "data/test_config/config_default_policies.yml"
DEFAULT_CONFIG = "rasa/configurations/default_config.yml"

# Possible improvement: instead of the following test, use configs that have different
# stuff than the default it should end up with. that ensures that nothing is overwritten


@pytest.mark.parametrize(
    "config_path, autoconfig_keys",
    [
        (EMPTY_CONFIG, {"pipeline", "policies"}),
        (CONFIG_PIPELINE_ONLY, {"policies"}),
        (CONFIG_POLICIES_ONLY, {"pipeline"}),
        (DEFAULT_CONFIG, set()),
    ],
)
def test_get_autoconfiguration(config_path: Text, autoconfig_keys: Set):
    config = io_utils.read_config_file(config_path)
    actual = autoconfig.get_autoconfiguration(config)

    expected = io_utils.read_config_file(DEFAULT_CONFIG)
    expected["autoconfigured"] = autoconfig_keys

    assert actual == expected


@pytest.mark.parametrize("keys", (["policies"], ["pipeline"], ["policies", "pipeline"]))
def test_create_config_for_keys(keys: List[Text]):
    default_config = io_utils.read_config_file(DEFAULT_CONFIG)

    config = {}
    autoconfig.create_config_for_keys(config, keys)

    assert config["autoconfigured"] == set(keys)
    for k in keys:
        assert config[k] == default_config[k]
