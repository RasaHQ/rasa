import tempfile
import os

import pytest
from rasa.constants import (
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS,
    CONFIG_MANDATORY_KEYS_NLU,
)

from rasa.model import unpack_model

from rasa.train import _package_model, _get_valid_config
from tests.core.test_model import _fingerprint


@pytest.mark.parametrize(
    "parameters",
    [
        {"model_name": "test-1234", "prefix": None},
        {"model_name": None, "prefix": "core-"},
        {"model_name": None, "prefix": None},
    ],
)
def test_package_model(trained_rasa_model, parameters):
    output_path = tempfile.mkdtemp()
    train_path = unpack_model(trained_rasa_model)

    model_path = _package_model(
        _fingerprint(),
        output_path,
        train_path,
        parameters["model_name"],
        parameters["prefix"],
    )

    assert os.path.exists(model_path)

    file_name = os.path.basename(model_path)

    if parameters["model_name"]:
        assert parameters["model_name"] in file_name

    if parameters["prefix"]:
        assert parameters["prefix"] in file_name

    assert file_name.endswith(".tar.gz")


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "config_data": {"language": "en", "pipeline": "supervised"},
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
        },
        {"config_data": {}, "mandatory_keys": CONFIG_MANDATORY_KEYS},
        {
            "config_data": {
                "policy": ["KerasPolicy", "FallbackPolicy"],
                "imports": "other-folder",
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_NLU,
        },
    ],
)
def test_get_valid_config(parameters):
    import rasa.utils.io

    config_path = os.path.join(tempfile.mkdtemp(), "config.yml")
    rasa.utils.io.write_yaml_file(parameters["config_data"], config_path)

    config_path = _get_valid_config(config_path, parameters["mandatory_keys"])
    config_data = rasa.utils.io.read_yaml_file(config_path)

    for k in parameters["mandatory_keys"]:
        assert k in config_data

    for k, v in parameters["config_data"].items():
        assert config_data[k] == v
