from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tempfile
import pytest
import json
import os
import io

import rasa_nlu
from conftest import CONFIG_DEFAULTS_PATH
from rasa_nlu.config import RasaNLUConfig


with io.open(CONFIG_DEFAULTS_PATH, "r") as f:
    defaults = json.load(f)
    # Special treatment for these two, as they are absolute directories
    defaults["path"] = os.path.join(os.getcwd(), defaults["path"])
    defaults["response_log"] = os.path.join(os.getcwd(), defaults["response_log"])


def test_default_config(default_config):
    assert default_config.as_dict() == defaults


def test_blank_config():
    file_config = {}
    cmdline_args = {}
    env_vars = {}
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config.as_dict() == defaults


def test_invalid_config_json():
    file_config = """{"pipeline": [mitie]}"""   # invalid json
    cmdline_args = {}
    env_vars = {}
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(file_config)
        f.flush()
        with pytest.raises(rasa_nlu.config.InvalidConfigError):
            RasaNLUConfig(f.name, env_vars, cmdline_args)


def test_file_config_unchanged():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {}
    env_vars = {}
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config['path'] == "/path/to/dir"


def test_cmdline_overrides_init():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {"path": "/alternate/path"}
    env_vars = {}
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config['path'] == "/alternate/path"


def test_envvar_overrides_init():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {}
    env_vars = {"RASA_PATH": "/alternate/path"}
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config['path'] == "/alternate/path"


def test_cmdline_overrides_envvar():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {"path": "/another/path"}
    env_vars = {"RASA_PATH": "/alternate/path"}
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config['path'] == "/another/path"
