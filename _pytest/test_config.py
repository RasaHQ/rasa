from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tempfile
import pytest
import json
import os
import io

import six
from typing import Text

import rasa_nlu
from rasa_nlu.config import RasaNLUConfig


CONFIG_DEFAULTS_PATH = "config_defaults.json"

with io.open(CONFIG_DEFAULTS_PATH, "r") as f:
    defaults = json.load(f)
    # Special treatment for these two, as they are absolute directories
    defaults["path"] = os.path.join(os.getcwd(), defaults["path"])
    defaults["response_log"] = os.path.join(os.getcwd(), defaults["response_log"])


def test_default_config():
    final_config = RasaNLUConfig(CONFIG_DEFAULTS_PATH)
    assert dict(list(final_config.items())) == defaults


def test_blank_config():
    file_config = {}
    cmdline_args = {}
    env_vars = {}
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert dict(list(final_config.items())) == defaults


def test_invalid_config_json():
    file_config = """{"pipeline": [mitie]}"""   # invalid json
    cmdline_args = {}
    env_vars = {}
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json") as f:
        f.write(file_config)
        f.flush()
        with pytest.raises(rasa_nlu.config.InvalidConfigError):
            RasaNLUConfig(f.name, env_vars, cmdline_args)


def test_automatically_converts_to_unicode():
    env_args = {"RASA_some_test_key_u": str("some test value"), str("RASA_some_test_key_str"): str("some test value")}
    cmd_args = {"some_other_test_key_str": str("some test value"), str("some_other_test_key_u"): str("some test value")}
    final_config = RasaNLUConfig(CONFIG_DEFAULTS_PATH, env_args, cmd_args)

    assert type(final_config["some_test_key_u"]) is Text
    assert type(final_config["some_test_key_str"]) is Text
    assert type(final_config["some_other_test_key_str"]) is Text
    assert type(final_config["some_other_test_key_u"]) is Text


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
