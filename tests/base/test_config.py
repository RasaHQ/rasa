from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import os
import tempfile

import pytest
from typing import Text

import rasa_nlu
from rasa_nlu.config import RasaNLUConfig, InvalidConfigError
from rasa_nlu.registry import registered_pipeline_templates
from tests.conftest import CONFIG_DEFAULTS_PATH
from tests.utilities import write_file_config

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
    f = write_file_config(file_config)
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
    f = write_file_config(file_config)
    final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
    assert final_config['path'] == "/path/to/dir"


def test_cmdline_overrides_init():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {"path": "/alternate/path"}
    env_vars = {}
    f = write_file_config(file_config)
    final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
    assert final_config['path'] == "/alternate/path"


def test_envvar_overrides_init():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {}
    env_vars = {"RASA_PATH": "/alternate/path"}
    f = write_file_config(file_config)
    final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
    assert final_config['path'] == "/alternate/path"


def test_cmdline_overrides_envvar():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {"path": "/another/path"}
    env_vars = {"RASA_PATH": "/alternate/path"}
    f = write_file_config(file_config)
    final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
    assert final_config['path'] == "/another/path"


def test_pipeline_splits_list():
    file_config = {}
    cmdline_args = {"pipeline": "nlp_spacy,ner_spacy"}
    env_vars = {}
    f = write_file_config(file_config)
    final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
    assert final_config['pipeline'] == ["nlp_spacy", "ner_spacy"]


def test_invalid_pipeline_template():
    file_config = {}
    cmdline_args = {"pipeline": "my_made_up_name"}
    env_vars = {}
    f = write_file_config(file_config)
    with pytest.raises(InvalidConfigError) as execinfo:
        RasaNLUConfig(f.name, env_vars, cmdline_args)
    assert "unknown pipeline template" in str(execinfo.value)


def test_pipeline_looksup_registry():
    pipeline_template = list(registered_pipeline_templates)[0]
    file_config = {}
    cmdline_args = {"pipeline": pipeline_template}
    env_vars = {}
    f = write_file_config(file_config)
    final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
    assert final_config['pipeline'] == registered_pipeline_templates[pipeline_template]


def test_default_config_file():
    final_config = RasaNLUConfig()
    assert len(final_config) > 1
