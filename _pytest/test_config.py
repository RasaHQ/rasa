import tempfile

from rasa_nlu.config import RasaNLUConfig
import json
import os
import logging

defaults = {
  "backend": "mitie",
  "config": "config.json",
  "data": None,
  "emulate": None,
  "language": "en",
  "log_file": None,
  "log_level": logging.INFO,
  "mitie_file": os.path.join("data", "total_word_feature_extractor.dat"),
  "num_threads": 1,
  "fine_tune_spacy_ner": False,
  "path": os.path.join(os.getcwd(), "models"),
  "port": 5000,
  "server_model_dirs": None,
  "token": None,
  "response_log": os.path.join(os.getcwd(), "logs")
}


def test_default_config():
    final_config = RasaNLUConfig()
    assert dict(final_config.items()) == defaults


def test_blank_config():
    file_config = {}
    cmdline_args = {}
    env_vars = {}
    with tempfile.NamedTemporaryFile(suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert dict(final_config.items()) == defaults


def test_file_config_unchanged():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {}
    env_vars = {}
    with tempfile.NamedTemporaryFile(suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config['path'] == "/path/to/dir"


def test_cmdline_overrides_init():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {"path": "/alternate/path"}
    env_vars = {}
    with tempfile.NamedTemporaryFile(suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config['path'] == "/alternate/path"


def test_envvar_overrides_init():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {}
    env_vars = {"RASA_PATH": "/alternate/path"}
    with tempfile.NamedTemporaryFile(suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config['path'] == "/alternate/path"


def test_cmdline_overrides_envvar():
    file_config = {"path": "/path/to/dir"}
    cmdline_args = {"path": "/another/path"}
    env_vars = {"RASA_PATH": "/alternate/path"}
    with tempfile.NamedTemporaryFile(suffix="_tmp_config_file.json") as f:
        f.write(json.dumps(file_config))
        f.flush()
        final_config = RasaNLUConfig(f.name, env_vars, cmdline_args)
        assert final_config['path'] == "/another/path"
