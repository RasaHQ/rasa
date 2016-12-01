from rasa_nlu import util
from rasa_nlu import config_keys


def test_blank_config():
    init_config = {}
    cmdline_args = dict.fromkeys(config_keys, None)
    env_vars = {}
    result = {}
    final_config = util.update_config(init_config, cmdline_args, env_vars)
    assert final_config == result


def test_init_config_unchanged():
    init_config = {"path": "/path/to/dir"}
    cmdline_args = dict.fromkeys(config_keys, None)
    env_vars = {}
    result = {"path": "/path/to/dir"}
    final_config = util.update_config(init_config, cmdline_args, env_vars)
    assert final_config == result


def test_cmdline_overrides_init():
    init_config = {"path": "/path/to/dir"}
    cmdline_args = {"path": "/alternate/path"}
    env_vars = {}
    result = {"path": "/alternate/path"}
    final_config = util.update_config(init_config, cmdline_args, env_vars)
    assert final_config == result


def test_envvar_overrides_init():
    init_config = {"path": "/path/to/dir"}
    cmdline_args = {}
    env_vars = {"RASA_PATH": "/alternate/path"}
    result = {"path": "/alternate/path"}
    final_config = util.update_config(init_config, cmdline_args, env_vars)
    assert final_config == result


def test_cmdline_overrides_envvar():
    init_config = {"path": "/path/to/dir"}
    cmdline_args = {"path": "/another/path"}
    env_vars = {"RASA_PATH": "/alternate/path"}
    result = {"path": "/another/path"}
    final_config = util.update_config(init_config, cmdline_args, env_vars)
    assert final_config == result
