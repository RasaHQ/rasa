from rasa_nlu import util
from rasa_nlu import config_keys


def test_update_config():

    test_cases = [
      {
        "init_config": {},
        "cmdline_args": dict.fromkeys(config_keys, None),
        "env_vars": {},
        "result": {}
      },
      {
        "init_config": {"path": "/path/to/dir"},
        "cmdline_args": dict.fromkeys(config_keys, None),
        "env_vars": {},
        "result": {"path": "/path/to/dir"}
      },
      {
        "init_config": {"path": "/path/to/dir"},
        "cmdline_args": {"path": "/alternate/path"},
        "env_vars": {},
        "result": {"path": "/alternate/path"}
      },
      {
        "init_config": {"path": "/path/to/dir"},
        "cmdline_args": {},
        "env_vars": {"RASA_PATH": "/alternate/path"},
        "result": {"path": "/alternate/path"}
      },
      {
        "init_config": {"path": "/path/to/dir"},
        "cmdline_args": {"path": "/another/path"},
        "env_vars": {"RASA_PATH": "/alternate/path"},
        "result": {"path": "/another/path"}
      }
    ]

    for case in test_cases:
        final_config = util.update_config(case["init_config"], case["cmdline_args"], case["env_vars"])
        print(case)
        assert final_config == case["result"]
