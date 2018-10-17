import pytest

from data.test_config.example_policy import ExamplePolicy
from rasa_core.config import load, handle_precedence_and_defaults
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.ensemble import PolicyEnsemble


def test_handle_precedence_and_defaults():

    config_data = {'policies':[
        {'name':'FallbackPolicy', 'nlu_threshold':0.5},
        {'name':'KerasPolicy'}
    ]}
    fallback_args = {
        'nlu_threshold': 1,
        'core_threshold':1,
        'fallback_action_name':'some_name'
    }
    expected_config_data = {'policies': [
        {'name': 'FallbackPolicy', 'nlu_threshold' :1,
        'core_threshold': 1, 'fallback_action_name': 'some_name'},
        {'name': 'KerasPolicy', 'max_history': 3}
    ]}
    new_config_data = handle_precedence_and_defaults(
                            config_data, fallback_args, max_history)
    assert new_config_data == expected_config_data

def test_load():

    loaded = load("data/test_config/example_config.yaml", None, None)
    assert loaded == [MemoizationPolicy(max_history=5),
                      ExamplePolicy(example_arg=10)]
