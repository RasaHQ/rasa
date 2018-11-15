from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import pytest

from tests.conftest import ExamplePolicy
from rasa_core.config import load
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.policies.form_policy import FormPolicy
from rasa_core.policies.ensemble import PolicyEnsemble


@pytest.mark.parametrize("filename", glob.glob(
            "data/test_config/example_config.yaml"))
def test_load_config(filename):
    loaded = load(filename)
    assert len(loaded) == 2
    assert isinstance(loaded[0], MemoizationPolicy)
    assert isinstance(loaded[1], ExamplePolicy)


def test_ensemble_from_dict():
    ensemble_dict = {'policies': [
            {'epochs': 50, 'name': 'KerasPolicy', 'featurizer': [
                {'max_history': 5, 'name': 'MaxHistoryTrackerFeaturizer',
                 'state_featurizer': [
                     {'name': 'BinarySingleStateFeaturizer'}]}]},
            {'max_history': 5, 'name': 'MemoizationPolicy'},
            {'core_threshold': 0.7, 'name': 'FallbackPolicy',
             'nlu_threshold': 0.7,
             'fallback_action_name': 'action_default_fallback'},
            {'name': 'FormPolicy'}]}

    ensemble = PolicyEnsemble.from_dict(ensemble_dict)
    assert len(ensemble) == 4
    assert any([isinstance(p, MemoizationPolicy) for p in ensemble])
    assert any([isinstance(p, KerasPolicy) for p in ensemble])
    assert any([isinstance(p, FallbackPolicy) for p in ensemble])
    assert any([isinstance(p, FormPolicy) for p in ensemble])
