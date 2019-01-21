import glob
import pytest

from tests.conftest import ExamplePolicy
from rasa_core.config import load
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.policies.form_policy import FormPolicy
from rasa_core.policies.ensemble import PolicyEnsemble
from rasa_core.featurizers import BinarySingleStateFeaturizer, \
    MaxHistoryTrackerFeaturizer


@pytest.mark.parametrize("filename", glob.glob(
    "data/test_config/example_config.yaml"))
def test_load_config(filename):
    loaded = load(filename)
    assert len(loaded) == 2
    assert isinstance(loaded[0], MemoizationPolicy)
    assert isinstance(loaded[1], ExamplePolicy)


def test_ensemble_from_dict():
    def check_memoization(p):
        assert p.max_history == 5

    def check_keras(p):
        featurizer = p.featurizer
        state_featurizer = featurizer.state_featurizer
        # Assert policy
        assert p.epochs == 50
        # Assert featurizer
        assert isinstance(featurizer, MaxHistoryTrackerFeaturizer)
        assert featurizer.max_history == 5
        # Assert state_featurizer
        assert isinstance(state_featurizer, BinarySingleStateFeaturizer)

    def check_fallback(p):
        assert p.fallback_action_name == 'action_default_fallback'
        assert p.nlu_threshold == 0.7
        assert p.core_threshold == 0.7

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

    # Check if all policies are present
    assert len(ensemble) == 4
    # MemoizationPolicy is parent of FormPolicy
    assert any([isinstance(p, MemoizationPolicy) and
                not isinstance(p, FormPolicy) for p in ensemble])
    assert any([isinstance(p, KerasPolicy) for p in ensemble])
    assert any([isinstance(p, FallbackPolicy) for p in ensemble])
    assert any([isinstance(p, FormPolicy) for p in ensemble])

    # Verify policy configurations
    for policy in ensemble:
        if isinstance(policy, MemoizationPolicy) \
                and not isinstance(policy, FormPolicy):
            check_memoization(policy)
        elif isinstance(policy, KerasPolicy):
            check_keras(policy)
        elif isinstance(policy, FallbackPolicy):
            check_fallback(policy)
