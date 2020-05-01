import glob
import pytest

from tests.core.conftest import ExamplePolicy
from rasa.core.config import load
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.ensemble import PolicyEnsemble
from rasa.core.featurizers import (
    BinarySingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
)


@pytest.mark.parametrize("filename", glob.glob("data/test_config/example_config.yaml"))
def test_load_config(filename):
    loaded = load(filename)
    assert len(loaded) == 2
    assert isinstance(loaded[0], MemoizationPolicy)
    assert isinstance(loaded[1], ExamplePolicy)


def test_ensemble_from_dict():
    def check_memoization(p):
        assert p.max_history == 5
        assert p.priority == 3

    def check_fallback(p):
        assert p.fallback_action_name == "action_default_fallback"
        assert p.nlu_threshold == 0.7
        assert p.core_threshold == 0.7
        assert p.priority == 2

    def check_form(p):
        assert p.priority == 1

    ensemble_dict = {
        "policies": [
            {"max_history": 5, "priority": 3, "name": "MemoizationPolicy"},
            {
                "core_threshold": 0.7,
                "priority": 2,
                "name": "FallbackPolicy",
                "nlu_threshold": 0.7,
                "fallback_action_name": "action_default_fallback",
            },
            {"name": "FormPolicy", "priority": 1},
        ]
    }
    ensemble = PolicyEnsemble.from_dict(ensemble_dict)

    # Check if all policies are present
    assert len(ensemble) == 3
    # MemoizationPolicy is parent of FormPolicy
    assert any(
        [
            isinstance(p, MemoizationPolicy) and not isinstance(p, FormPolicy)
            for p in ensemble
        ]
    )
    assert any([isinstance(p, FallbackPolicy) for p in ensemble])
    assert any([isinstance(p, FormPolicy) for p in ensemble])

    # Verify policy configurations
    for policy in ensemble:
        if isinstance(policy, MemoizationPolicy):
            if isinstance(policy, FormPolicy):
                check_form(policy)
            else:
                check_memoization(policy)
        elif isinstance(policy, FallbackPolicy):
            check_fallback(policy)
