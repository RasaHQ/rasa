import glob
from typing import Dict, Text, Optional

import pytest

from rasa.shared.core.constants import (
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
)
from rasa.shared.core.events import ActionExecuted
from tests.core.conftest import ExamplePolicy
import rasa.core.config
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.ensemble import PolicyEnsemble


@pytest.mark.parametrize("filename", glob.glob("data/test_config/example_config.yaml"))
def test_load_config(filename):
    loaded = rasa.core.config.load(filename)
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


@pytest.mark.parametrize(
    "config, expected_config, nr_new_rules, expected_triggered_action",
    [
        # Nothing to be migrated
        ({"policies": []}, {"policies": []}, 0, None),
        # Migrate `FallbackPolicy` with default config
        (
            {"policies": [{"name": "FallbackPolicy"}], "pipeline": []},
            {
                "policies": [
                    {
                        "name": "RulePolicy",
                        "core_fallback_threshold": 0.3,
                        "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": 0.3,
                        "ambiguity_threshold": 0.1,
                    }
                ],
            },
            1,
            ACTION_DEFAULT_FALLBACK_NAME,
        ),
        # Migrate `FallbackPolicy` if it's fully configured
        (
            {
                "policies": [
                    {
                        "name": "FallbackPolicy",
                        "nlu_threshold": 0.123,
                        "ambiguity_threshold": 0.9123,
                        "core_threshold": 0.421,
                        "fallback_action_name": "i_got_this",
                    }
                ],
                "pipeline": [],
            },
            {
                "policies": [
                    {
                        "name": "RulePolicy",
                        "core_fallback_threshold": 0.421,
                        "core_fallback_action_name": "i_got_this",
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": 0.123,
                        "ambiguity_threshold": 0.9123,
                    }
                ],
            },
            1,
            "i_got_this",
        ),
        # Migrate if `FallbackClassifier` is already present.
        # Don't override already configured settings.
        (
            {
                "policies": [{"name": "FallbackPolicy"}],
                "pipeline": [{"name": "FallbackClassifier", "threshold": 0.4}],
            },
            {
                "policies": [
                    {
                        "name": "RulePolicy",
                        "core_fallback_threshold": 0.3,
                        "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": 0.4,
                        "ambiguity_threshold": 0.1,
                    }
                ],
            },
            1,
            ACTION_DEFAULT_FALLBACK_NAME,
        ),
        # Migrate `FallbackPolicy` if `RulePolicy` is already configured
        # Don't override already configured settings
        (
            {
                "policies": [
                    {"name": "FallbackPolicy"},
                    {"name": "RulePolicy", "core_fallback_threshold": 1},
                ],
                "pipeline": [],
            },
            {
                "policies": [
                    {
                        "name": "RulePolicy",
                        "core_fallback_threshold": 1,
                        "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": 0.3,
                        "ambiguity_threshold": 0.1,
                    }
                ],
            },
            1,
            ACTION_DEFAULT_FALLBACK_NAME,
        ),
        # Migrate `TwoStageFallbackPolicy` with default config
        (
            {"policies": [{"name": "TwoStageFallbackPolicy"}], "pipeline": []},
            {
                "policies": [
                    {
                        "name": "RulePolicy",
                        "core_fallback_threshold": 0.3,
                        "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": 0.3,
                        "ambiguity_threshold": 0.1,
                    }
                ],
            },
            1,
            ACTION_TWO_STAGE_FALLBACK_NAME,
        ),
        # Migrate `TwoStageFallbackPolicy` with customized config
        (
            {
                "policies": [
                    {
                        "name": "TwoStageFallbackPolicy",
                        "nlu_threshold": 0.123,
                        "ambiguity_threshold": 0.9123,
                        "core_threshold": 0.421,
                        "fallback_core_action_name": "my_core_fallback",
                    }
                ],
                "pipeline": [],
            },
            {
                "policies": [
                    {
                        "name": "RulePolicy",
                        "core_fallback_threshold": 0.421,
                        "core_fallback_action_name": "my_core_fallback",
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": 0.123,
                        "ambiguity_threshold": 0.9123,
                    }
                ],
            },
            1,
            ACTION_TWO_STAGE_FALLBACK_NAME,
        ),
    ],
)
def test_migrate_fallback_policy(
    config: Dict,
    expected_config: Dict,
    nr_new_rules: int,
    expected_triggered_action: Optional[Text],
):
    updated_config, added_rules = rasa.core.config.migrate_fallback_policies(config)

    assert updated_config == expected_config
    assert len(added_rules) == nr_new_rules

    if nr_new_rules > 0:
        assert any(
            isinstance(event, ActionExecuted)
            and event.action_name == expected_triggered_action
            for event in added_rules[0].events
        )

    # TODO: Test that correct action is triggered in FAQ rule!
