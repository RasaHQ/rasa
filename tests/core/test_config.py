import glob
from typing import Dict, Text, Optional, Any

import pytest

import rasa.core.config
from rasa.core.constants import (
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_CORE_FALLBACK_THRESHOLD,
    DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
)
from rasa.core.policies.ensemble import PolicyEnsemble
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.shared.core.constants import (
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
import rasa.shared.utils.io
from tests.core.conftest import ExamplePolicy


@pytest.mark.parametrize("filename", glob.glob("data/test_config/example_config.yaml"))
def test_load_config(filename):
    loaded = rasa.core.config.load(filename)
    assert len(loaded) == 2
    assert isinstance(loaded[0], MemoizationPolicy)
    assert isinstance(loaded[1], ExamplePolicy)


@pytest.mark.parametrize(
    "config, expected_config, expected_triggered_action",
    [
        # Nothing to be migrated
        ({"policies": []}, {"policies": []}, None),
        # Migrate `FallbackPolicy` with default config
        (
            {"policies": [{"name": "FallbackPolicy"}], "pipeline": []},
            {
                "policies": [
                    {
                        "name": "RulePolicy",
                        "core_fallback_threshold": DEFAULT_CORE_FALLBACK_THRESHOLD,
                        "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": DEFAULT_NLU_FALLBACK_THRESHOLD,
                        "ambiguity_threshold": DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
                    }
                ],
            },
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
                        "core_fallback_threshold": DEFAULT_CORE_FALLBACK_THRESHOLD,
                        "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": 0.4,
                        "ambiguity_threshold": DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
                    }
                ],
            },
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
                        "threshold": DEFAULT_NLU_FALLBACK_THRESHOLD,
                        "ambiguity_threshold": DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
                    }
                ],
            },
            ACTION_DEFAULT_FALLBACK_NAME,
        ),
        # Migrate `TwoStageFallbackPolicy` with default config
        (
            {"policies": [{"name": "TwoStageFallbackPolicy"}], "pipeline": []},
            {
                "policies": [
                    {
                        "name": "RulePolicy",
                        "core_fallback_threshold": DEFAULT_CORE_FALLBACK_THRESHOLD,
                        "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
                    }
                ],
                "pipeline": [
                    {
                        "name": "FallbackClassifier",
                        "threshold": DEFAULT_NLU_FALLBACK_THRESHOLD,
                        "ambiguity_threshold": DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
                    }
                ],
            },
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
            ACTION_TWO_STAGE_FALLBACK_NAME,
        ),
    ],
)
def test_migrate_fallback_policy(
    config: Dict, expected_config: Dict, expected_triggered_action: Optional[Text]
):
    updated_config, added_rule = rasa.core.config.migrate_fallback_policies(config)

    assert updated_config == expected_config
    if not expected_triggered_action:
        return

    assert added_rule

    assert any(
        isinstance(event, ActionExecuted)
        and event.action_name == expected_triggered_action
        for event in added_rule.events
    )


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


TEST_MIGRATED_MAPPING_POLICIES = [
    # no changes, no MappingPolicy
    (
        {"policies": [{"name": "MemoizationPolicy"}]},
        {"intents": ["greet", "leave"]},
        {
            "config": {"policies": [{"name": "MemoizationPolicy"}]},
            "domain_intents": ["greet", "leave"],
            "rules": [],
            "rules_count": 0,
        },
    ),
    # MappingPolicy but no rules
    (
        {"policies": [{"name": "MemoizationPolicy"}, {"name": "MappingPolicy"}]},
        {"intents": ["greet", "leave"]},
        {
            "config": {"policies": [{"name": "MemoizationPolicy"}]},
            "domain_intents": ["greet", "leave"],
            "rules": [],
            "rules_count": 0,
        },
    ),
    # MappingPolicy + rules
    (
        {"policies": [{"name": "MemoizationPolicy"}, {"name": "MappingPolicy"}]},
        {
            "intents": [{"greet": {"triggers": "action_greet"}}, "leave"],
            "actions": ["action_greet"],
        },
        {
            "config": {
                "policies": [{"name": "MemoizationPolicy"}, {"name": "RulePolicy"}]
            },
            "domain_intents": ["greet", "leave"],
            "rules": [
                {
                    "rule": "Rule to map `greet` intent to `action_greet` (automatic conversion)",
                    "steps": [{"intent": "greet"}, {"action": "action_greet"}],
                }
            ],
            "rules_count": 1,
        },
    ),
]


@pytest.mark.parametrize(
    "config,domain_dict,expected_results", TEST_MIGRATED_MAPPING_POLICIES
)
def test_migrate_mapping_policy_to_rules(
    config: Dict[Text, Any],
    domain_dict: Dict[Text, Any],
    expected_results: Dict[Text, Any],
):
    domain = Domain.from_dict(domain_dict)

    config, domain, rules = rasa.core.config.migrate_mapping_policy_to_rules(
        config, domain
    )

    assert config == expected_results["config"]
    assert domain.cleaned_domain()["intents"] == expected_results["domain_intents"]

    assert len(rules) == expected_results["rules_count"]
    rule_writer = YAMLStoryWriter()
    assert (
        rasa.shared.utils.io.read_yaml(rule_writer.dumps(rules)).get("rules", [])
        == expected_results["rules"]
    )
