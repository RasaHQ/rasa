from pathlib import Path
from typing import List, Any, Text, Optional

import pytest
import copy

from rasa.core.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter

from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.events import UserUttered, ActiveLoop, Event, SlotSet
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.policies.ensemble import (
    PolicyEnsemble,
    InvalidPolicyConfig,
    SimplePolicyEnsemble,
)
from rasa.core.policies.rule_policy import RulePolicy
import rasa.core.actions.action

from tests.core import utilities
from rasa.core.constants import FORM_POLICY_PRIORITY
from rasa.shared.core.events import ActionExecuted
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
)


class WorkingPolicy(Policy):
    @classmethod
    def load(cls, _) -> Policy:
        return WorkingPolicy()

    def persist(self, _) -> None:
        pass

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        pass

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, WorkingPolicy)


def test_policy_loading_simple(tmp_path: Path):
    original_policy_ensemble = PolicyEnsemble([WorkingPolicy()])
    original_policy_ensemble.train([], None, RegexInterpreter())
    original_policy_ensemble.persist(str(tmp_path))

    loaded_policy_ensemble = PolicyEnsemble.load(str(tmp_path))
    assert original_policy_ensemble.policies == loaded_policy_ensemble.policies


class ConstantPolicy(Policy):
    def __init__(
        self,
        priority: Optional[int] = None,
        predict_index: Optional[int] = None,
        confidence: float = 1,
        is_end_to_end_prediction: bool = False,
        events: Optional[List[Event]] = None,
        optional_events: Optional[List[Event]] = None,
    ) -> None:
        super().__init__(priority=priority)
        self.predict_index = predict_index
        self.confidence = confidence
        self.is_end_to_end_prediction = is_end_to_end_prediction
        self.events = events or []
        self.optional_events = optional_events or []

    @classmethod
    def load(cls, _) -> Policy:
        pass

    def persist(self, _) -> None:
        pass

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        result = [0.0] * domain.num_actions
        result[self.predict_index] = self.confidence

        return PolicyPrediction(
            result,
            self.__class__.__name__,
            policy_priority=self.priority,
            is_end_to_end_prediction=self.is_end_to_end_prediction,
            events=self.events,
            optional_events=self.optional_events,
        )


def test_policy_priority():
    domain = Domain.load("data/test_domains/default.yml")
    tracker = DialogueStateTracker.from_events("test", [UserUttered("hi")], [])

    priority_1 = ConstantPolicy(priority=1, predict_index=0)
    priority_2 = ConstantPolicy(priority=2, predict_index=1)

    policy_ensemble_0 = SimplePolicyEnsemble([priority_1, priority_2])
    policy_ensemble_1 = SimplePolicyEnsemble([priority_2, priority_1])

    priority_2_result = priority_2.predict_action_probabilities(
        tracker, domain, RegexInterpreter()
    )

    i = 1  # index of priority_2 in ensemble_0
    prediction = policy_ensemble_0.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    assert prediction.policy_name == "policy_{}_{}".format(i, type(priority_2).__name__)
    assert prediction.probabilities == priority_2_result.probabilities

    i = 0  # index of priority_2 in ensemble_1
    prediction = policy_ensemble_1.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    assert prediction.policy_name == "policy_{}_{}".format(i, type(priority_2).__name__)
    assert prediction.probabilities == priority_2_result.probabilities


def test_fallback_mapping_restart():
    domain = Domain.load("data/test_domains/default.yml")
    events = [
        ActionExecuted(ACTION_DEFAULT_FALLBACK_NAME, timestamp=1),
        utilities.user_uttered(USER_INTENT_RESTART, 1, timestamp=2),
    ]
    tracker = DialogueStateTracker.from_events("test", events, [])

    two_stage_fallback_policy = TwoStageFallbackPolicy(
        priority=2, deny_suggestion_intent_name="deny"
    )
    mapping_policy = MappingPolicy(priority=1)

    mapping_fallback_ensemble = SimplePolicyEnsemble(
        [two_stage_fallback_policy, mapping_policy]
    )

    prediction = mapping_fallback_ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    index_of_mapping_policy = 1
    next_action = rasa.core.actions.action.action_for_index(
        prediction.max_confidence_index, domain, None
    )

    assert (
        prediction.policy_name
        == f"policy_{index_of_mapping_policy}_{MappingPolicy.__name__}"
    )
    assert next_action.name() == ACTION_RESTART_NAME


@pytest.mark.parametrize(
    "events",
    [
        [
            ActiveLoop("test-form"),
            ActionExecuted(ACTION_LISTEN_NAME),
            utilities.user_uttered(USER_INTENT_RESTART, 1),
        ],
        [
            ActionExecuted(ACTION_LISTEN_NAME),
            utilities.user_uttered(USER_INTENT_RESTART, 1),
        ],
    ],
)
def test_mapping_wins_over_form(events: List[Event]):
    domain = """
    forms:
    - test-form
    """
    domain = Domain.from_yaml(domain)
    tracker = DialogueStateTracker.from_events("test", events, [])

    ensemble = SimplePolicyEnsemble(
        [
            MappingPolicy(),
            ConstantPolicy(priority=1, predict_index=0),
            FormPolicy(),
            FallbackPolicy(),
        ]
    )
    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )

    next_action = rasa.core.actions.action.action_for_index(
        prediction.max_confidence_index, domain, None
    )

    index_of_mapping_policy = 0
    assert (
        prediction.policy_name
        == f"policy_{index_of_mapping_policy}_{MappingPolicy.__name__}"
    )
    assert next_action.name() == ACTION_RESTART_NAME


@pytest.mark.parametrize(
    "ensemble",
    [
        SimplePolicyEnsemble(
            [
                FormPolicy(),
                ConstantPolicy(FORM_POLICY_PRIORITY - 1, 0),
                FallbackPolicy(),
            ]
        ),
        SimplePolicyEnsemble([FormPolicy(), MappingPolicy()]),
    ],
)
def test_form_wins_over_everything_else(ensemble: SimplePolicyEnsemble):
    form_name = "test-form"
    domain = f"""
    forms:
    - {form_name}
    """
    domain = Domain.from_yaml(domain)

    events = [
        ActiveLoop("test-form"),
        ActionExecuted(ACTION_LISTEN_NAME),
        utilities.user_uttered("test", 1),
    ]
    tracker = DialogueStateTracker.from_events("test", events, [])
    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )

    next_action = rasa.core.actions.action.action_for_index(
        prediction.max_confidence_index, domain, None
    )

    index_of_form_policy = 0
    assert (
        prediction.policy_name == f"policy_{index_of_form_policy}_{FormPolicy.__name__}"
    )
    assert next_action.name() == form_name


def test_fallback_wins_over_mapping():
    domain = Domain.load("data/test_domains/default.yml")
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        # Low confidence should trigger fallback
        utilities.user_uttered(USER_INTENT_RESTART, 0.0001),
    ]
    tracker = DialogueStateTracker.from_events("test", events, [])

    ensemble = SimplePolicyEnsemble([FallbackPolicy(), MappingPolicy()])

    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    index_of_fallback_policy = 0
    next_action = rasa.core.actions.action.action_for_index(
        prediction.max_confidence_index, domain, None
    )

    assert (
        prediction.policy_name
        == f"policy_{index_of_fallback_policy}_{FallbackPolicy.__name__}"
    )
    assert next_action.name() == ACTION_DEFAULT_FALLBACK_NAME


class LoadReturnsNonePolicy(Policy):
    @classmethod
    def load(cls, _) -> None:
        return None

    def persist(self, _) -> None:
        pass

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        pass


def test_policy_loading_load_returns_none(tmp_path: Path):
    original_policy_ensemble = PolicyEnsemble([LoadReturnsNonePolicy()])
    original_policy_ensemble.train([], None, RegexInterpreter())
    original_policy_ensemble.persist(str(tmp_path))

    with pytest.raises(Exception):
        PolicyEnsemble.load(str(tmp_path))


class LoadReturnsWrongTypePolicy(Policy):
    @classmethod
    def load(cls, _) -> Text:
        return ""

    def persist(self, _) -> None:
        pass

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        pass


def test_policy_loading_load_returns_wrong_type(tmp_path: Path):
    original_policy_ensemble = PolicyEnsemble([LoadReturnsWrongTypePolicy()])
    original_policy_ensemble.train([], None, RegexInterpreter())
    original_policy_ensemble.persist(str(tmp_path))

    with pytest.raises(Exception):
        PolicyEnsemble.load(str(tmp_path))


@pytest.mark.parametrize(
    "valid_config",
    [
        {"policy": [{"name": "MemoizationPolicy"}]},
        {"policies": [{"name": "MemoizationPolicy"}]},
    ],
)
def test_valid_policy_configurations(valid_config):
    assert PolicyEnsemble.from_dict(valid_config)


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"police": [{"name": "MemoizationPolicy"}]},
        {"policies": []},
        {"policies": [{"name": "ykaüoppodas"}]},
        {"policy": [{"name": "ykaüoppodas"}]},
        {"policy": [{"name": "ykaüoppodas.bladibla"}]},
    ],
)
def test_invalid_policy_configurations(invalid_config):
    with pytest.raises(InvalidPolicyConfig):
        PolicyEnsemble.from_dict(invalid_config)


def test_from_dict_does_not_change_passed_dict_parameter():
    config = {
        "policies": [
            {
                "name": "TEDPolicy",
                "featurizer": [
                    {
                        "name": "MaxHistoryTrackerFeaturizer",
                        "max_history": 5,
                        "state_featurizer": [{"name": "BinarySingleStateFeaturizer"}],
                    }
                ],
            }
        ]
    }

    config_copy = copy.deepcopy(config)
    PolicyEnsemble.from_dict(config_copy)

    assert config == config_copy


def test_rule_based_data_warnings_no_rule_trackers():
    trackers = [DialogueStateTracker("some-id", slots=[], is_rule_tracker=False)]
    policies = [RulePolicy()]
    ensemble = SimplePolicyEnsemble(policies)

    with pytest.warns(UserWarning) as record:
        ensemble.train(trackers, Domain.empty(), RegexInterpreter())

    assert (
        "Found a rule-based policy in your pipeline but no rule-based training data."
    ) in record[0].message.args[0]


def test_rule_based_data_warnings_no_rule_policy():
    trackers = [DialogueStateTracker("some-id", slots=[], is_rule_tracker=True)]
    policies = [FallbackPolicy()]
    ensemble = SimplePolicyEnsemble(policies)

    with pytest.warns(UserWarning) as record:
        ensemble.train(trackers, Domain.empty(), RegexInterpreter())

    assert (
        "Found rule-based training data but no policy supporting rule-based data."
    ) in record[0].message.args[0]


@pytest.mark.parametrize(
    "policies",
    [
        ["RulePolicy", "MappingPolicy"],
        ["RulePolicy", "FallbackPolicy"],
        ["RulePolicy", "TwoStageFallbackPolicy"],
        ["RulePolicy", "FormPolicy"],
        ["RulePolicy", "FallbackPolicy", "FormPolicy"],
    ],
)
def test_mutual_exclusion_of_rule_policy_and_old_rule_like_policies(
    policies: List[Text],
):
    policy_config = [{"name": policy_name} for policy_name in policies]
    with pytest.warns(UserWarning):
        PolicyEnsemble.from_dict({"policies": policy_config})


def test_end_to_end_prediction_supersedes_others(default_domain: Domain):
    expected_action_index = 2
    expected_confidence = 0.5
    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(priority=100, predict_index=0),
            ConstantPolicy(
                priority=1,
                predict_index=expected_action_index,
                confidence=expected_confidence,
                is_end_to_end_prediction=True,
            ),
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])

    prediction = ensemble.probabilities_using_best_policy(
        tracker, default_domain, RegexInterpreter()
    )

    assert prediction.max_confidence == expected_confidence
    assert prediction.max_confidence_index == expected_action_index
    assert prediction.policy_name == f"policy_1_{ConstantPolicy.__name__}"


def test_prediction_applies_must_have_policy_events(default_domain: Domain):
    must_have_events = [ActionExecuted("my action")]

    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(priority=10, predict_index=1),
            ConstantPolicy(priority=1, predict_index=2, events=must_have_events),
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])

    prediction = ensemble.probabilities_using_best_policy(
        tracker, default_domain, RegexInterpreter()
    )

    # Policy 0 won due to higher prio
    assert prediction.policy_name == f"policy_0_{ConstantPolicy.__name__}"

    # Events of losing policy were applied nevertheless
    assert prediction.events == must_have_events


def test_prediction_applies_optional_policy_events(default_domain: Domain):
    optional_events = [ActionExecuted("my action")]
    must_have_events = [SlotSet("some slot", "some value")]

    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(
                priority=10,
                predict_index=1,
                events=must_have_events,
                optional_events=optional_events,
            ),
            ConstantPolicy(priority=1, predict_index=2),
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])

    prediction = ensemble.probabilities_using_best_policy(
        tracker, default_domain, RegexInterpreter()
    )

    # Policy 0 won due to higher prio
    assert prediction.policy_name == f"policy_0_{ConstantPolicy.__name__}"

    # Events of losing policy were applied nevertheless
    assert len(prediction.events) == len(optional_events) + len(must_have_events)
    assert all(event in prediction.events for event in optional_events)
    assert all(event in prediction.events for event in must_have_events)


def test_with_float_returning_policy(default_domain: Domain):
    expected_index = 3

    class OldPolicy(Policy):
        def predict_action_probabilities(
            self,
            tracker: DialogueStateTracker,
            domain: Domain,
            interpreter: NaturalLanguageInterpreter,
            **kwargs: Any,
        ) -> List[float]:
            prediction = [0.0] * default_domain.num_actions
            prediction[expected_index] = 3
            return prediction

    ensemble = SimplePolicyEnsemble(
        [ConstantPolicy(priority=1, predict_index=1), OldPolicy(priority=1)]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])

    with pytest.warns(FutureWarning):
        prediction = ensemble.probabilities_using_best_policy(
            tracker, default_domain, RegexInterpreter()
        )

    assert prediction.policy_name == f"policy_1_{OldPolicy.__name__}"
    assert prediction.max_confidence_index == expected_index


@pytest.mark.parametrize(
    "policy_name, confidence, not_in_training_data",
    [
        (f"policy_1_{MemoizationPolicy.__name__}", 1.0, False),
        (f"policy_1_{AugmentedMemoizationPolicy.__name__}", 1.0, False),
        (f"policy_1_{AugmentedMemoizationPolicy.__name__}", None, False),
        (f"policy_1_{RulePolicy.__name__}", 1.0, False),
        ("any", 0.0, True),
    ],
)
def test_is_not_in_training_data(
    policy_name: Text, confidence: Optional[float], not_in_training_data: bool
):
    assert (
        SimplePolicyEnsemble.is_not_in_training_data(policy_name, confidence)
        == not_in_training_data
    )
