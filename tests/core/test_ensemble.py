import pytest

from rasa.core.policies import Policy
from rasa.core.policies.ensemble import (
    PolicyEnsemble,
    InvalidPolicyConfig,
    SimplePolicyEnsemble,
)
from rasa.core.domain import Domain
from rasa.core.trackers import DialogueStateTracker
from rasa.core.events import UserUttered


class WorkingPolicy(Policy):
    @classmethod
    def load(cls, path):
        return WorkingPolicy()

    def persist(self, path):
        pass

    def train(self, training_trackers, domain, **kwargs):
        pass

    def predict_action_probabilities(self, tracker, domain):
        pass

    def __eq__(self, other):
        return isinstance(other, WorkingPolicy)


def test_policy_loading_simple(tmpdir):
    original_policy_ensemble = PolicyEnsemble([WorkingPolicy()])
    original_policy_ensemble.train([], None)
    original_policy_ensemble.persist(str(tmpdir))

    loaded_policy_ensemble = PolicyEnsemble.load(str(tmpdir))
    assert original_policy_ensemble.policies == loaded_policy_ensemble.policies


class ConstantPolicy(Policy):
    def __init__(self, priority: int = None, predict_index: int = None) -> None:
        super(ConstantPolicy, self).__init__(priority=priority)
        self.predict_index = predict_index

    @classmethod
    def load(cls, path):
        pass

    def persist(self, path):
        pass

    def train(self, training_trackers, domain, **kwargs):
        pass

    def predict_action_probabilities(self, tracker, domain):
        result = [0.0] * domain.num_actions
        result[self.predict_index] = 1.0
        return result


def test_policy_priority():
    domain = Domain.load("data/test_domains/default.yml")
    tracker = DialogueStateTracker.from_events("test", [UserUttered("hi")], [])

    priority_1 = ConstantPolicy(priority=1, predict_index=0)
    priority_2 = ConstantPolicy(priority=2, predict_index=1)

    policy_ensemble_0 = SimplePolicyEnsemble([priority_1, priority_2])
    policy_ensemble_1 = SimplePolicyEnsemble([priority_2, priority_1])

    priority_2_result = priority_2.predict_action_probabilities(tracker, domain)

    i = 1  # index of priority_2 in ensemble_0
    result, best_policy = policy_ensemble_0.probabilities_using_best_policy(
        tracker, domain
    )
    assert best_policy == "policy_{}_{}".format(i, type(priority_2).__name__)
    assert result.tolist() == priority_2_result

    i = 0  # index of priority_2 in ensemble_1
    result, best_policy = policy_ensemble_1.probabilities_using_best_policy(
        tracker, domain
    )
    assert best_policy == "policy_{}_{}".format(i, type(priority_2).__name__)
    assert result.tolist() == priority_2_result


class LoadReturnsNonePolicy(Policy):
    @classmethod
    def load(cls, path):
        return None

    def persist(self, path):
        pass

    def train(self, training_trackers, domain, **kwargs):
        pass

    def predict_action_probabilities(self, tracker, domain):
        pass


def test_policy_loading_load_returns_none(tmpdir):
    original_policy_ensemble = PolicyEnsemble([LoadReturnsNonePolicy()])
    original_policy_ensemble.train([], None)
    original_policy_ensemble.persist(str(tmpdir))

    with pytest.raises(Exception):
        PolicyEnsemble.load(str(tmpdir))


class LoadReturnsWrongTypePolicy(Policy):
    @classmethod
    def load(cls, path):
        return ""

    def persist(self, path):
        pass

    def train(self, training_trackers, domain, **kwargs):
        pass

    def predict_action_probabilities(self, tracker, domain):
        pass


def test_policy_loading_load_returns_wrong_type(tmpdir):
    original_policy_ensemble = PolicyEnsemble([LoadReturnsWrongTypePolicy()])
    original_policy_ensemble.train([], None)
    original_policy_ensemble.persist(str(tmpdir))

    with pytest.raises(Exception):
        PolicyEnsemble.load(str(tmpdir))


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
