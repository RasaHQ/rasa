import pytest

from rasa_core.policies import Policy
from rasa_core.policies.ensemble import PolicyEnsemble


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
    original_policy_ensemble.persist(tmpdir)

    loaded_policy_ensemble = PolicyEnsemble.load(tmpdir)
    assert original_policy_ensemble.policies == loaded_policy_ensemble.policies


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
    original_policy_ensemble.persist(tmpdir)

    with pytest.raises(Exception):
        PolicyEnsemble.load(tmpdir)


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
    original_policy_ensemble.persist(tmpdir)

    with pytest.raises(Exception):
        PolicyEnsemble.load(tmpdir)
