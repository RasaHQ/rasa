from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pytest

from rasa_core.channels import UserMessage
from rasa_core.domain import TemplateDomain
from rasa_core.featurizers import BinaryFeaturizer
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.scoring_policy import ScoringPolicy
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training_utils import extract_training_data_from_file, \
    extract_stories_from_file
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


def train_data(max_history, domain):
    return extract_training_data_from_file(
            "data/dsl_stories/stories_defaultdomain.md",
            domain=domain, max_history=max_history, remove_duplicates=True,
            featurizer=BinaryFeaturizer())


# We are going to use class style testing here since unfortunately pytest
# doesn't support using fixtures as arguments to its own parameterize yet
# (hence, we can't train a policy, declare it as a fixture and use the different
# fixtures of the different policies for the functional tests). Therefore, we
# are going to reverse this and train the policy within a class and collect the
# tests in a base class.
class PolicyTestCollection(object):
    """Tests every policy needs to fulfill.

    Each policy can declare further tests on its own."""

    max_history = 3  # this is the amount of history we test on

    def create_policy(self):
        raise NotImplementedError

    @pytest.fixture(scope="module")
    def trained_policy(self):
        default_domain = TemplateDomain.load(DEFAULT_DOMAIN_PATH)
        policy = self.create_policy()
        X, y = train_data(self.max_history, default_domain)
        policy.max_history = self.max_history
        policy.featurizer = BinaryFeaturizer()
        policy.train(X, y, default_domain)
        return policy

    def test_persist_and_load(self, trained_policy, default_domain, tmpdir):
        trained_policy.persist(tmpdir.strpath)
        loaded = trained_policy.__class__.load(tmpdir.strpath,
                                               trained_policy.featurizer,
                                               trained_policy.max_history)
        stories = extract_stories_from_file(
                DEFAULT_STORIES_FILE, default_domain)

        for story in stories:
            tracker = DialogueStateTracker("default", default_domain.slots)
            dialogue = story.as_dialogue("default", default_domain)
            tracker.recreate_from_dialogue(dialogue)
            predicted_probabilities = loaded.predict_action_probabilities(
                    tracker, default_domain)
            actual_probabilities = trained_policy.predict_action_probabilities(
                    tracker, default_domain)
            assert predicted_probabilities == actual_probabilities

    def test_prediction_on_empty_tracker(self, trained_policy, default_domain):
        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER,
                                       default_domain.slots,
                                       default_domain.topics,
                                       default_domain.default_topic)
        probabilities = trained_policy.predict_action_probabilities(
                tracker, default_domain)
        assert len(probabilities) == default_domain.num_actions
        assert max(probabilities) <= 1.0
        assert min(probabilities) >= 0.0

    def test_persist_and_load_empty_policy(self, tmpdir):
        empty_policy = self.create_policy()
        empty_policy.persist(tmpdir.strpath)
        loaded = empty_policy.__class__.load(tmpdir.strpath, BinaryFeaturizer(),
                                             empty_policy.max_history)
        assert loaded is not None


class TestKerasPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self):
        p = KerasPolicy()
        return p


class TestScoringPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self):
        p = ScoringPolicy()
        return p


class TestMemoizationPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self):
        p = MemoizationPolicy()
        return p

    def test_memorise(self, trained_policy, default_domain):
        X, y = train_data(self.max_history, default_domain)
        trained_policy.train(X, y, default_domain)

        for ii in range(X.shape[0]):
            assert trained_policy.recall(X[ii, :, :], default_domain) == y[ii]

        random_feature = np.random.randn(default_domain.num_features)
        assert trained_policy.recall(random_feature, default_domain) is None
