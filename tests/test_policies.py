from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core import training

try:  # py3
    from unittest.mock import patch
except ImportError:  # py2
    from mock import patch
import numpy as np
import pytest

from rasa_core.channels import UserMessage
from rasa_core.domain import TemplateDomain
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import \
    MemoizationPolicy, AugmentedMemoizationPolicy
from rasa_core.policies.sklearn_policy import SklearnPolicy
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.trackers import DialogueStateTracker
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE
from rasa_core.featurizers import (
    MaxHistoryTrackerFeaturizer,
    BinarySingleStateFeaturizer, FullDialogueTrackerFeaturizer)
from rasa_core.events import ActionExecuted
from tests.utilities import read_dialogue_file


def train_trackers(domain):
    trackers = training.load_data(
        DEFAULT_STORIES_FILE,
        domain
    )
    return trackers


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

    def create_policy(self, featurizer):
        raise NotImplementedError

    @pytest.fixture(scope="module")
    def featurizer(self):
        featurizer = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                                 max_history=self.max_history)
        return featurizer

    @pytest.fixture(scope="module")
    def trained_policy(self, featurizer):
        default_domain = TemplateDomain.load(DEFAULT_DOMAIN_PATH)
        policy = self.create_policy(featurizer)
        training_trackers = train_trackers(default_domain)
        policy.train(training_trackers, default_domain)
        return policy

    def test_persist_and_load(self, trained_policy, default_domain, tmpdir):
        trained_policy.persist(tmpdir.strpath)
        loaded = trained_policy.__class__.load(tmpdir.strpath)
        trackers = train_trackers(default_domain)

        for tracker in trackers:
            predicted_probabilities = loaded.predict_action_probabilities(
                tracker, default_domain)
            actual_probabilities = trained_policy.predict_action_probabilities(
                tracker, default_domain)
            assert predicted_probabilities == actual_probabilities

    def test_prediction_on_empty_tracker(self, trained_policy, default_domain):
        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                       default_domain.slots)
        probabilities = trained_policy.predict_action_probabilities(
            tracker, default_domain)
        assert len(probabilities) == default_domain.num_actions
        assert max(probabilities) <= 1.0
        assert min(probabilities) >= 0.0

    def test_persist_and_load_empty_policy(self, tmpdir):
        empty_policy = self.create_policy(None)
        empty_policy.persist(tmpdir.strpath)
        loaded = empty_policy.__class__.load(tmpdir.strpath)
        assert loaded is not None


class TestKerasPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        p = KerasPolicy(featurizer)
        return p


class TestFallbackPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        p = FallbackPolicy()
        return p

    @pytest.mark.parametrize(
        "nlu_confidence, prev_action_is_fallback, should_fallback",
        [
            (0.1, True, False),
            (0.1, False, True),
            (0.9, True, False),
            (0.9, False, False),
        ])
    def test_something(self,
                       trained_policy,
                       nlu_confidence,
                       prev_action_is_fallback,
                       should_fallback):
        last_action_name = trained_policy.fallback_action_name if \
            prev_action_is_fallback else 'not_fallback'
        assert trained_policy.should_fallback(
            nlu_confidence, last_action_name) is should_fallback


class TestMemoizationPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        max_history = None
        if isinstance(featurizer, MaxHistoryTrackerFeaturizer):
            max_history = featurizer.max_history
        p = MemoizationPolicy(max_history=max_history)
        return p

    def test_memorise(self, trained_policy, default_domain):
        trackers = train_trackers(default_domain)
        trained_policy.train(trackers, default_domain)

        (all_states, all_actions) = \
            trained_policy.featurizer.training_states_and_actions(
                trackers, default_domain)

        for tracker, states, actions in zip(trackers, all_states, all_actions):
            recalled = trained_policy.recall(states, tracker, default_domain)
            assert recalled == default_domain.index_for_action(actions[0])

        nums = np.random.randn(default_domain.num_states)
        random_states = [{f: num
                          for f, num in
                          zip(default_domain.input_states, nums)}]
        assert trained_policy._recall_states(random_states) is None

    def test_memorise_with_nlu(self, trained_policy, default_domain):
        filename = "data/test_dialogues/nlu_dialogue.json"
        dialogue = read_dialogue_file(filename)

        tracker = DialogueStateTracker(dialogue.name, default_domain.slots)
        tracker.recreate_from_dialogue(dialogue)
        states = trained_policy.featurizer.prediction_states([tracker],
                                                             default_domain)[0]

        recalled = trained_policy.recall(states, tracker, default_domain)
        assert recalled is not None


class TestAugmentedMemoizationPolicy(PolicyTestCollection):

    @pytest.fixture(scope="module")
    def create_policy(self, featurizer):
        max_history = None
        if isinstance(featurizer, MaxHistoryTrackerFeaturizer):
            max_history = featurizer.max_history
        p = AugmentedMemoizationPolicy(max_history=max_history)
        return p


class TestSklearnPolicy(PolicyTestCollection):

    def create_policy(self, featurizer, **kwargs):
        p = SklearnPolicy(featurizer, **kwargs)
        return p

    @pytest.yield_fixture
    def mock_search(self):
        with patch('rasa_core.policies.sklearn_policy.GridSearchCV') as gs:
            gs.best_estimator_ = 'mockmodel'
            gs.best_score_ = 0.123
            gs.return_value = gs  # for __init__
            yield gs

    @pytest.fixture(scope='module')
    def default_domain(self):
        return TemplateDomain.load(DEFAULT_DOMAIN_PATH)

    @pytest.fixture
    def tracker(self, default_domain):
        return DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                    default_domain.slots)

    @pytest.fixture(scope='module')
    def trackers(self, default_domain):
        return train_trackers(default_domain)

    def test_cv_none_does_not_trigger_search(
            self, mock_search, default_domain, trackers, featurizer):
        policy = self.create_policy(
            featurizer=featurizer,
            cv=None
        )
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count == 0
        assert policy.model != 'mockmodel'

    def test_cv_not_none_param_grid_none_triggers_search_without_params(
            self, mock_search, default_domain, trackers, featurizer):
        policy = self.create_policy(
            featurizer=featurizer,
            cv=3,
        )
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]['cv'] == 3
        assert mock_search.call_args_list[0][1]['param_grid'] == {}
        assert policy.model == 'mockmodel'

    def test_cv_not_none_param_grid_none_triggers_search_with_params(
            self, mock_search, default_domain, trackers, featurizer):
        param_grid = {'n_estimators': 50}
        policy = self.create_policy(
            featurizer=featurizer,
            cv=3,
            param_grid=param_grid,
        )
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]['cv'] == 3
        assert mock_search.call_args_list[0][1]['param_grid'] == param_grid
        assert policy.model == 'mockmodel'

    def test_missing_classes_filled_correctly(
            self, default_domain, trackers, tracker, featurizer):
        # Pretend that a couple of classes are missing and check that
        # those classes are predicted as 0, while the other class
        # probabilities are predicted normally.
        policy = self.create_policy(featurizer=featurizer, cv=None)

        classes = [1, 3]
        new_trackers = []
        for tr in trackers:
            new_tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                               default_domain.slots)
            for e in tr.applied_events():
                if isinstance(e, ActionExecuted):
                    new_action = default_domain.action_for_index(
                        np.random.choice(classes)).name()
                    new_tracker.update(ActionExecuted(new_action))
                else:
                    new_tracker.update(e)

            new_trackers.append(new_tracker)

        policy.train(new_trackers, domain=default_domain)
        predicted_probabilities = policy.predict_action_probabilities(
            tracker, default_domain)

        assert len(predicted_probabilities) == default_domain.num_actions
        assert np.allclose(sum(predicted_probabilities), 1.0)
        for i, prob in enumerate(predicted_probabilities):
            if i in classes:
                assert prob >= 0.0
            else:
                assert prob == 0.0

    def test_train_kwargs_are_set_on_model(
            self, default_domain, trackers, featurizer):
        policy = self.create_policy(featurizer=featurizer, cv=None)
        policy.train(trackers, domain=default_domain, C=123)
        assert policy.model.C == 123

    def test_train_with_shuffle_false(
            self, default_domain, trackers, featurizer):
        policy = self.create_policy(featurizer=featurizer, shuffle=False)
        # does not raise
        policy.train(trackers, domain=default_domain)
