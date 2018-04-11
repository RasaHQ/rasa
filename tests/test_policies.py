from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:  # py3
    from unittest.mock import patch
except ImportError:  # py2
    from mock import patch
import numpy as np
import pytest

from rasa_core.channels import UserMessage
from rasa_core.domain import TemplateDomain
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.scoring_policy import ScoringPolicy
from rasa_core.policies.sklearn_policy import SklearnPolicy
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training import (
    DialogueTrainingData)
from tests.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE


# TODO new tests due to new system
def train_data(max_history, domain):
    return extract_training_data(
            DEFAULT_STORIES_FILE, domain,
            BinaryFeaturizer(), max_history=max_history, remove_duplicates=True)


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
        training_data = train_data(self.max_history, default_domain)
        policy.max_history = self.max_history
        policy.featurizer = BinaryFeaturizer()
        policy.train(training_data, default_domain)
        return policy

    def test_persist_and_load(self, trained_policy, default_domain, tmpdir):
        trained_policy.persist(tmpdir.strpath)
        loaded = trained_policy.__class__.load(tmpdir.strpath,
                                               trained_policy.featurizer,
                                               trained_policy.max_history)
        trackers = extract_trackers(
                DEFAULT_STORIES_FILE, default_domain, BinaryFeaturizer())

        for tracker in trackers:
            predicted_probabilities = loaded.predict_action_probabilities(
                    tracker, default_domain)
            actual_probabilities = trained_policy.predict_action_probabilities(
                    tracker, default_domain)
            assert predicted_probabilities == actual_probabilities

    def test_prediction_on_empty_tracker(self, trained_policy, default_domain):
        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
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
        training_data = train_data(self.max_history, default_domain)
        trained_policy.train(training_data, default_domain)

        for ii in range(training_data.num_examples()):
            recalled = trained_policy.recall(training_data.X[ii, :, :],
                                             default_domain)
            assert recalled == training_data.y[ii]

        random_feature = np.random.randn(default_domain.num_features)
        assert trained_policy.recall(random_feature, default_domain) is None


class TestSklearnPolicy(PolicyTestCollection):
    def create_policy(self, **kwargs):
        p = SklearnPolicy(**kwargs)
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
                                    default_domain.slots,
                                    default_domain.topics,
                                    default_domain.default_topic)

    @pytest.fixture(scope='module')
    def data(self, default_domain):
        return train_data(self.max_history, default_domain)

    def test_cv_none_does_not_trigger_search(
            self, mock_search, default_domain, data):
        policy = self.create_policy(
            featurizer=BinaryFeaturizer(),
            max_history=self.max_history,
            cv=None,
        )
        policy.train(data, domain=default_domain)

        assert mock_search.call_count == 0
        assert policy.model != 'mockmodel'

    def test_cv_not_none_param_grid_none_triggers_search_without_params(
            self, mock_search, default_domain, data):
        policy = self.create_policy(
            featurizer=BinaryFeaturizer(),
            max_history=self.max_history,
            cv=3,
        )
        policy.train(data, domain=default_domain)

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]['cv'] == 3
        assert mock_search.call_args_list[0][1]['param_grid'] == {}
        assert policy.model == 'mockmodel'

    def test_cv_not_none_param_grid_none_triggers_search_with_params(
            self, mock_search, default_domain, data):
        param_grid = {'n_estimators': 50}
        policy = self.create_policy(
            featurizer=BinaryFeaturizer(),
            max_history=self.max_history,
            cv=3,
            param_grid=param_grid,
        )
        policy.train(data, domain=default_domain)

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]['cv'] == 3
        assert mock_search.call_args_list[0][1]['param_grid'] == param_grid
        assert policy.model == 'mockmodel'

    def test_continue_training_with_unsuitable_model_raises(
            self, default_domain, data):
        policy = self.create_policy(
            featurizer=BinaryFeaturizer(),
            max_history=self.max_history,
            cv=None,
        )
        policy.train(data, domain=default_domain)

        with pytest.raises(TypeError) as exc:
            policy.continue_training(data, domain=default_domain)

        assert exc.value.args[0] == (
            "Continuing training is only possible with "
            "sklearn models that support 'partial_fit'.")

    def test_missing_classes_filled_correctly(
            self, default_domain, data, tracker):
        # Pretend that a couple of classes are missing and check that
        # those classes are predicted as 0, while the other class
        # probabilities are predicted normally.
        policy = self.create_policy(
            featurizer=BinaryFeaturizer(),
            max_history=self.max_history,
            cv=None,
        )
        X = data.X
        classes = [3, 4, 7]
        y = np.asarray([np.random.choice(classes) for _ in X])
        data = DialogueTrainingData(X, y)

        policy.train(data, domain=default_domain)
        predicted_probabilities = policy.predict_action_probabilities(
            tracker, default_domain)

        assert len(predicted_probabilities) == 8
        assert np.allclose(sum(predicted_probabilities), 1.0)
        for i, prob in enumerate(predicted_probabilities):
            if i in classes:
                assert prob >= 0.0
            else:
                assert prob == 0.0

    def test_train_kwargs_are_set_on_model(self, default_domain, data):
        policy = self.create_policy(
            featurizer=BinaryFeaturizer(),
            max_history=self.max_history,
            cv=None,
        )
        policy.train(data, domain=default_domain, C=123)
        assert policy.model.C == 123

    def test_train_with_shuffle_false(self, default_domain, data):
        policy = self.create_policy(shuffle=False)
        # does not raise
        policy.train(data, domain=default_domain)
