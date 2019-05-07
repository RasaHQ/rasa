from unittest.mock import patch

import numpy as np
import pytest

import rasa.utils.io
from rasa.core import training, utils
from rasa.core.actions.action import (
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_LISTEN_NAME,
    ActionRevertFallbackEvents,
)
from rasa.core.channels import UserMessage
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.events import ActionExecuted
from rasa.core.featurizers import (
    BinarySingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.core.policies import TwoStageFallbackPolicy
from rasa.core.policies.embedding_policy import EmbeddingPolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.keras_policy import KerasPolicy
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from rasa.core.policies.sklearn_policy import SklearnPolicy
from rasa.core.trackers import DialogueStateTracker
from tests.core.conftest import DEFAULT_DOMAIN_PATH, DEFAULT_STORIES_FILE
from tests.core.utilities import get_tracker, read_dialogue_file, user_uttered


def tf_defaults():
    return {
        "tf_config": {
            "device_count": {"CPU": 4},
            # tell tf.Session to use CPU limit, if you have
            # more CPU, you can increase this value appropriately
            "inter_op_parallelism_threads": 0,
            # the number of threads in the thread pool available
            # for each process for blocking operation nodes set to 0
            # to allow the system to select the appropriate value.
            "intra_op_parallelism_threads": 0,  # tells the degree of thread
            # parallelism of the tf.Session operation.
            # the smaller the value, the less reuse the thread will have
            # and the more likely it will use more CPU cores.
            # if the value is 0,
            # tensorflow will automatically select an appropriate value.
            "gpu_options": {"allow_growth": True}
            # if set True, will try to allocate
            # as much GPU memory as possible to support running
        }
    }


def session_config():
    import tensorflow as tf

    return tf.ConfigProto(**tf_defaults()["tf_config"])


async def train_trackers(domain, augmentation_factor=20):
    return await training.load_data(
        DEFAULT_STORIES_FILE, domain, augmentation_factor=augmentation_factor
    )


@pytest.fixture(scope="module")
def loop():
    from pytest_sanic.plugin import loop as sanic_loop

    return rasa.utils.io.enable_async_loop_debugging(next(sanic_loop()))


# We are going to use class style testing here since unfortunately pytest
# doesn't support using fixtures as arguments to its own parameterize yet
# (hence, we can't train a policy, declare it as a fixture and use the
# different fixtures of the different policies for the functional tests).
# Therefore, we are going to reverse this and train the policy within a class
# and collect the tests in a base class.
# noinspection PyMethodMayBeStatic
class PolicyTestCollection(object):
    """Tests every policy needs to fulfill.

    Each policy can declare further tests on its own."""

    max_history = 3  # this is the amount of history we test on

    def create_policy(self, featurizer, priority):
        raise NotImplementedError

    @pytest.fixture(scope="module")
    def featurizer(self):
        featurizer = MaxHistoryTrackerFeaturizer(
            BinarySingleStateFeaturizer(), max_history=self.max_history
        )
        return featurizer

    @pytest.fixture(scope="module")
    def priority(self):
        return 1

    @pytest.fixture(scope="module")
    async def trained_policy(self, featurizer, priority):
        default_domain = Domain.load(DEFAULT_DOMAIN_PATH)
        policy = self.create_policy(featurizer, priority)
        training_trackers = await train_trackers(default_domain, augmentation_factor=20)
        policy.train(training_trackers, default_domain)
        return policy

    async def test_persist_and_load(self, trained_policy, default_domain, tmpdir):
        trained_policy.persist(tmpdir.strpath)
        loaded = trained_policy.__class__.load(tmpdir.strpath)
        trackers = await train_trackers(default_domain, augmentation_factor=20)

        for tracker in trackers:
            predicted_probabilities = loaded.predict_action_probabilities(
                tracker, default_domain
            )
            actual_probabilities = trained_policy.predict_action_probabilities(
                tracker, default_domain
            )
            assert predicted_probabilities == actual_probabilities

    def test_prediction_on_empty_tracker(self, trained_policy, default_domain):
        tracker = DialogueStateTracker(
            UserMessage.DEFAULT_SENDER_ID, default_domain.slots
        )
        probabilities = trained_policy.predict_action_probabilities(
            tracker, default_domain
        )
        assert len(probabilities) == default_domain.num_actions
        assert max(probabilities) <= 1.0
        assert min(probabilities) >= 0.0

    @pytest.mark.filterwarnings(
        "ignore:.*without a trained model present.*:UserWarning"
    )
    def test_persist_and_load_empty_policy(self, tmpdir):
        empty_policy = self.create_policy(None, None)
        empty_policy.persist(tmpdir.strpath)
        loaded = empty_policy.__class__.load(tmpdir.strpath)
        assert loaded is not None

    def test_tf_config(self, trained_policy, tmpdir):
        if hasattr(trained_policy, "session"):
            # noinspection PyProtectedMember
            assert trained_policy.session._config is None
            trained_policy.persist(tmpdir.strpath)
            loaded = trained_policy.__class__.load(tmpdir.strpath)
            # noinspection PyProtectedMember
            assert loaded.session._config is None


class TestKerasPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        p = KerasPolicy(featurizer, priority)
        return p


class TestKerasPolicyWithTfConfig(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        p = KerasPolicy(featurizer, priority, **tf_defaults())
        return p

    def test_tf_config(self, trained_policy, tmpdir):
        # noinspection PyProtectedMember
        assert trained_policy.session._config == session_config()
        trained_policy.persist(tmpdir.strpath)
        loaded = trained_policy.__class__.load(tmpdir.strpath)
        # noinspection PyProtectedMember
        assert loaded.session._config == session_config()


class TestFallbackPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        p = FallbackPolicy(priority=priority)
        return p

    @pytest.mark.parametrize(
        "nlu_confidence, last_action_name, should_nlu_fallback",
        [
            (0.1, "some_action", False),
            (0.1, "action_listen", True),
            (0.9, "some_action", False),
            (0.9, "action_listen", False),
        ],
    )
    def test_should_nlu_fallback(
        self, trained_policy, nlu_confidence, last_action_name, should_nlu_fallback
    ):
        assert (
            trained_policy.should_nlu_fallback(nlu_confidence, last_action_name)
            is should_nlu_fallback
        )


class TestMappingPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        p = MappingPolicy()
        return p


class TestMemoizationPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        max_history = None
        if isinstance(featurizer, MaxHistoryTrackerFeaturizer):
            max_history = featurizer.max_history
        p = MemoizationPolicy(priority=priority, max_history=max_history)
        return p

    async def test_memorise(self, trained_policy, default_domain):
        trackers = await train_trackers(default_domain, augmentation_factor=20)
        trained_policy.train(trackers, default_domain)
        lookup_with_augmentation = trained_policy.lookup

        trackers = [
            t for t in trackers if not hasattr(t, "is_augmented") or not t.is_augmented
        ]

        (
            all_states,
            all_actions,
        ) = trained_policy.featurizer.training_states_and_actions(
            trackers, default_domain
        )

        for tracker, states, actions in zip(trackers, all_states, all_actions):
            recalled = trained_policy.recall(states, tracker, default_domain)
            assert recalled == default_domain.index_for_action(actions[0])

        nums = np.random.randn(default_domain.num_states)
        random_states = [{f: num for f, num in zip(default_domain.input_states, nums)}]
        assert trained_policy._recall_states(random_states) is None

        # compare augmentation for augmentation_factor of 0 and 20:
        trackers_no_augmentation = await train_trackers(
            default_domain, augmentation_factor=0
        )
        trained_policy.train(trackers_no_augmentation, default_domain)
        lookup_no_augmentation = trained_policy.lookup

        assert lookup_no_augmentation == lookup_with_augmentation

    def test_memorise_with_nlu(self, trained_policy, default_domain):
        filename = "data/test_dialogues/default.json"
        dialogue = read_dialogue_file(filename)

        tracker = DialogueStateTracker(dialogue.name, default_domain.slots)
        tracker.recreate_from_dialogue(dialogue)
        states = trained_policy.featurizer.prediction_states([tracker], default_domain)[
            0
        ]

        recalled = trained_policy.recall(states, tracker, default_domain)
        assert recalled is not None


class TestAugmentedMemoizationPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        max_history = None
        if isinstance(featurizer, MaxHistoryTrackerFeaturizer):
            max_history = featurizer.max_history
        p = AugmentedMemoizationPolicy(priority=priority, max_history=max_history)
        return p


class TestSklearnPolicy(PolicyTestCollection):
    def create_policy(self, featurizer, priority, **kwargs):
        p = SklearnPolicy(featurizer, priority, **kwargs)
        return p

    @pytest.yield_fixture
    def mock_search(self):
        with patch("rasa.core.policies.sklearn_policy.GridSearchCV") as gs:
            gs.best_estimator_ = "mockmodel"
            gs.best_score_ = 0.123
            gs.return_value = gs  # for __init__
            yield gs

    @pytest.fixture(scope="module")
    def default_domain(self):
        return Domain.load(DEFAULT_DOMAIN_PATH)

    @pytest.fixture
    def tracker(self, default_domain):
        return DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID, default_domain.slots)

    @pytest.fixture(scope="module")
    async def trackers(self, default_domain):
        return await train_trackers(default_domain, augmentation_factor=20)

    def test_cv_none_does_not_trigger_search(
        self, mock_search, default_domain, trackers, featurizer, priority
    ):
        policy = self.create_policy(featurizer=featurizer, priority=priority, cv=None)
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count == 0
        assert policy.model != "mockmodel"

    def test_cv_not_none_param_grid_none_triggers_search_without_params(
        self, mock_search, default_domain, trackers, featurizer, priority
    ):

        policy = self.create_policy(featurizer=featurizer, priority=priority, cv=3)
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]["cv"] == 3
        assert mock_search.call_args_list[0][1]["param_grid"] == {}
        assert policy.model == "mockmodel"

    def test_cv_not_none_param_grid_none_triggers_search_with_params(
        self, mock_search, default_domain, trackers, featurizer, priority
    ):
        param_grid = {"n_estimators": 50}
        policy = self.create_policy(
            featurizer=featurizer, priority=priority, cv=3, param_grid=param_grid
        )
        policy.train(trackers, domain=default_domain)

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]["cv"] == 3
        assert mock_search.call_args_list[0][1]["param_grid"] == param_grid
        assert policy.model == "mockmodel"

    def test_missing_classes_filled_correctly(
        self, default_domain, trackers, tracker, featurizer, priority
    ):
        # Pretend that a couple of classes are missing and check that
        # those classes are predicted as 0, while the other class
        # probabilities are predicted normally.
        policy = self.create_policy(featurizer=featurizer, priority=priority, cv=None)

        classes = [1, 3]
        new_trackers = []
        for tr in trackers:
            new_tracker = DialogueStateTracker(
                UserMessage.DEFAULT_SENDER_ID, default_domain.slots
            )
            for e in tr.applied_events():
                if isinstance(e, ActionExecuted):
                    new_action = default_domain.action_for_index(
                        np.random.choice(classes), action_endpoint=None
                    ).name()
                    new_tracker.update(ActionExecuted(new_action))
                else:
                    new_tracker.update(e)

            new_trackers.append(new_tracker)

        policy.train(new_trackers, domain=default_domain)
        predicted_probabilities = policy.predict_action_probabilities(
            tracker, default_domain
        )

        assert len(predicted_probabilities) == default_domain.num_actions
        assert np.allclose(sum(predicted_probabilities), 1.0)
        for i, prob in enumerate(predicted_probabilities):
            if i in classes:
                assert prob >= 0.0
            else:
                assert prob == 0.0

    def test_train_kwargs_are_set_on_model(
        self, default_domain, trackers, featurizer, priority
    ):
        policy = self.create_policy(
            featurizer=featurizer, priority=priority, cv=None, C=123
        )
        policy.train(trackers, domain=default_domain)
        assert policy.model.C == 123

    def test_train_with_shuffle_false(
        self, default_domain, trackers, featurizer, priority
    ):
        policy = self.create_policy(
            featurizer=featurizer, priority=priority, shuffle=False
        )
        # does not raise
        policy.train(trackers, domain=default_domain)


class TestEmbeddingPolicyNoAttention(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy(
            priority=priority, attn_before_rnn=False, attn_after_rnn=False
        )
        return p


class TestEmbeddingPolicyAttentionBeforeRNN(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy(
            priority=priority, attn_before_rnn=True, attn_after_rnn=False
        )
        return p


class TestEmbeddingPolicyAttentionAfterRNN(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy(
            priority=priority, attn_before_rnn=False, attn_after_rnn=True
        )
        return p


class TestEmbeddingPolicyAttentionBoth(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy(
            priority=priority, attn_before_rnn=True, attn_after_rnn=True
        )
        return p


class TestEmbeddingPolicyWithTfConfig(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        # use standard featurizer from EmbeddingPolicy,
        # since it is using FullDialogueTrackerFeaturizer
        p = EmbeddingPolicy(priority=priority, **tf_defaults())
        return p

    def test_tf_config(self, trained_policy, tmpdir):
        # noinspection PyProtectedMember
        assert trained_policy.session._config == session_config()
        trained_policy.persist(tmpdir.strpath)
        loaded = trained_policy.__class__.load(tmpdir.strpath)
        # noinspection PyProtectedMember
        assert loaded.session._config == session_config()


class TestFormPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        p = FormPolicy(priority=priority)
        return p

    async def test_memorise(self, trained_policy, default_domain):
        domain = Domain.load("data/test_domains/form.yml")
        trackers = await training.load_data("data/test_stories/stories_form.md", domain)
        trained_policy.train(trackers, domain)

        (
            all_states,
            all_actions,
        ) = trained_policy.featurizer.training_states_and_actions(trackers, domain)

        for tracker, states, actions in zip(trackers, all_states, all_actions):
            for state in states:
                if state is not None:
                    # check that 'form: inform' was ignored
                    assert "intent_inform" not in state.keys()
            recalled = trained_policy.recall(states, tracker, domain)
            active_form = trained_policy._get_active_form_name(states[-1])

            if states[0] is not None and states[-1] is not None:
                # explicitly set intents and actions before listen after
                # which FormPolicy should not predict a form action and
                # should add FormValidation(False) event
                # @formatter:off
                is_no_validation = (
                    (
                        "prev_some_form" in states[0].keys()
                        and "intent_default" in states[-1].keys()
                    )
                    or (
                        "prev_some_form" in states[0].keys()
                        and "intent_stop" in states[-1].keys()
                    )
                    or (
                        "prev_utter_ask_continue" in states[0].keys()
                        and "intent_affirm" in states[-1].keys()
                    )
                    or (
                        "prev_utter_ask_continue" in states[0].keys()
                        and "intent_deny" in states[-1].keys()
                    )
                )
                # @formatter:on
            else:
                is_no_validation = False

            if "intent_start_form" in states[-1]:
                # explicitly check that intent that starts the form
                # is not memorized as non validation intent
                assert recalled is None
            elif is_no_validation:
                assert recalled == active_form
            else:
                assert recalled is None

        nums = np.random.randn(domain.num_states)
        random_states = [{f: num for f, num in zip(domain.input_states, nums)}]
        assert trained_policy.recall(random_states, None, domain) is None


class TestTwoStageFallbackPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self, featurizer, priority):
        p = TwoStageFallbackPolicy(
            priority=priority, deny_suggestion_intent_name="deny"
        )
        return p

    @pytest.fixture(scope="class")
    def default_domain(self):
        content = """
        actions:
          - utter_hello

        intents:
          - greet
          - bye
          - affirm
          - deny
        """
        return Domain.from_yaml(content)

    @staticmethod
    def _get_next_action(policy, events, domain):
        tracker = get_tracker(events)

        scores = policy.predict_action_probabilities(tracker, domain)
        index = scores.index(max(scores))
        return domain.action_names[index]

    @staticmethod
    async def _get_tracker_after_reverts(events, dispatcher, domain):
        tracker = get_tracker(events)
        action = ActionRevertFallbackEvents()
        events += await action.run(dispatcher, tracker, domain)

        return get_tracker(events)

    def test_ask_affirmation(self, trained_policy, default_domain):
        events = [ActionExecuted(ACTION_LISTEN_NAME), user_uttered("Hi", 0.2)]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_DEFAULT_ASK_AFFIRMATION_NAME

    async def test_affirmation(self, default_dispatcher_collecting, default_domain):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 1),
            ActionExecuted("utter_hello"),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 1),
        ]

        tracker = await self._get_tracker_after_reverts(
            events, default_dispatcher_collecting, default_domain
        )

        assert "greet" == tracker.latest_message.parse_data["intent"]["name"]
        assert tracker.export_stories() == (
            "## sender\n* greet\n    - utter_hello\n* greet\n"
        )

    def test_ask_rephrase(self, trained_policy, default_domain):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("deny", 1),
        ]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_DEFAULT_ASK_REPHRASE_NAME

    async def test_successful_rephrasing(
        self, trained_policy, default_dispatcher_collecting, default_domain
    ):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("deny", 1),
            ActionExecuted(ACTION_DEFAULT_ASK_REPHRASE_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("bye", 1),
        ]

        tracker = await self._get_tracker_after_reverts(
            events, default_dispatcher_collecting, default_domain
        )

        assert "bye" == tracker.latest_message.parse_data["intent"]["name"]
        assert tracker.export_stories() == "## sender\n* bye\n"

    def test_affirm_rephrased_intent(self, trained_policy, default_domain):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("deny", 1),
            ActionExecuted(ACTION_DEFAULT_ASK_REPHRASE_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
        ]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_DEFAULT_ASK_AFFIRMATION_NAME

    async def test_affirmed_rephrasing(
        self, trained_policy, default_dispatcher_collecting, default_domain
    ):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("deny", 1),
            ActionExecuted(ACTION_DEFAULT_ASK_REPHRASE_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("bye", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("bye", 1),
        ]

        tracker = await self._get_tracker_after_reverts(
            events, default_dispatcher_collecting, default_domain
        )

        assert "bye" == tracker.latest_message.parse_data["intent"]["name"]
        assert tracker.export_stories() == "## sender\n* bye\n"

    def test_denied_rephrasing_affirmation(self, trained_policy, default_domain):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("deny", 1),
            ActionExecuted(ACTION_DEFAULT_ASK_REPHRASE_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("bye", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("deny", 1),
        ]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_DEFAULT_FALLBACK_NAME

    async def test_rephrasing_instead_affirmation(
        self, trained_policy, default_dispatcher_collecting, default_domain
    ):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 1),
            ActionExecuted("utter_hello"),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("bye", 1),
        ]

        tracker = await self._get_tracker_after_reverts(
            events, default_dispatcher_collecting, default_domain
        )

        assert "bye" == tracker.latest_message.parse_data["intent"]["name"]
        assert tracker.export_stories() == (
            "## sender\n* greet\n    - utter_hello\n* bye\n"
        )

    def test_unknown_instead_affirmation(self, trained_policy, default_domain):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
        ]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_DEFAULT_FALLBACK_NAME

    def test_listen_after_hand_off(self, trained_policy, default_domain):
        events = [ActionExecuted(ACTION_DEFAULT_FALLBACK_NAME)]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_LISTEN_NAME

    def test_exception_if_intent_not_present(self, trained_policy):
        content = """
                actions:
                  - utter_hello

                intents:
                  - greet
                """
        domain = Domain.from_yaml(content)

        events = [ActionExecuted(ACTION_DEFAULT_FALLBACK_NAME)]

        tracker = get_tracker(events)
        with pytest.raises(InvalidDomain):
            trained_policy.predict_action_probabilities(tracker, domain)
