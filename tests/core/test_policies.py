from pathlib import Path
from typing import Type, List, Text, Tuple, Optional, Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.generator import TrackerWithCachedStates

from rasa.core import training
import rasa.core.actions.action
from rasa.shared.constants import DEFAULT_SENDER_ID
from rasa.shared.core.training_data.story_writer.markdown_story_writer import (
    MarkdownStoryWriter,
)
from rasa.shared.nlu.constants import ACTION_NAME, INTENT_NAME_KEY
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_BACK_NAME,
    PREVIOUS_ACTION,
    USER,
)
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.events import ActionExecuted, ConversationPaused, Event
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
)
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.policy import SupportedData, Policy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from rasa.core.policies.sklearn_policy import SklearnPolicy
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.formats.markdown import INTENT
from rasa.utils.tensorflow.constants import (
    SIMILARITY_TYPE,
    RANKING_LENGTH,
    LOSS_TYPE,
    SCALE_LOSS,
    EVAL_NUM_EXAMPLES,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
)
from rasa.train import train_core
from rasa.utils import train_utils
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH_WITH_MAPPING,
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_STORIES_FILE,
)
from tests.core.utilities import get_tracker, read_dialogue_file, user_uttered


async def train_trackers(
    domain: Domain, augmentation_factor: int = 20
) -> List[TrackerWithCachedStates]:
    return await training.load_data(
        DEFAULT_STORIES_FILE, domain, augmentation_factor=augmentation_factor
    )


# We are going to use class style testing here since unfortunately pytest
# doesn't support using fixtures as arguments to its own parameterize yet
# (hence, we can't train a policy, declare it as a fixture and use the
# different fixtures of the different policies for the functional tests).
# Therefore, we are going to reverse this and train the policy within a class
# and collect the tests in a base class.
# noinspection PyMethodMayBeStatic
class PolicyTestCollection:
    """Tests every policy needs to fulfill.

    Each policy can declare further tests on its own."""

    max_history = 3  # this is the amount of history we test on

    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: Optional[int]
    ) -> Policy:
        raise NotImplementedError

    @pytest.fixture(scope="module")
    def featurizer(self) -> TrackerFeaturizer:
        featurizer = MaxHistoryTrackerFeaturizer(
            SingleStateFeaturizer(), max_history=self.max_history
        )
        return featurizer

    @pytest.fixture(scope="module")
    def priority(self) -> int:
        return 1

    @pytest.fixture(scope="module")
    def default_domain(self) -> Domain:
        return Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)

    @pytest.fixture(scope="module")
    def tracker(self, default_domain: Domain) -> DialogueStateTracker:
        return DialogueStateTracker(DEFAULT_SENDER_ID, default_domain.slots)

    @pytest.fixture(scope="module")
    async def trained_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        default_domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
        policy = self.create_policy(featurizer, priority)
        training_trackers = await train_trackers(default_domain, augmentation_factor=20)
        policy.train(training_trackers, default_domain, RegexInterpreter())
        return policy

    def test_featurizer(self, trained_policy: Policy, tmp_path: Path):
        assert isinstance(trained_policy.featurizer, MaxHistoryTrackerFeaturizer)
        assert trained_policy.featurizer.max_history == self.max_history
        assert isinstance(
            trained_policy.featurizer.state_featurizer, SingleStateFeaturizer
        )
        trained_policy.persist(str(tmp_path))
        loaded = trained_policy.__class__.load(str(tmp_path))
        assert isinstance(loaded.featurizer, MaxHistoryTrackerFeaturizer)
        assert loaded.featurizer.max_history == self.max_history
        assert isinstance(loaded.featurizer.state_featurizer, SingleStateFeaturizer)

    async def test_persist_and_load(
        self, trained_policy: Policy, default_domain: Domain, tmp_path: Path
    ):
        trained_policy.persist(str(tmp_path))
        loaded = trained_policy.__class__.load(str(tmp_path))
        trackers = await train_trackers(default_domain, augmentation_factor=20)

        for tracker in trackers:
            predicted_probabilities = loaded.predict_action_probabilities(
                tracker, default_domain, RegexInterpreter()
            )
            actual_probabilities = trained_policy.predict_action_probabilities(
                tracker, default_domain, RegexInterpreter()
            )
            assert predicted_probabilities == actual_probabilities

    def test_prediction_on_empty_tracker(
        self, trained_policy: Policy, default_domain: Domain
    ):
        tracker = DialogueStateTracker(DEFAULT_SENDER_ID, default_domain.slots)
        probabilities = trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        ).probabilities
        assert len(probabilities) == default_domain.num_actions
        assert max(probabilities) <= 1.0
        assert min(probabilities) >= 0.0

    @pytest.mark.filterwarnings(
        "ignore:.*without a trained model present.*:UserWarning"
    )
    def test_persist_and_load_empty_policy(self, tmp_path: Path):
        empty_policy = self.create_policy(None, None)
        empty_policy.persist(str(tmp_path))
        loaded = empty_policy.__class__.load(str(tmp_path))
        assert loaded is not None

    @staticmethod
    def _get_next_action(policy: Policy, events: List[Event], domain: Domain) -> Text:
        tracker = get_tracker(events)

        scores = policy.predict_action_probabilities(
            tracker, domain, RegexInterpreter()
        ).probabilities
        index = scores.index(max(scores))
        return domain.action_names[index]


class TestSklearnPolicy(PolicyTestCollection):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int, **kwargs: Any
    ) -> SklearnPolicy:
        return SklearnPolicy(featurizer, priority, **kwargs)

    @pytest.yield_fixture
    def mock_search(self):
        with patch("rasa.core.policies.sklearn_policy.GridSearchCV") as gs:
            gs.best_estimator_ = "mockmodel"
            gs.best_score_ = 0.123
            gs.return_value = gs  # for __init__
            yield gs

    @pytest.fixture(scope="module")
    async def trackers(self, default_domain: Domain) -> List[TrackerWithCachedStates]:
        return await train_trackers(default_domain, augmentation_factor=20)

    def test_additional_train_args_do_not_raise(
        self,
        default_domain: Domain,
        trackers: List[TrackerWithCachedStates],
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
    ):
        policy = self.create_policy(featurizer=featurizer, priority=priority, cv=None)
        policy.train(
            trackers,
            domain=default_domain,
            interpreter=RegexInterpreter(),
            this_is_not_a_feature=True,
        )

    def test_cv_none_does_not_trigger_search(
        self,
        mock_search,
        default_domain: Domain,
        trackers: List[TrackerWithCachedStates],
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
    ):
        policy = self.create_policy(featurizer=featurizer, priority=priority, cv=None)
        policy.train(trackers, domain=default_domain, interpreter=RegexInterpreter())

        assert mock_search.call_count == 0
        assert policy.model != "mockmodel"

    def test_cv_not_none_param_grid_none_triggers_search_without_params(
        self,
        mock_search,
        default_domain: Domain,
        trackers: List[TrackerWithCachedStates],
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
    ):

        policy = self.create_policy(featurizer=featurizer, priority=priority, cv=3)
        policy.train(trackers, domain=default_domain, interpreter=RegexInterpreter())

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]["cv"] == 3
        assert mock_search.call_args_list[0][1]["param_grid"] == {}
        assert policy.model == "mockmodel"

    def test_cv_not_none_param_grid_none_triggers_search_with_params(
        self,
        mock_search,
        default_domain: Domain,
        trackers: List[TrackerWithCachedStates],
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
    ):
        param_grid = {"n_estimators": 50}
        policy = self.create_policy(
            featurizer=featurizer, priority=priority, cv=3, param_grid=param_grid
        )
        policy.train(trackers, domain=default_domain, interpreter=RegexInterpreter())

        assert mock_search.call_count > 0
        assert mock_search.call_args_list[0][1]["cv"] == 3
        assert mock_search.call_args_list[0][1]["param_grid"] == param_grid
        assert policy.model == "mockmodel"

    def test_missing_classes_filled_correctly(
        self,
        default_domain: Domain,
        trackers: List[TrackerWithCachedStates],
        tracker: DialogueStateTracker,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
    ):
        # Pretend that a couple of classes are missing and check that
        # those classes are predicted as 0, while the other class
        # probabilities are predicted normally.
        policy = self.create_policy(featurizer=featurizer, priority=priority, cv=None)

        classes = [1, 3]
        new_trackers = []
        for tr in trackers:
            new_tracker = DialogueStateTracker(DEFAULT_SENDER_ID, default_domain.slots)
            for e in tr.applied_events():
                if isinstance(e, ActionExecuted):
                    new_action = rasa.core.actions.action.action_for_index(
                        np.random.choice(classes), default_domain, action_endpoint=None
                    ).name()
                    new_tracker.update(ActionExecuted(new_action))
                else:
                    new_tracker.update(e)

            new_trackers.append(new_tracker)

        policy.train(
            new_trackers, domain=default_domain, interpreter=RegexInterpreter()
        )
        predicted_probabilities = policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        ).probabilities

        assert len(predicted_probabilities) == default_domain.num_actions
        assert np.allclose(sum(predicted_probabilities), 1.0)
        for i, prob in enumerate(predicted_probabilities):
            if i in classes:
                assert prob >= 0.0
            else:
                assert prob == 0.0

    def test_train_kwargs_are_set_on_model(
        self,
        default_domain: Domain,
        trackers: List[TrackerWithCachedStates],
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
    ):
        policy = self.create_policy(
            featurizer=featurizer, priority=priority, cv=None, C=123
        )
        policy.train(trackers, domain=default_domain, interpreter=RegexInterpreter())
        assert policy.model.C == 123

    def test_train_with_shuffle_false(
        self,
        default_domain: Domain,
        trackers: List[TrackerWithCachedStates],
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
    ):
        policy = self.create_policy(
            featurizer=featurizer, priority=priority, shuffle=False
        )
        # does not raise
        policy.train(trackers, domain=default_domain, interpreter=RegexInterpreter())


class TestTEDPolicy(PolicyTestCollection):
    def test_train_model_checkpointing(self, tmp_path: Path):
        model_name = "core-checkpointed-model"
        best_model_file = tmp_path / (model_name + ".tar.gz")
        assert not best_model_file.exists()

        train_core(
            domain="data/test_domains/default.yml",
            stories="data/test_stories/stories_defaultdomain.md",
            output=str(tmp_path),
            fixed_model_name=model_name,
            config="data/test_config/config_ted_policy_model_checkpointing.yml",
        )

        assert best_model_file.exists()

    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(featurizer=featurizer, priority=priority)

    def test_similarity_type(self, trained_policy: TEDPolicy):
        assert trained_policy.config[SIMILARITY_TYPE] == "inner"

    def test_ranking_length(self, trained_policy: TEDPolicy):
        assert trained_policy.config[RANKING_LENGTH] == 10

    def test_normalization(
        self,
        trained_policy: TEDPolicy,
        tracker: DialogueStateTracker,
        default_domain: Domain,
        monkeypatch: MonkeyPatch,
    ):
        # first check the output is what we expect
        predicted_probabilities = trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        ).probabilities
        # count number of non-zero confidences
        assert (
            sum([confidence > 0 for confidence in predicted_probabilities])
            == trained_policy.config[RANKING_LENGTH]
        )
        # check that the norm is still 1
        assert sum(predicted_probabilities) == pytest.approx(1)

        # also check our function is called
        mock = Mock()
        monkeypatch.setattr(train_utils, "normalize", mock.normalize)
        trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )

        mock.normalize.assert_called_once()

    async def test_gen_batch(self, trained_policy: TEDPolicy, default_domain: Domain):
        training_trackers = await train_trackers(default_domain, augmentation_factor=0)
        interpreter = RegexInterpreter()
        training_data, label_ids = trained_policy.featurize_for_training(
            training_trackers, default_domain, interpreter
        )
        label_data, all_labels = trained_policy._create_label_data(
            default_domain, interpreter
        )
        model_data = trained_policy._create_model_data(
            training_data, label_ids, all_labels
        )
        batch_size = 2
        (
            batch_label_ids,
            batch_entities_mask,
            batch_entities_sentence_1,
            batch_entities_sentence_2,
            batch_entities_sentence_3,
            batch_intent_mask,
            batch_intent_sentence_1,
            batch_intent_sentence_2,
            batch_intent_sentence_3,
            batch_slots_mask,
            batch_slots_sentence_1,
            batch_slots_sentence_2,
            batch_slots_sentence_3,
            batch_action_name_mask,
            batch_action_name_sentence_1,
            batch_action_name_sentence_2,
            batch_action_name_sentence_3,
            batch_dialogue_length,
        ) = next(model_data._gen_batch(batch_size=batch_size))

        assert (
            batch_intent_mask.shape[0] == batch_size
            and batch_action_name_mask.shape[0] == batch_size
            and batch_entities_mask.shape[0] == batch_size
            and batch_slots_mask.shape[0] == batch_size
        )
        assert (
            batch_intent_sentence_3[1]
            == batch_action_name_sentence_3[1]
            == batch_entities_sentence_3[1]
            == batch_slots_sentence_3[1]
        )

        (
            batch_label_ids,
            batch_entities_mask,
            batch_entities_sentence_1,
            batch_entities_sentence_2,
            batch_entities_sentence_3,
            batch_intent_mask,
            batch_intent_sentence_1,
            batch_intent_sentence_2,
            batch_intent_sentence_3,
            batch_slots_mask,
            batch_slots_sentence_1,
            batch_slots_sentence_2,
            batch_slots_sentence_3,
            batch_action_name_mask,
            batch_action_name_sentence_1,
            batch_action_name_sentence_2,
            batch_action_name_sentence_3,
            batch_dialogue_length,
        ) = next(
            model_data._gen_batch(
                batch_size=batch_size, batch_strategy="balanced", shuffle=True
            )
        )

        assert (
            batch_intent_mask.shape[0] == batch_size
            and batch_action_name_mask.shape[0] == batch_size
            and batch_entities_mask.shape[0] == batch_size
            and batch_slots_mask.shape[0] == batch_size
        )
        assert (
            batch_intent_sentence_3[1]
            == batch_action_name_sentence_3[1]
            == batch_entities_sentence_3[1]
            == batch_slots_sentence_3[1]
        )


class TestTEDPolicyMargin(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer, priority=priority, **{LOSS_TYPE: "margin"}
        )

    def test_similarity_type(self, trained_policy: TEDPolicy):
        assert trained_policy.config[SIMILARITY_TYPE] == "cosine"

    def test_normalization(
        self,
        trained_policy: Policy,
        tracker: DialogueStateTracker,
        default_domain: Domain,
        monkeypatch: MonkeyPatch,
    ):
        # Mock actual normalization method
        mock = Mock()
        monkeypatch.setattr(train_utils, "normalize", mock.normalize)
        trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )

        # function should not get called for margin loss_type
        mock.normalize.assert_not_called()


class TestTEDPolicyWithEval(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            priority=priority,
            **{SCALE_LOSS: False, EVAL_NUM_EXAMPLES: 4},
        )


class TestTEDPolicyNoNormalization(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer, priority=priority, **{RANKING_LENGTH: 0}
        )

    def test_ranking_length(self, trained_policy: TEDPolicy):
        assert trained_policy.config[RANKING_LENGTH] == 0

    def test_normalization(
        self,
        trained_policy: Policy,
        tracker: DialogueStateTracker,
        default_domain: Domain,
        monkeypatch: MonkeyPatch,
    ):
        # first check the output is what we expect
        predicted_probabilities = trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        ).probabilities
        # there should be no normalization
        assert all([confidence > 0 for confidence in predicted_probabilities])

        # also check our function is not called
        mock = Mock()
        monkeypatch.setattr(train_utils, "normalize", mock.normalize)
        trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )

        mock.normalize.assert_not_called()


class TestTEDPolicyLowRankingLength(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer, priority=priority, **{RANKING_LENGTH: 3}
        )

    def test_ranking_length(self, trained_policy: TEDPolicy):
        assert trained_policy.config[RANKING_LENGTH] == 3


class TestTEDPolicyHighRankingLength(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer, priority=priority, **{RANKING_LENGTH: 11}
        )

    def test_ranking_length(self, trained_policy: TEDPolicy):
        assert trained_policy.config[RANKING_LENGTH] == 11


class TestTEDPolicyWithStandardFeaturizer(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        # use standard featurizer from TEDPolicy,
        # since it is using MaxHistoryTrackerFeaturizer
        # if max_history is not specified
        return TEDPolicy(priority=priority)

    def test_featurizer(self, trained_policy: Policy, tmp_path: Path):
        assert isinstance(trained_policy.featurizer, MaxHistoryTrackerFeaturizer)
        assert isinstance(
            trained_policy.featurizer.state_featurizer, SingleStateFeaturizer
        )
        trained_policy.persist(str(tmp_path))
        loaded = trained_policy.__class__.load(str(tmp_path))
        assert isinstance(loaded.featurizer, MaxHistoryTrackerFeaturizer)
        assert isinstance(loaded.featurizer.state_featurizer, SingleStateFeaturizer)


class TestTEDPolicyWithMaxHistory(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        # use standard featurizer from TEDPolicy,
        # since it is using MaxHistoryTrackerFeaturizer
        # if max_history is specified
        return TEDPolicy(priority=priority, max_history=self.max_history)

    def test_featurizer(self, trained_policy: Policy, tmp_path: Path):
        assert isinstance(trained_policy.featurizer, MaxHistoryTrackerFeaturizer)
        assert trained_policy.featurizer.max_history == self.max_history
        assert isinstance(
            trained_policy.featurizer.state_featurizer, SingleStateFeaturizer
        )
        trained_policy.persist(str(tmp_path))
        loaded = trained_policy.__class__.load(str(tmp_path))
        assert isinstance(loaded.featurizer, MaxHistoryTrackerFeaturizer)
        assert loaded.featurizer.max_history == self.max_history
        assert isinstance(loaded.featurizer.state_featurizer, SingleStateFeaturizer)


class TestTEDPolicyWithRelativeAttention(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            priority=priority,
            **{
                KEY_RELATIVE_ATTENTION: True,
                VALUE_RELATIVE_ATTENTION: True,
                MAX_RELATIVE_POSITION: 5,
            },
        )


class TestTEDPolicyWithRelativeAttentionMaxHistoryOne(TestTEDPolicy):

    max_history = 1

    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            priority=priority,
            **{
                KEY_RELATIVE_ATTENTION: True,
                VALUE_RELATIVE_ATTENTION: True,
                MAX_RELATIVE_POSITION: 5,
            },
        )


class TestMemoizationPolicy(PolicyTestCollection):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        max_history = None
        if isinstance(featurizer, MaxHistoryTrackerFeaturizer):
            max_history = featurizer.max_history
        return MemoizationPolicy(priority=priority, max_history=max_history)

    def test_featurizer(self, trained_policy: Policy, tmp_path: Path):
        assert isinstance(trained_policy.featurizer, MaxHistoryTrackerFeaturizer)
        assert trained_policy.featurizer.state_featurizer is None
        trained_policy.persist(str(tmp_path))
        loaded = trained_policy.__class__.load(str(tmp_path))
        assert isinstance(loaded.featurizer, MaxHistoryTrackerFeaturizer)
        assert loaded.featurizer.state_featurizer is None

    async def test_memorise(
        self, trained_policy: MemoizationPolicy, default_domain: Domain
    ):
        trackers = await train_trackers(default_domain, augmentation_factor=20)
        trained_policy.train(trackers, default_domain, RegexInterpreter())
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
            assert recalled == actions[0]

        nums = np.random.randn(default_domain.num_states)
        random_states = [{f: num for f, num in zip(default_domain.input_states, nums)}]
        assert trained_policy._recall_states(random_states) is None

        # compare augmentation for augmentation_factor of 0 and 20:
        trackers_no_augmentation = await train_trackers(
            default_domain, augmentation_factor=0
        )
        trained_policy.train(
            trackers_no_augmentation, default_domain, RegexInterpreter()
        )
        lookup_no_augmentation = trained_policy.lookup

        assert lookup_no_augmentation == lookup_with_augmentation

    def test_memorise_with_nlu(
        self, trained_policy: MemoizationPolicy, default_domain: Domain
    ):
        filename = "data/test_dialogues/default.json"
        dialogue = read_dialogue_file(filename)

        tracker = DialogueStateTracker(dialogue.name, default_domain.slots)
        tracker.recreate_from_dialogue(dialogue)
        states = trained_policy.featurizer.prediction_states([tracker], default_domain)[
            0
        ]

        recalled = trained_policy.recall(states, tracker, default_domain)
        assert recalled is not None


class TestAugmentedMemoizationPolicy(TestMemoizationPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        max_history = None
        if isinstance(featurizer, MaxHistoryTrackerFeaturizer):
            max_history = featurizer.max_history
        return AugmentedMemoizationPolicy(priority=priority, max_history=max_history)


class TestFormPolicy(TestMemoizationPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return FormPolicy(priority=priority)

    def _test_for_previous_action_and_intent(
        self, states: List[State], intent: Text, action_name: Text
    ) -> bool:
        previous_action_as_expected = False
        intent_as_expected = False
        if states[0].get(PREVIOUS_ACTION):
            previous_action_as_expected = (
                states[0].get(PREVIOUS_ACTION).get(ACTION_NAME) == action_name
            )
        if states[-1].get(USER):
            intent_as_expected = states[-1].get(USER).get(INTENT) == intent
        return previous_action_as_expected and intent_as_expected

    async def test_memorise(self, trained_policy: FormPolicy, default_domain: Domain):
        domain = Domain.load("data/test_domains/form.yml")
        trackers = await training.load_data("data/test_stories/stories_form.md", domain)
        trained_policy.train(trackers, domain, RegexInterpreter())

        (
            all_states,
            all_actions,
        ) = trained_policy.featurizer.training_states_and_actions(trackers, domain)

        for tracker, states, actions in zip(trackers, all_states, all_actions):
            for state in states:
                if state is not None:
                    # check that 'form: inform' was ignored
                    if state.get(USER):
                        assert not state.get(USER).get(INTENT) == "inform"
            recalled = trained_policy.recall(states, tracker, domain)
            active_form = trained_policy._get_active_form_name(states[-1])

            if states[0] is not None and states[-1] is not None:
                # explicitly set intents and actions before listen after
                # which FormPolicy should not predict a form action and
                # should add FormValidation(False) event
                # @formatter:off
                is_no_validation = (
                    self._test_for_previous_action_and_intent(
                        states, "default", "some_form"
                    )
                    or self._test_for_previous_action_and_intent(
                        states, "stop", "some_form"
                    )
                    or self._test_for_previous_action_and_intent(
                        states, "affirm", "utter_ask_continue"
                    )
                    or self._test_for_previous_action_and_intent(
                        states, "deny", "utter_ask_continue"
                    )
                    # comes from the fact that intent_inform after utter_ask_continue
                    # is not read from stories
                    or self._test_for_previous_action_and_intent(
                        states, "stop", "utter_ask_continue"
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

    def test_memorise_with_nlu(self, trained_policy, default_domain):
        pass


class TestMappingPolicy(PolicyTestCollection):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return MappingPolicy()

    def test_featurizer(self, trained_policy: Policy, tmp_path: Path):
        assert trained_policy.featurizer is None
        trained_policy.persist(str(tmp_path))
        loaded = trained_policy.__class__.load(str(tmp_path))
        assert loaded.featurizer is None

    @pytest.fixture(scope="module")
    def domain_with_mapping(self) -> Domain:
        return Domain.load(DEFAULT_DOMAIN_PATH_WITH_MAPPING)

    @pytest.fixture
    def tracker(self, domain_with_mapping: Domain) -> DialogueStateTracker:
        return DialogueStateTracker(DEFAULT_SENDER_ID, domain_with_mapping.slots)

    @pytest.fixture(
        params=[
            ("default", "utter_default"),
            ("greet", "utter_greet"),
            (USER_INTENT_RESTART, ACTION_RESTART_NAME),
            (USER_INTENT_BACK, ACTION_BACK_NAME),
        ]
    )
    def intent_mapping(self, request) -> Tuple[Text, Text]:
        return request.param

    def test_predict_mapped_action(
        self,
        priority: int,
        domain_with_mapping: Domain,
        intent_mapping: Tuple[Text, Text],
    ):
        policy = self.create_policy(None, priority)
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered(intent_mapping[0], 1),
        ]

        assert (
            self._get_next_action(policy, events, domain_with_mapping)
            == intent_mapping[1]
        )

    def test_restart_if_paused(self, priority: int, domain_with_mapping: Domain):
        policy = self.create_policy(None, priority)
        events = [ConversationPaused(), user_uttered(USER_INTENT_RESTART, 1)]

        assert (
            self._get_next_action(policy, events, domain_with_mapping)
            == ACTION_RESTART_NAME
        )

    def test_predict_action_listen(
        self,
        priority: int,
        domain_with_mapping: Domain,
        intent_mapping: Tuple[Text, Text],
    ):
        policy = self.create_policy(None, priority)
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered(intent_mapping[0], 1),
            ActionExecuted(intent_mapping[1], policy="policy_0_MappingPolicy"),
        ]
        tracker = get_tracker(events)
        scores = policy.predict_action_probabilities(
            tracker, domain_with_mapping, RegexInterpreter()
        ).probabilities
        index = scores.index(max(scores))
        action_planned = domain_with_mapping.action_names[index]
        assert action_planned == ACTION_LISTEN_NAME
        assert scores != [0] * domain_with_mapping.num_actions

    def test_do_not_follow_other_policy(
        self,
        priority: int,
        domain_with_mapping: Domain,
        intent_mapping: Tuple[Text, Text],
    ):
        policy = self.create_policy(None, priority)
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered(intent_mapping[0], 1),
            ActionExecuted(intent_mapping[1], policy="other_policy"),
        ]
        tracker = get_tracker(events)
        scores = policy.predict_action_probabilities(
            tracker, domain_with_mapping, RegexInterpreter()
        ).probabilities
        assert scores == [0] * domain_with_mapping.num_actions


class TestFallbackPolicy(PolicyTestCollection):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return FallbackPolicy(priority=priority)

    def test_featurizer(self, trained_policy: Policy, tmp_path: Path):
        assert trained_policy.featurizer is None
        trained_policy.persist(str(tmp_path))
        loaded = trained_policy.__class__.load(str(tmp_path))
        assert loaded.featurizer is None

    @pytest.mark.parametrize(
        "top_confidence, all_confidences, last_action_name, should_nlu_fallback",
        [
            (0.1, [0.1], "some_action", False),
            (0.1, [0.1], "action_listen", True),
            (0.9, [0.9, 0.1], "some_action", False),
            (0.9, [0.9, 0.1], "action_listen", False),
            (0.4, [0.4, 0.35], "some_action", False),
            (0.4, [0.4, 0.35], "action_listen", True),
            (0.9, [0.9, 0.85], "action_listen", True),
        ],
    )
    def test_should_nlu_fallback(
        self,
        trained_policy: TwoStageFallbackPolicy,
        top_confidence: float,
        all_confidences: List[float],
        last_action_name: Text,
        should_nlu_fallback: bool,
    ):
        nlu_data = {
            "intent": {"confidence": top_confidence},
            "intent_ranking": [
                {"confidence": confidence} for confidence in all_confidences
            ],
        }
        assert (
            trained_policy.should_nlu_fallback(nlu_data, last_action_name)
            is should_nlu_fallback
        )


class TestTwoStageFallbackPolicy(TestFallbackPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TwoStageFallbackPolicy(
            priority=priority, deny_suggestion_intent_name="deny"
        )

    @pytest.fixture(scope="class")
    def default_domain(self) -> Domain:
        content = """
        intents:
          - greet
          - bye
          - affirm
          - deny
        """
        return Domain.from_yaml(content)

    @staticmethod
    async def _get_tracker_after_reverts(
        events: List[Event],
        channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        domain: Domain,
    ) -> DialogueStateTracker:
        tracker = get_tracker(events)
        action = rasa.core.actions.action.ActionRevertFallbackEvents()
        events += await action.run(channel, nlg, tracker, domain)

        return get_tracker(events)

    def test_ask_affirmation(self, trained_policy: Policy, default_domain: Domain):
        events = [ActionExecuted(ACTION_LISTEN_NAME), user_uttered("Hi", 0.2)]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_DEFAULT_ASK_AFFIRMATION_NAME

    async def test_affirmation(
        self,
        default_channel: OutputChannel,
        default_nlg: NaturalLanguageGenerator,
        default_domain: Domain,
    ):
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
            events, default_channel, default_nlg, default_domain
        )

        assert "greet" == tracker.latest_message.parse_data["intent"][INTENT_NAME_KEY]
        assert tracker.export_stories(MarkdownStoryWriter()) == (
            "## sender\n* greet\n    - utter_hello\n* greet\n"
        )

    def test_ask_rephrase(self, trained_policy: Policy, default_domain: Domain):
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
        self,
        default_channel: OutputChannel,
        default_nlg: NaturalLanguageGenerator,
        default_domain: Domain,
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
            events, default_channel, default_nlg, default_domain
        )

        assert "bye" == tracker.latest_message.parse_data["intent"][INTENT_NAME_KEY]
        assert tracker.export_stories(MarkdownStoryWriter()) == "## sender\n* bye\n"

    def test_affirm_rephrased_intent(
        self, trained_policy: Policy, default_domain: Domain
    ):
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
        self,
        default_channel: OutputChannel,
        default_nlg: NaturalLanguageGenerator,
        default_domain: Domain,
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
            events, default_channel, default_nlg, default_domain
        )

        assert "bye" == tracker.latest_message.parse_data["intent"][INTENT_NAME_KEY]
        assert tracker.export_stories(MarkdownStoryWriter()) == "## sender\n* bye\n"

    def test_denied_rephrasing_affirmation(
        self, trained_policy: Policy, default_domain: Domain
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
            user_uttered("deny", 1),
        ]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_DEFAULT_FALLBACK_NAME

    async def test_rephrasing_instead_affirmation(
        self,
        default_channel: OutputChannel,
        default_nlg: NaturalLanguageGenerator,
        default_domain: Domain,
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
            events, default_channel, default_nlg, default_domain
        )

        assert "bye" == tracker.latest_message.parse_data["intent"][INTENT_NAME_KEY]
        assert tracker.export_stories(MarkdownStoryWriter()) == (
            "## sender\n* greet\n    - utter_hello\n* bye\n"
        )

    def test_unknown_instead_affirmation(
        self, trained_policy: Policy, default_domain: Domain
    ):
        events = [
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
            ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            user_uttered("greet", 0.2),
        ]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_DEFAULT_FALLBACK_NAME

    def test_listen_after_hand_off(
        self, trained_policy: Policy, default_domain: Domain
    ):
        events = [ActionExecuted(ACTION_DEFAULT_FALLBACK_NAME)]

        next_action = self._get_next_action(trained_policy, events, default_domain)

        assert next_action == ACTION_LISTEN_NAME


@pytest.mark.parametrize(
    "policy,supported_data",
    [
        (TEDPolicy, SupportedData.ML_DATA),
        (RulePolicy, SupportedData.ML_AND_RULE_DATA),
        (MemoizationPolicy, SupportedData.ML_DATA),
    ],
)
def test_supported_data(policy: Type[Policy], supported_data: SupportedData):
    assert policy.supported_data() == supported_data


class OnlyRulePolicy(Policy):
    """Test policy that supports only rule-based training data."""

    @staticmethod
    def supported_data() -> SupportedData:
        return SupportedData.RULE_DATA


@pytest.mark.parametrize(
    "policy,n_rule_trackers,n_ml_trackers",
    [
        (TEDPolicy(), 0, 3),
        (RulePolicy(), 2, 3),
        (OnlyRulePolicy, 2, 0),  # policy can be passed as a `type` as well
    ],
)
def test_get_training_trackers_for_policy(
    policy: Policy, n_rule_trackers: int, n_ml_trackers: int
):
    # create five trackers (two rule-based and three ML trackers)
    trackers = [
        DialogueStateTracker("id1", slots=[], is_rule_tracker=True),
        DialogueStateTracker("id2", slots=[], is_rule_tracker=False),
        DialogueStateTracker("id3", slots=[], is_rule_tracker=False),
        DialogueStateTracker("id4", slots=[], is_rule_tracker=True),
        DialogueStateTracker("id5", slots=[], is_rule_tracker=False),
    ]

    trackers = SupportedData.trackers_for_policy(policy, trackers)

    rule_trackers = [tracker for tracker in trackers if tracker.is_rule_tracker]
    ml_trackers = [tracker for tracker in trackers if not tracker.is_rule_tracker]

    assert len(rule_trackers) == n_rule_trackers
    assert len(ml_trackers) == n_ml_trackers


@pytest.mark.parametrize(
    "policy", [FormPolicy, MappingPolicy, FallbackPolicy, TwoStageFallbackPolicy]
)
def test_deprecation_warnings_for_old_rule_like_policies(policy: Type[Policy]):
    with pytest.warns(FutureWarning):
        policy(None)
