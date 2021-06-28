from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import numpy as np
import pytest
import tests.core.test_policies
from _pytest.monkeypatch import MonkeyPatch
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
)
from rasa.core.policies.policy import Policy
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
)
from rasa.shared.exceptions import RasaException, InvalidConfigException
from rasa.utils.tensorflow.data_generator import RasaBatchDataGenerator
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.model_training import train_core
from rasa.utils import train_utils
from rasa.utils.tensorflow.constants import (
    EVAL_NUM_EXAMPLES,
    KEY_RELATIVE_ATTENTION,
    LOSS_TYPE,
    MAX_RELATIVE_POSITION,
    RANKING_LENGTH,
    SCALE_LOSS,
    SIMILARITY_TYPE,
    VALUE_RELATIVE_ATTENTION,
    MODEL_CONFIDENCE,
    COSINE,
    AUTO,
    LINEAR_NORM,
    EVAL_NUM_EPOCHS,
)
from tests.core.test_policies import PolicyTestCollection
from rasa.shared.constants import DEFAULT_SENDER_ID, DEFAULT_CORE_SUBDIRECTORY_NAME

UTTER_GREET_ACTION = "utter_greet"
GREET_INTENT_NAME = "greet"
DOMAIN_YAML = f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
"""


def get_checkpoint_dir_path(train_path: Path, ted_pos: Optional[int] = 0) -> Path:
    """
    Produce the path of the checkpoint directory for TED.

    This is very tightly coupled to the persist methods of PolicyEnsemble, Agent, and
    TEDPolicy.
    Args:
        train_path: the path passed to model_training.train_core for training output.
        ted_pos: the position of TED in the policies listed in the config.
    """
    policy_dir_name = Path("policy_{}_{}".format(ted_pos, TEDPolicy.__name__))
    policy_path = train_path / DEFAULT_CORE_SUBDIRECTORY_NAME / policy_dir_name
    return policy_path / "checkpoints"


def test_diagnostics():
    domain = Domain.from_yaml(DOMAIN_YAML)
    policy = TEDPolicy()
    GREET_RULE = DialogueStateTracker.from_events(
        "greet rule",
        evts=[
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    policy.train([GREET_RULE], domain, RegexInterpreter())
    prediction = policy.predict_action_probabilities(
        GREET_RULE, domain, RegexInterpreter()
    )

    assert prediction.diagnostic_data
    assert "attention_weights" in prediction.diagnostic_data
    assert isinstance(prediction.diagnostic_data.get("attention_weights"), np.ndarray)


class TestTEDPolicy(PolicyTestCollection):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> TEDPolicy:
        return TEDPolicy(featurizer=featurizer, priority=priority)

    def test_train_model_checkpointing(self, tmp_path: Path):
        checkpoint_dir = get_checkpoint_dir_path(tmp_path)
        assert not checkpoint_dir.is_dir()

        train_core(
            domain="data/test_domains/default.yml",
            stories="data/test_yaml_stories/stories_defaultdomain.yml",
            train_path=str(tmp_path),
            output=str(tmp_path),
            config="data/test_config/config_ted_policy_model_checkpointing.yml",
        )
        assert checkpoint_dir.is_dir()

    def test_doesnt_checkpoint_with_no_checkpointing(self, tmp_path: Path):
        checkpoint_dir = get_checkpoint_dir_path(tmp_path)
        assert not checkpoint_dir.is_dir()

        train_core(
            domain="data/test_domains/default.yml",
            stories="data/test_yaml_stories/stories_defaultdomain.yml",
            train_path=str(tmp_path),
            output=str(tmp_path),
            config="data/test_config/config_ted_policy_no_model_checkpointing.yml",
        )
        assert not checkpoint_dir.is_dir()

    def test_doesnt_checkpoint_with_zero_eval_num_examples(self, tmp_path: Path):
        checkpoint_dir = get_checkpoint_dir_path(tmp_path)
        assert not checkpoint_dir.is_dir()
        config_file = "config_ted_policy_model_checkpointing_zero_eval_num_examples.yml"
        with pytest.warns(UserWarning) as warning:
            train_core(
                domain="data/test_domains/default.yml",
                stories="data/test_yaml_stories/stories_defaultdomain.yml",
                train_path=str(tmp_path),
                output=str(tmp_path),
                config=f"data/test_config/{config_file}",
            )
        warn_text = (
            f"You have opted to save the best model, but the value of "
            f"'{EVAL_NUM_EXAMPLES}' is not greater than 0. No checkpoint model will be "
            f"saved."
        )
        assert not checkpoint_dir.is_dir()
        assert len([w for w in warning if warn_text in str(w.message)]) == 1

    def test_train_fails_with_checkpoint_zero_eval_num_epochs(self, tmp_path: Path):
        checkpoint_dir = get_checkpoint_dir_path(tmp_path)
        assert not checkpoint_dir.is_dir()
        config_file = "config_ted_policy_model_checkpointing_zero_every_num_epochs.yml"
        with pytest.raises(InvalidConfigException):
            with pytest.warns(UserWarning) as warning:
                train_core(
                    domain="data/test_domains/default.yml",
                    stories="data/test_yaml_stories/stories_defaultdomain.yml",
                    train_path=str(tmp_path),
                    output=str(tmp_path),
                    config=f"data/test_config/{config_file}",
                )
        warn_text = (
            f"You have opted to save the best model, but the value of "
            f"'{EVAL_NUM_EPOCHS}' is not -1 or greater than 0. Training will fail."
        )
        assert len([w for w in warning if warn_text in str(w.message)]) == 1
        assert not checkpoint_dir.is_dir()

    async def test_raise_rasa_exception_no_user_features(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        default_domain: Domain,
        tmp_path: Path,
    ):
        stories = tmp_path / "stories.yml"
        stories.write_text(
            """
            version: "2.0"
            stories:
            - story: test path
              steps:
              - action: utter_greet
            """
        )
        policy = self.create_policy(featurizer=featurizer, priority=priority)
        import tests.core.test_policies

        training_trackers = await tests.core.test_policies.train_trackers(
            default_domain, str(stories), augmentation_factor=20
        )
        with pytest.raises(RasaException) as e:
            policy.train(training_trackers, default_domain, RegexInterpreter())

        assert "No user features specified. Cannot train 'TED' model." == str(e.value)

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
        prediction = trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )
        assert not prediction.is_end_to_end_prediction
        # count number of non-zero confidences
        assert (
            sum([confidence > 0 for confidence in prediction.probabilities])
            == trained_policy.config[RANKING_LENGTH]
        )
        # check that the norm is still 1
        assert sum(prediction.probabilities) == pytest.approx(1)

        # also check our function is called
        mock = Mock()
        monkeypatch.setattr(train_utils, "normalize", mock.normalize)
        trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )

        mock.normalize.assert_called_once()

    async def test_gen_batch(
        self, trained_policy: TEDPolicy, default_domain: Domain, stories_path: Path
    ):
        training_trackers = await tests.core.test_policies.train_trackers(
            default_domain, stories_path, augmentation_factor=0
        )
        interpreter = RegexInterpreter()
        training_data, label_ids, entity_tags = trained_policy._featurize_for_training(
            training_trackers, default_domain, interpreter
        )
        _, all_labels = trained_policy._create_label_data(default_domain, interpreter)
        model_data = trained_policy._create_model_data(
            training_data, label_ids, entity_tags, all_labels
        )
        batch_size = 2
        data_generator = RasaBatchDataGenerator(
            model_data, batch_size=batch_size, shuffle=False, batch_strategy="sequence"
        )
        iterator = iter(data_generator)
        # model data keys were sorted, so the order is alphabetical
        (
            (
                batch_action_name_mask,
                _,
                _,
                batch_action_name_sentence_shape,
                batch_dialogue_length,
                batch_entities_mask,
                _,
                _,
                batch_entities_sentence_shape,
                batch_intent_mask,
                _,
                _,
                batch_intent_sentence_shape,
                batch_label_ids,
                batch_slots_mask,
                _,
                _,
                batch_slots_sentence_shape,
            ),
            _,
        ) = next(iterator)

        assert (
            batch_label_ids.shape[0] == batch_size
            and batch_dialogue_length.shape[0] == batch_size
        )
        # batch and dialogue dimensions are NOT combined for masks
        assert (
            batch_slots_mask.shape[0] == batch_size
            and batch_intent_mask.shape[0] == batch_size
            and batch_entities_mask.shape[0] == batch_size
            and batch_action_name_mask.shape[0] == batch_size
        )
        # some features might be "fake" so there sequence is `0`
        seq_len = max(
            [
                batch_intent_sentence_shape[1],
                batch_action_name_sentence_shape[1],
                batch_entities_sentence_shape[1],
                batch_slots_sentence_shape[1],
            ]
        )
        assert (
            batch_intent_sentence_shape[1] == seq_len
            or batch_intent_sentence_shape[1] == 0
        )
        assert (
            batch_action_name_sentence_shape[1] == seq_len
            or batch_action_name_sentence_shape[1] == 0
        )
        assert (
            batch_entities_sentence_shape[1] == seq_len
            or batch_entities_sentence_shape[1] == 0
        )
        assert (
            batch_slots_sentence_shape[1] == seq_len
            or batch_slots_sentence_shape[1] == 0
        )

        data_generator = RasaBatchDataGenerator(
            model_data, batch_size=batch_size, shuffle=True, batch_strategy="balanced"
        )
        iterator = iter(data_generator)

        (
            (
                batch_action_name_mask,
                _,
                _,
                batch_action_name_sentence_shape,
                batch_dialogue_length,
                batch_entities_mask,
                _,
                _,
                batch_entities_sentence_shape,
                batch_intent_mask,
                _,
                _,
                batch_intent_sentence_shape,
                batch_label_ids,
                batch_slots_mask,
                _,
                _,
                batch_slots_sentence_shape,
            ),
            _,
        ) = next(iterator)

        assert (
            batch_label_ids.shape[0] == batch_size
            and batch_dialogue_length.shape[0] == batch_size
        )
        # some features might be "fake" so there sequence is `0`
        seq_len = max(
            [
                batch_intent_sentence_shape[1],
                batch_action_name_sentence_shape[1],
                batch_entities_sentence_shape[1],
                batch_slots_sentence_shape[1],
            ]
        )
        assert (
            batch_intent_sentence_shape[1] == seq_len
            or batch_intent_sentence_shape[1] == 0
        )
        assert (
            batch_action_name_sentence_shape[1] == seq_len
            or batch_action_name_sentence_shape[1] == 0
        )
        assert (
            batch_entities_sentence_shape[1] == seq_len
            or batch_entities_sentence_shape[1] == 0
        )
        assert (
            batch_slots_sentence_shape[1] == seq_len
            or batch_slots_sentence_shape[1] == 0
        )


class TestTEDPolicyMargin(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer, priority=priority, **{LOSS_TYPE: "margin"}
        )

    def test_similarity_type(self, trained_policy: TEDPolicy):
        assert trained_policy.config[SIMILARITY_TYPE] == COSINE

    def test_confidence_type(self, trained_policy: TEDPolicy):
        assert trained_policy.config[MODEL_CONFIDENCE] == AUTO

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

    def test_prediction_on_empty_tracker(
        self, trained_policy: Policy, default_domain: Domain
    ):
        tracker = DialogueStateTracker(DEFAULT_SENDER_ID, default_domain.slots)
        prediction = trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )
        assert not prediction.is_end_to_end_prediction
        assert len(prediction.probabilities) == default_domain.num_actions
        assert max(prediction.probabilities) <= 1.0
        assert min(prediction.probabilities) >= -1.0


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


class TestTEDPolicyLinearNormConfidence(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer, priority=priority, **{MODEL_CONFIDENCE: LINEAR_NORM}
        )

    def test_confidence_type(self, trained_policy: TEDPolicy):
        assert trained_policy.config[MODEL_CONFIDENCE] == LINEAR_NORM

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

        output_sums_to_1 = sum(predicted_probabilities) == pytest.approx(1)
        assert output_sums_to_1

        # also check our function is not called
        mock = Mock()
        monkeypatch.setattr(train_utils, "normalize", mock.normalize)
        trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )

        mock.normalize.assert_not_called()

    def test_prediction_on_empty_tracker(
        self, trained_policy: Policy, default_domain: Domain
    ):
        tracker = DialogueStateTracker(DEFAULT_SENDER_ID, default_domain.slots)
        prediction = trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )
        assert not prediction.is_end_to_end_prediction
        assert len(prediction.probabilities) == default_domain.num_actions
        assert max(prediction.probabilities) <= 1.0
        assert min(prediction.probabilities) >= 0.0


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
