from pathlib import Path
from typing import Optional, List, Type, Dict, Text, Any
from unittest.mock import Mock
import numpy as np
import pytest
import tests.core.test_policies
from _pytest.monkeypatch import MonkeyPatch
from _pytest.logging import LogCaptureFixture
from rasa.core.constants import POLICY_PRIORITY, POLICY_MAX_HISTORY

from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
)
from rasa.core.policies.policy import Policy2 as Policy
from rasa.core.policies.ted_policy import TEDPolicy2 as TEDPolicy
from rasa.core.policies.ted_policy import TEDPolicy as Rasa2TEDPolicy
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.constants import ACTION_LISTEN_NAME, ACTION_UNLIKELY_INTENT_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    Event,
    EntitiesAdded,
    ActiveLoop,
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
    LABEL,
    MASK,
    SENTENCE,
    IDS,
    EVAL_NUM_EPOCHS,
    EPOCHS,
)
from rasa.shared.nlu.constants import ACTION_NAME
from rasa.utils.tensorflow import model_data_utils
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
    policy_dir_name = Path("policy_{}_{}".format(ted_pos, Rasa2TEDPolicy.__name__))
    policy_path = train_path / DEFAULT_CORE_SUBDIRECTORY_NAME / policy_dir_name
    return policy_path / "checkpoints"


def test_diagnostics(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    domain = Domain.from_yaml(DOMAIN_YAML)
    policy = TEDPolicy(
        {}, default_model_storage, Resource(""), default_execution_context
    )
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
    @staticmethod
    def _policy_class_to_test() -> Type[TEDPolicy]:
        return TEDPolicy

    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> TEDPolicy:
        return TEDPolicy(
            {POLICY_PRIORITY: priority, **(config or {})},
            featurizer=featurizer,
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )

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

    @pytest.mark.parametrize(
        "should_finetune, epoch_override, expected_epoch_value",
        [
            (
                True,
                TEDPolicy.default_config[EPOCHS] + 1,
                TEDPolicy.default_config[EPOCHS] + 1,
            ),
            (
                False,
                TEDPolicy.default_config[EPOCHS] + 1,
                TEDPolicy.default_config[EPOCHS],
            ),  # trained_policy uses default epochs during training
        ],
    )
    def test_epoch_override_when_loaded(
        self,
        trained_policy: TEDPolicy,
        should_finetune: bool,
        epoch_override: int,
        expected_epoch_value: int,
        resource: Resource,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
    ):
        execution_context.is_finetuning = should_finetune
        loaded_policy = trained_policy.__class__.load(
            {"epoch_override": epoch_override},
            model_storage,
            resource,
            execution_context,
        )

        assert loaded_policy.config[EPOCHS] == expected_epoch_value

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

    def test_training_with_no_intent(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        default_domain: Domain,
        tmp_path: Path,
        caplog: LogCaptureFixture,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
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
        policy = self.create_policy(
            featurizer=featurizer,
            priority=priority,
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )
        import tests.core.test_policies

        training_trackers = tests.core.test_policies.train_trackers(
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

    def test_label_data_assembly(
        self, trained_policy: TEDPolicy, default_domain: Domain
    ):
        interpreter = RegexInterpreter()

        state_featurizer = trained_policy.featurizer.state_featurizer
        encoded_all_labels = state_featurizer.encode_all_labels(
            default_domain, interpreter
        )

        attribute_data, _ = model_data_utils.convert_to_data_format(encoded_all_labels)
        assembled_label_data = trained_policy._assemble_label_data(
            attribute_data, default_domain
        )
        assembled_label_data_signature = assembled_label_data.get_signature()

        assert list(assembled_label_data_signature.keys()) == [
            f"{LABEL}_{ACTION_NAME}",
            f"{LABEL}",
        ]
        assert assembled_label_data.num_examples == default_domain.num_actions
        assert list(
            assembled_label_data_signature[f"{LABEL}_{ACTION_NAME}"].keys()
        ) == [MASK, SENTENCE,]
        assert list(assembled_label_data_signature[LABEL].keys()) == [IDS]
        assert (
            assembled_label_data_signature[f"{LABEL}_{ACTION_NAME}"][SENTENCE][0].units
            == default_domain.num_actions
        )

    def test_gen_batch(
        self, trained_policy: TEDPolicy, default_domain: Domain, stories_path: Path
    ):
        training_trackers = tests.core.test_policies.train_trackers(
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

    @pytest.mark.parametrize(
        "tracker_events_with_action, tracker_events_without_action",
        [
            (
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                ],
            ),
            (
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    EntitiesAdded(entities=[{"entity": "name", "value": "Peter"},]),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                    ActionExecuted("utter_greet"),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    EntitiesAdded(entities=[{"entity": "name", "value": "Peter"},]),
                    ActionExecuted("utter_greet"),
                ],
            ),
            (
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                    ActionExecuted("some_form"),
                    ActiveLoop("some_form"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="default", intent={"name": "default"}),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                    ActionExecuted("some_form"),
                    ActiveLoop("some_form"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="default", intent={"name": "default"}),
                ],
            ),
        ],
    )
    def test_ignore_action_unlikely_intent(
        self,
        trained_policy: TEDPolicy,
        default_domain: Domain,
        tracker_events_with_action: List[Event],
        tracker_events_without_action: List[Event],
    ):
        interpreter = RegexInterpreter()
        tracker_with_action = DialogueStateTracker.from_events(
            "test 1", evts=tracker_events_with_action
        )
        tracker_without_action = DialogueStateTracker.from_events(
            "test 2", evts=tracker_events_without_action
        )
        prediction_with_action = trained_policy.predict_action_probabilities(
            tracker_with_action, default_domain, interpreter
        )
        prediction_without_action = trained_policy.predict_action_probabilities(
            tracker_without_action, default_domain, interpreter
        )

        # If the weights didn't change then both trackers
        # should result in same prediction.
        assert (
            prediction_with_action.probabilities
            == prediction_without_action.probabilities
        )


class TestTEDPolicyMargin(TestTEDPolicy):
    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        return TEDPolicy(
            {POLICY_PRIORITY: priority, LOSS_TYPE: "margin", **(config or {})},
            featurizer=featurizer,
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
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
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            config={
                POLICY_PRIORITY: priority,
                SCALE_LOSS: False,
                EVAL_NUM_EXAMPLES: 4,
                **(config or {}),
            },
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )


class TestTEDPolicyNoNormalization(TestTEDPolicy):
    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            config={RANKING_LENGTH: 0, POLICY_PRIORITY: priority, **(config or {})},
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
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
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            config={
                POLICY_PRIORITY: priority,
                MODEL_CONFIDENCE: LINEAR_NORM,
                **(config or {}),
            },
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
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
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            config={POLICY_PRIORITY: priority, RANKING_LENGTH: 3, **(config or {})},
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )

    def test_ranking_length(self, trained_policy: TEDPolicy):
        assert trained_policy.config[RANKING_LENGTH] == 3


class TestTEDPolicyHighRankingLength(TestTEDPolicy):
    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            config={POLICY_PRIORITY: priority, RANKING_LENGTH: 11, **(config or {})},
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )

    def test_ranking_length(self, trained_policy: TEDPolicy):
        assert trained_policy.config[RANKING_LENGTH] == 11


class TestTEDPolicyWithStandardFeaturizer(TestTEDPolicy):
    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        # use standard featurizer from TEDPolicy,
        # since it is using MaxHistoryTrackerFeaturizer
        # if max_history is not specified
        return TEDPolicy(
            config={POLICY_PRIORITY: priority, **(config or {})},
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )

    def test_featurizer(
        self,
        trained_policy: Policy,
        resource: Resource,
        model_storage: ModelStorage,
        tmp_path: Path,
        execution_context: ExecutionContext,
    ):
        assert isinstance(trained_policy.featurizer, MaxHistoryTrackerFeaturizer)
        assert isinstance(
            trained_policy.featurizer.state_featurizer, SingleStateFeaturizer
        )

        loaded = trained_policy.__class__.load(
            {}, model_storage, resource, execution_context
        )

        assert isinstance(loaded.featurizer, MaxHistoryTrackerFeaturizer)
        assert isinstance(loaded.featurizer.state_featurizer, SingleStateFeaturizer)


class TestTEDPolicyWithMaxHistory(TestTEDPolicy):
    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        # use standard featurizer from TEDPolicy,
        # since it is using MaxHistoryTrackerFeaturizer
        # if max_history is specified
        return TEDPolicy(
            config={
                POLICY_PRIORITY: priority,
                POLICY_MAX_HISTORY: self.max_history,
                **(config or {}),
            },
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )

    def test_featurizer(
        self,
        trained_policy: Policy,
        resource: Resource,
        model_storage: ModelStorage,
        tmp_path: Path,
        execution_context: ExecutionContext,
    ):
        assert isinstance(trained_policy.featurizer, MaxHistoryTrackerFeaturizer)
        assert trained_policy.featurizer.max_history == self.max_history
        assert isinstance(
            trained_policy.featurizer.state_featurizer, SingleStateFeaturizer
        )

        loaded = trained_policy.__class__.load(
            {}, model_storage, resource, execution_context
        )

        assert isinstance(loaded.featurizer, MaxHistoryTrackerFeaturizer)
        assert loaded.featurizer.max_history == self.max_history
        assert isinstance(loaded.featurizer.state_featurizer, SingleStateFeaturizer)


class TestTEDPolicyWithRelativeAttention(TestTEDPolicy):
    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            config={
                POLICY_PRIORITY: priority,
                KEY_RELATIVE_ATTENTION: True,
                VALUE_RELATIVE_ATTENTION: True,
                MAX_RELATIVE_POSITION: 5,
                **(config or {}),
            },
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )


class TestTEDPolicyWithRelativeAttentionMaxHistoryOne(TestTEDPolicy):

    max_history = 1

    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> Policy:
        return TEDPolicy(
            featurizer=featurizer,
            config={
                POLICY_PRIORITY: priority,
                KEY_RELATIVE_ATTENTION: True,
                VALUE_RELATIVE_ATTENTION: True,
                MAX_RELATIVE_POSITION: 5,
                **(config or {}),
            },
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )
