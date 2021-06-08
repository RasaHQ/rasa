from pathlib import Path
from typing import Optional, Dict, List
import tensorflow as tf
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.logging import LogCaptureFixture

from rasa.core.featurizers.single_state_featurizer import (
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.core.featurizers.tracker_featurizers import (
    IntentMaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
)
from rasa.core.policies.ted_policy import PREDICTION_FEATURES
from rasa.core.policies.intent_ted_policy import IntentTEDPolicy, IntentTED
from rasa.shared.core.constants import ACTION_UNLIKELY_INTENT_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.utils.tensorflow.constants import (
    IGNORE_INTENTS_LIST,
    LABEL,
    MASK,
    SENTENCE,
    IDS,
)
from rasa.shared.nlu.constants import INTENT
from rasa.utils.tensorflow import model_data_utils
from tests.core.test_policies import train_trackers
from tests.core.policies.test_ted_policy import TestTEDPolicy


class TestIntentTEDPolicy(TestTEDPolicy):
    def create_policy(
        self, featurizer: Optional[TrackerFeaturizer], priority: int
    ) -> IntentTEDPolicy:
        return IntentTEDPolicy(featurizer=featurizer, priority=priority)

    @pytest.fixture(scope="class")
    def featurizer(self) -> TrackerFeaturizer:
        featurizer = IntentMaxHistoryTrackerFeaturizer(
            IntentTokenizerSingleStateFeaturizer(), max_history=self.max_history
        )
        return featurizer

    @pytest.mark.skip
    def test_normalization(
        self,
        trained_policy: IntentTEDPolicy,
        tracker: DialogueStateTracker,
        default_domain: Domain,
        monkeypatch: MonkeyPatch,
    ):
        # No normalization is done for IntentTEDPolicy and
        # hence this test is overridden to do nothing.
        assert True

    def test_label_data_assembly(
        self, trained_policy: IntentTEDPolicy, default_domain: Domain
    ):
        interpreter = RegexInterpreter()

        # Construct input data
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
            f"{LABEL}_{INTENT}",
            LABEL,
        ]
        assert assembled_label_data.num_examples == len(default_domain.intents)
        assert list(assembled_label_data_signature[f"{LABEL}_{INTENT}"].keys()) == [
            MASK,
            SENTENCE,
        ]
        assert list(assembled_label_data_signature[LABEL].keys()) == [IDS]
        assert assembled_label_data_signature[f"{LABEL}_{INTENT}"][SENTENCE][
            0
        ].units == len(default_domain.intents)

    async def test_training_with_no_intent(
        self,
        featurizer: Optional[TrackerFeaturizer],
        priority: int,
        default_domain: Domain,
        tmp_path: Path,
        caplog: LogCaptureFixture,
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
        policy.train(training_trackers, default_domain, RegexInterpreter())

        assert (
            "Can not train 'IntentTEDPolicy'. No data was provided. "
            "Skipping training of the policy." in caplog.text
        )

    async def test_prepared_data_for_threshold_prediction(
        self,
        trained_policy: IntentTEDPolicy,
        default_domain: Domain,
        stories_path: Path,
    ):
        training_trackers = await train_trackers(
            default_domain, stories_path, augmentation_factor=0
        )
        interpreter = RegexInterpreter()
        training_model_data, _ = trained_policy._prepare_for_training(
            training_trackers, default_domain, interpreter
        )

        data_for_prediction = trained_policy._prepare_data_for_prediction(
            training_model_data
        )

        assert set(data_for_prediction.data.keys()).issubset(PREDICTION_FEATURES)

    def test_similarities_collection_for_label_ids(self):
        label_ids = np.array([[0, 1], [1, -1], [1, 0], [2, -1]])
        outputs = {
            "similarities": np.array(
                [
                    [[1.2, 0.3, 0.2]],
                    [[0.5, 0.2, 1.6]],
                    [[0.1, 0.6, 0.8]],
                    [[2.3, 0.1, 0.3]],
                ]
            )
        }
        label_id_similarities = IntentTED._collect_label_id_similarities_from_outputs(
            label_ids, outputs
        )

        # Should contain similarities for all label ids.
        assert sorted(list(label_id_similarities.keys())) == [0, 1, 2]

        # Cross-check that the collected similarities are correct for each label id.
        assert label_id_similarities[0] == [1.2]
        assert label_id_similarities[1] == [0.2, 0.6]
        assert label_id_similarities[2] == [0.3]

    @pytest.mark.parametrize(
        "similarities, expected_thresholds",
        [
            ({0: [1.2], 1: [-0.2, 0.6]}, {0: 1.2, 1: -0.2}),
            ({0: [0.2, 0.1, 0.8], 1: [0.9, 0.6]}, {0: 0.1, 1: 0.6}),
            ({0: [-0.2, -0.1, -0.8], 1: [-0.9, -0.6]}, {0: -0.8, 1: -0.9}),
        ],
    )
    def test_threshold_computation_from_similarities(
        self,
        similarities: Dict[int, List[float]],
        expected_thresholds: Dict[int, float],
    ):
        computed_thresholds = IntentTED._pick_threshold_from_similarities(similarities)
        assert computed_thresholds == expected_thresholds

    async def test_post_training_threshold_computation(
        self,
        trained_policy: IntentTEDPolicy,
        default_domain: Domain,
        stories_path: Path,
    ):
        training_trackers = await train_trackers(
            default_domain, stories_path, augmentation_factor=0
        )
        interpreter = RegexInterpreter()
        training_model_data, label_ids = trained_policy._prepare_for_training(
            training_trackers, default_domain, interpreter
        )

        trained_policy.calculate_label_thresholds_post_training(
            training_model_data, label_ids
        )

        # -1 is used for padding and hence is not expected in the keys
        expected_keys = list(np.unique(label_ids))
        expected_keys.remove(-1)

        assert sorted(list(trained_policy.label_thresholds.keys())) == sorted(
            expected_keys
        )

    @pytest.mark.parametrize(
        "predicted_similarity, threshold_value, is_unlikely",
        [(1.2, 0.2, False), (0.3, -0.1, False), (-1.5, 0.03, True)],
    )
    def test_unlikely_intent_check(
        self,
        trained_policy: IntentTEDPolicy,
        default_domain: Domain,
        predicted_similarity: float,
        threshold_value: float,
        is_unlikely: bool,
    ):
        # Construct dummy similarities
        similarities = np.array([[0.0] * len(default_domain.intents)])
        dummy_intent_index = 4
        similarities[0, dummy_intent_index] = predicted_similarity

        # Record the original thresholds because the test will modify it.
        # Need to revert it back so that the `trained_policy` can
        # be reused across tests.
        original_label_thresholds = trained_policy.label_thresholds

        trained_policy.label_thresholds[dummy_intent_index] = threshold_value
        query_intent = default_domain.intents[dummy_intent_index]

        unlikely_intent_prediction = trained_policy._check_unlikely_intent(
            default_domain, similarities, query_intent
        )

        assert is_unlikely == unlikely_intent_prediction

        # Revert back the label thresholds to original values.
        trained_policy.label_thresholds = original_label_thresholds

    def test_should_check_for_intent(
        self, trained_policy: IntentTEDPolicy, default_domain: Domain
    ):
        intent_index = 0
        assert (
            trained_policy._should_check_for_intent(
                default_domain.intents[intent_index], default_domain
            )
            is False
        )

        intent_index = 4
        assert trained_policy._should_check_for_intent(
            default_domain.intents[intent_index], default_domain
        )

        trained_policy.config[IGNORE_INTENTS_LIST] = [
            default_domain.intents[intent_index]
        ]
        assert (
            trained_policy._should_check_for_intent(
                default_domain.intents[intent_index], default_domain
            )
            is False
        )

        # Revert back the ignore intents list to empty
        # so that the original `trained_policy` object is not modified.
        trained_policy.config[IGNORE_INTENTS_LIST] = []

    def test_no_action_unlikely_intent_prediction(
        self, trained_policy: IntentTEDPolicy, default_domain: Domain
    ):
        expected_probabilities = [0] * default_domain.num_actions

        interpreter = RegexInterpreter()
        tracker = DialogueStateTracker(sender_id="init", slots=default_domain.slots)
        prediction = trained_policy.predict_action_probabilities(
            tracker, default_domain, interpreter
        )

        assert prediction.probabilities == expected_probabilities

        tracker.update_with_events(
            [
                UserUttered(text="hello", intent={"name": "greet"}),
                ActionExecuted(action_name="utter_greet"),
            ],
            default_domain,
        )
        prediction = trained_policy.predict_action_probabilities(
            tracker, default_domain, interpreter
        )

        assert prediction.probabilities == expected_probabilities

        # Preserve the original model because the test will modify it.
        # Need to revert it back so that the `trained_policy` can be
        # reused across tests.
        original_model = trained_policy.model

        trained_policy.model = None

        prediction = trained_policy.predict_action_probabilities(
            tracker, default_domain, interpreter
        )

        assert prediction.probabilities == expected_probabilities

        trained_policy.model = original_model

    @pytest.mark.parametrize(
        "predicted_similarity, threshold_value, is_unlikely",
        [(1.2, 0.2, False), (0.3, -0.1, False), (-1.5, 0.03, True)],
    )
    def test_action_unlikely_intent_prediction(
        self,
        trained_policy: IntentTEDPolicy,
        default_domain: Domain,
        predicted_similarity,
        threshold_value,
        is_unlikely,
        monkeypatch: MonkeyPatch,
    ):
        # Record the original thresholds because the test will modify it.
        # Need to revert it back so that the `trained_policy` can
        # be reused across tests.
        original_label_thresholds = trained_policy.label_thresholds

        similarities = np.array([[[0.0] * len(default_domain.intents)]])

        dummy_intent_index = 4
        similarities[0, 0, dummy_intent_index] = predicted_similarity
        query_intent = default_domain.intents[dummy_intent_index]

        trained_policy.label_thresholds[dummy_intent_index] = threshold_value

        interpreter = RegexInterpreter()
        tracker = DialogueStateTracker(sender_id="init", slots=default_domain.slots)

        tracker.update_with_events(
            [UserUttered(text="hello", intent={"name": query_intent})], default_domain,
        )

        # Preset the model predictions to the similarity values
        # so that we don't need to hardcode for particular model predictions.
        monkeypatch.setattr(
            trained_policy.model,
            "run_inference",
            lambda data: {"similarities": similarities},
        )

        prediction = trained_policy.predict_action_probabilities(
            tracker, default_domain, interpreter
        )

        if not is_unlikely:
            assert prediction.probabilities == [0.0] * default_domain.num_actions
        else:
            assert (
                prediction.probabilities[
                    default_domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME)
                ]
                == 1.0
            )

            # Make sure metadata is correct.
            expected_action_metadata = {
                intent: {
                    "score": similarities[0, 0, default_domain.intents.index(intent)],
                    "threshold": trained_policy.label_thresholds[
                        default_domain.intents.index(intent)
                    ],
                }
                for intent in default_domain.intents
                if default_domain.intents.index(intent)
                in trained_policy.label_thresholds
            }
            assert expected_action_metadata == prediction.action_metadata

        trained_policy.label_thresholds = original_label_thresholds

    def test_label_embedding_collection(self, trained_policy: IntentTEDPolicy):
        label_ids = tf.constant([[[2], [-1]], [[1], [2]], [[0], [-1]]], dtype=tf.int32)

        all_label_embeddings = np.random.random((10, 20))

        # `-1` is used as padding label id. The embedding for it
        # will be the same as `label_id=0`
        expected_extracted_label_embeddings = tf.constant(
            np.concatenate(
                [
                    all_label_embeddings[2],
                    all_label_embeddings[0],
                    all_label_embeddings[1],
                    all_label_embeddings[2],
                    all_label_embeddings[0],
                    all_label_embeddings[0],
                ]
            ).reshape((3, 2, 20)),
            dtype=tf.float32,
        )

        actual_extracted_label_embeddings = trained_policy.model._get_labels_embed(
            label_ids, tf.constant(all_label_embeddings, dtype=tf.float32)
        )

        assert np.all(
            expected_extracted_label_embeddings == actual_extracted_label_embeddings
        )
