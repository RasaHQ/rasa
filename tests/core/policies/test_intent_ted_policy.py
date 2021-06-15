from pathlib import Path
from typing import Optional, List
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
from rasa.core.policies.intent_ted_policy import IntentTEDPolicy
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
    POSITIVE_SCORES_KEY,
    NEGATIVE_SCORES_KEY,
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

    @staticmethod
    def persist_and_load_policy(trained_policy: IntentTEDPolicy, tmp_path: Path):
        trained_policy.persist(tmp_path)
        return IntentTEDPolicy.load(tmp_path)

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
                    [[1.2, 0.3, 0.8]],
                    [[0.01, 0.1, 1.7]],
                ]
            )
        }
        label_id_similarities = IntentTEDPolicy._collect_label_id_grouped_scores(
            outputs, label_ids
        )

        # Should contain similarities for all label ids except padding token.
        assert sorted(list(label_id_similarities.keys())) == [0, 1, 2]

        # Cross-check that the collected similarities are correct for each label id.
        assert label_id_similarities[0] == {
            POSITIVE_SCORES_KEY: [1.2],
            NEGATIVE_SCORES_KEY: [0.5, 0.01],
        }
        assert label_id_similarities[1] == {
            POSITIVE_SCORES_KEY: [0.3, 0.2],
            NEGATIVE_SCORES_KEY: [0.1],
        }
        assert label_id_similarities[2] == {
            POSITIVE_SCORES_KEY: [1.7],
            NEGATIVE_SCORES_KEY: [0.2, 1.6, 0.8],
        }

    def test_label_quantiles_computation(self):
        label_id_scores = {
            0: {
                POSITIVE_SCORES_KEY: [1.3, 0.2],
                NEGATIVE_SCORES_KEY: [
                    -0.1,
                    -1.2,
                    -2.3,
                    -4.1,
                    -0.5,
                    0.2,
                    0.8,
                    0.9,
                    -3.2,
                    -2.7,
                ],
            },
            3: {POSITIVE_SCORES_KEY: [1.3, 0.2], NEGATIVE_SCORES_KEY: [-0.1]},
            6: {POSITIVE_SCORES_KEY: [1.3, 0.2], NEGATIVE_SCORES_KEY: []},
        }
        expected_thresholds = {
            0: [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                -0.1,
                -0.1,
                -0.5,
                -0.5,
                -1.2,
                -1.2,
                -1.2,
                -2.3,
                -2.3,
                -2.7,
                -2.7,
                -3.2,
                -3.2,
                -4.1,
                -4.1,
            ],
            3: [
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
            ],
            6: [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
            ],
        }
        thresholds = IntentTEDPolicy._compute_label_quantiles(label_id_scores)
        assert sorted(list(thresholds.keys())) == sorted(
            list(expected_thresholds.keys())
        )
        for label_id, tolerance_thresholds in thresholds.items():
            assert expected_thresholds[label_id] == tolerance_thresholds

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

        trained_policy.compute_label_quantiles_post_training(
            training_model_data, label_ids
        )

        computed_thresholds = trained_policy.label_quantiles

        # -1 is used for padding and hence is not expected in the keys
        expected_keys = list(np.unique(label_ids))
        expected_keys.remove(-1)

        assert sorted(list(computed_thresholds.keys())) == sorted(expected_keys)

    @pytest.mark.parametrize(
        "tolerance, expected_thresholds",
        [
            (0.0, [0.2, -0.1, 0.2]),
            (0.75, [-2.9, -0.1, -4.3]),
            (0.72, [-2.7, -0.1, -4.0]),
            (0.78, [-2.9, -0.1, -4.3]),
            (1.0, [-4.1, -0.1, -5.5]),
        ],
    )
    def test_pick_thresholds_for_labels(
        self, tolerance: float, expected_thresholds: List[float]
    ):
        label_id_tolerance_thresholds = {
            0: [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                -0.1,
                -0.1,
                -0.5,
                -0.5,
                -1.2,
                -1.2,
                -2.3,
                -2.3,
                -2.7,
                -2.9,
                -3.2,
                -3.2,
                -4.1,
                -4.1,
            ],
            3: [
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
            ],
            4: [0.2 - (index * 0.3) for index in range(20)],
        }
        thresholds = IntentTEDPolicy._pick_thresholds(
            label_id_tolerance_thresholds, tolerance
        )
        assert sorted(list(thresholds.keys())) == sorted(
            list(label_id_tolerance_thresholds.keys())
        )
        computed_values = list(thresholds.values())
        assert expected_thresholds == computed_values

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
        tmp_path: Path,
    ):
        loaded_policy = self.persist_and_load_policy(trained_policy, tmp_path)
        # Construct dummy similarities
        similarities = np.array([[0.0] * len(default_domain.intents)])
        dummy_intent_index = 4
        similarities[0, dummy_intent_index] = predicted_similarity

        loaded_policy.label_thresholds[dummy_intent_index] = threshold_value
        query_intent = default_domain.intents[dummy_intent_index]

        unlikely_intent_prediction = loaded_policy._check_unlikely_intent(
            default_domain, similarities, query_intent
        )

        assert is_unlikely == unlikely_intent_prediction

    def test_should_check_for_intent(
        self, trained_policy: IntentTEDPolicy, default_domain: Domain, tmp_path: Path
    ):
        loaded_policy = self.persist_and_load_policy(trained_policy, tmp_path)

        intent_index = 0
        assert (
            loaded_policy._should_check_for_intent(
                default_domain.intents[intent_index], default_domain
            )
            is False
        )

        intent_index = 4
        assert loaded_policy._should_check_for_intent(
            default_domain.intents[intent_index], default_domain
        )

        loaded_policy.config[IGNORE_INTENTS_LIST] = [
            default_domain.intents[intent_index]
        ]
        assert (
            loaded_policy._should_check_for_intent(
                default_domain.intents[intent_index], default_domain
            )
            is False
        )

    def test_no_action_unlikely_intent_prediction(
        self, trained_policy: IntentTEDPolicy, default_domain: Domain, tmp_path: Path
    ):
        loaded_policy = self.persist_and_load_policy(trained_policy, tmp_path)

        expected_probabilities = [0] * default_domain.num_actions

        interpreter = RegexInterpreter()
        tracker = DialogueStateTracker(sender_id="init", slots=default_domain.slots)
        prediction = loaded_policy.predict_action_probabilities(
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
        prediction = loaded_policy.predict_action_probabilities(
            tracker, default_domain, interpreter
        )

        assert prediction.probabilities == expected_probabilities

        loaded_policy.model = None

        prediction = loaded_policy.predict_action_probabilities(
            tracker, default_domain, interpreter
        )

        assert prediction.probabilities == expected_probabilities

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
        tmp_path: Path,
    ):
        loaded_policy = self.persist_and_load_policy(trained_policy, tmp_path)

        similarities = np.array([[[0.0] * len(default_domain.intents)]])

        dummy_intent_index = 4
        similarities[0, 0, dummy_intent_index] = predicted_similarity
        query_intent = default_domain.intents[dummy_intent_index]

        loaded_policy.label_thresholds[dummy_intent_index] = threshold_value

        interpreter = RegexInterpreter()
        tracker = DialogueStateTracker(sender_id="init", slots=default_domain.slots)

        tracker.update_with_events(
            [UserUttered(text="hello", intent={"name": query_intent})], default_domain,
        )

        # Preset the model predictions to the similarity values
        # so that we don't need to hardcode for particular model predictions.
        monkeypatch.setattr(
            loaded_policy.model,
            "run_inference",
            lambda data: {"similarities": similarities},
        )

        prediction = loaded_policy.predict_action_probabilities(
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
                    "threshold": loaded_policy.label_thresholds[
                        default_domain.intents.index(intent)
                    ],
                }
                for intent in default_domain.intents
                if default_domain.intents.index(intent)
                in loaded_policy.label_thresholds
            }
            assert expected_action_metadata == prediction.action_metadata

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
