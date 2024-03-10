import json
from pathlib import Path
from typing import Optional, List, Dict, Type
import tensorflow as tf
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.logging import LogCaptureFixture
import logging

from rasa.core.featurizers.single_state_featurizer import (
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import IntentMaxHistoryTrackerFeaturizer
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.policies.ted_policy import PREDICTION_FEATURES
from rasa.core.policies.unexpected_intent_policy import (
    UnexpecTEDIntentPolicy,
    RankingCandidateMetadata,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.constants import ACTION_UNLIKELY_INTENT_NAME, ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    EntitiesAdded,
    SlotSet,
    ActionExecutionRejected,
    ActiveLoop,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.tensorflow.constants import (
    IGNORE_INTENTS_LIST,
    LABEL,
    MASK,
    SENTENCE,
    IDS,
    POSITIVE_SCORES_KEY,
    NEGATIVE_SCORES_KEY,
    RANKING_KEY,
    RANKING_LENGTH,
)
from rasa.shared.nlu.constants import INTENT
from rasa.shared.core.events import Event
from rasa.utils.tensorflow import model_data_utils
from tests.core.test_policies import train_trackers
from tests.core.policies.test_ted_policy import TestTEDPolicy


class TestUnexpecTEDIntentPolicy(TestTEDPolicy):
    @staticmethod
    def _policy_class_to_test() -> Type[UnexpecTEDIntentPolicy]:

        return UnexpecTEDIntentPolicy

    @pytest.fixture(scope="class")
    def featurizer(self) -> TrackerFeaturizer:
        featurizer = IntentMaxHistoryTrackerFeaturizer(
            IntentTokenizerSingleStateFeaturizer(), max_history=self.max_history
        )
        return featurizer

    @staticmethod
    def persist_and_load_policy(
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ):
        return trained_policy.__class__.load(
            trained_policy.config, model_storage, resource, execution_context
        )

    def test_ranking_length(self, trained_policy: UnexpecTEDIntentPolicy):
        assert trained_policy.config[RANKING_LENGTH] == LABEL_RANKING_LENGTH

    def test_ranking_length_and_renormalization(
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        tracker: DialogueStateTracker,
        default_domain: Domain,
    ):
        precomputations = None
        prediction_metadata = trained_policy.predict_action_probabilities(
            tracker, default_domain, precomputations
        ).action_metadata
        assert (
            prediction_metadata is None
            or len(prediction_metadata[RANKING_KEY])
            == trained_policy.config[RANKING_LENGTH]
        )

    def test_label_data_assembly(
        self, trained_policy: UnexpecTEDIntentPolicy, default_domain: Domain
    ):

        # Construct input data
        state_featurizer = trained_policy.featurizer.state_featurizer
        encoded_all_labels = state_featurizer.encode_all_labels(
            default_domain, precomputations=None
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

    def test_training_with_no_intent(
        self,
        featurizer: Optional[TrackerFeaturizer],
        default_domain: Domain,
        tmp_path: Path,
        caplog: LogCaptureFixture,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ):
        stories = tmp_path / "stories.yml"
        stories.write_text(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            stories:
            - story: test path
              steps:
              - action: utter_greet
            """
        )
        policy = self.create_policy(
            featurizer=featurizer,
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )
        import tests.core.test_policies

        training_trackers = tests.core.test_policies.train_trackers(
            default_domain, str(stories), augmentation_factor=20
        )

        with pytest.warns(UserWarning):
            policy.train(training_trackers, default_domain, precomputations=None)

    def test_prepared_data_for_threshold_prediction(
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        default_domain: Domain,
        stories_path: Path,
    ):
        training_trackers = train_trackers(
            default_domain, stories_path, augmentation_factor=0
        )
        training_model_data, _ = trained_policy._prepare_for_training(
            training_trackers, default_domain, precomputations=None
        )

        data_for_prediction = trained_policy._prepare_data_for_prediction(
            training_model_data
        )

        assert set(data_for_prediction.data.keys()).issubset(PREDICTION_FEATURES)

    def test_similarities_collection_for_label_ids(self):
        label_ids = np.array([[0, 1], [1, -1], [2, -1]])
        outputs = {
            "similarities": np.array(
                [[[1.2, 0.3, 0.2]], [[0.5, 0.2, 1.6]], [[0.01, 0.1, 1.7]]]
            )
        }
        label_id_similarities = UnexpecTEDIntentPolicy._collect_label_id_grouped_scores(
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
            NEGATIVE_SCORES_KEY: [0.2, 1.6],
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
        thresholds = UnexpecTEDIntentPolicy._compute_label_quantiles(label_id_scores)
        assert sorted(list(thresholds.keys())) == sorted(
            list(expected_thresholds.keys())
        )
        for label_id, tolerance_thresholds in thresholds.items():
            assert expected_thresholds[label_id] == tolerance_thresholds

    def test_post_training_threshold_computation(
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        default_domain: Domain,
        stories_path: Path,
    ):
        training_trackers = train_trackers(
            default_domain, stories_path, augmentation_factor=0
        )
        training_model_data, label_ids = trained_policy._prepare_for_training(
            training_trackers, default_domain, precomputations=None
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
        thresholds = UnexpecTEDIntentPolicy._pick_thresholds(
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
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        default_domain: Domain,
        predicted_similarity: float,
        threshold_value: float,
        is_unlikely: bool,
        tmp_path: Path,
    ):
        loaded_policy = self.persist_and_load_policy(
            trained_policy, model_storage, resource, execution_context
        )
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
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        default_domain: Domain,
        tmp_path: Path,
    ):
        loaded_policy = self.persist_and_load_policy(
            trained_policy, model_storage, resource, execution_context
        )

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
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        default_domain: Domain,
        tmp_path: Path,
    ):
        loaded_policy = self.persist_and_load_policy(
            trained_policy, model_storage, resource, execution_context
        )

        expected_probabilities = [0] * default_domain.num_actions

        precomputations = None
        tracker = DialogueStateTracker(sender_id="init", slots=default_domain.slots)
        prediction = loaded_policy.predict_action_probabilities(
            tracker, default_domain, precomputations
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
            tracker, default_domain, precomputations
        )

        assert prediction.probabilities == expected_probabilities

        loaded_policy.model = None

        prediction = loaded_policy.predict_action_probabilities(
            tracker, default_domain, precomputations
        )

        assert prediction.probabilities == expected_probabilities

    @pytest.mark.parametrize(
        "predicted_similarity, threshold_value, is_unlikely",
        [(1.2, 0.2, False), (0.3, -0.1, False), (-1.5, 0.03, True)],
    )
    def test_action_unlikely_intent_prediction(
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        default_domain: Domain,
        predicted_similarity: float,
        threshold_value: float,
        is_unlikely: bool,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ):
        loaded_policy = self.persist_and_load_policy(
            trained_policy, model_storage, resource, execution_context
        )

        similarities = np.array([[[0.0] * len(default_domain.intents)]])

        dummy_intent_index = 4
        similarities[0, 0, dummy_intent_index] = predicted_similarity
        query_intent = default_domain.intents[dummy_intent_index]

        loaded_policy.label_thresholds[dummy_intent_index] = threshold_value

        precomputations = None
        tracker = DialogueStateTracker(sender_id="init", slots=default_domain.slots)

        tracker.update_with_events(
            [UserUttered(text="hello", intent={"name": query_intent})], default_domain
        )

        # Preset the model predictions to the similarity values
        # so that we don't need to hardcode for particular model predictions.
        monkeypatch.setattr(
            loaded_policy.model,
            "run_inference",
            lambda data: {"similarities": similarities},
        )

        prediction = loaded_policy.predict_action_probabilities(
            tracker, default_domain, precomputations
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

            # Make sure metadata is set. The exact structure
            # of the metadata is tested separately and
            # not as part of this test.
            assert prediction.action_metadata is not None
            # Assert metadata is serializable
            assert json.dumps(prediction.action_metadata)

    @pytest.mark.parametrize(
        "tracker_events, should_skip",
        [
            ([], True),
            ([ActionExecuted("action_listen")], True),
            (
                [
                    ActionExecuted("action_listen"),
                    UserUttered("hi", intent={"name": "greet"}),
                ],
                False,
            ),
            (
                [
                    ActionExecuted("action_listen"),
                    UserUttered("hi", intent={"name": "greet"}),
                    EntitiesAdded([{"name": "dummy"}]),
                ],
                False,
            ),
            (
                [
                    ActionExecuted("action_listen"),
                    UserUttered("hi", intent={"name": "greet"}),
                    SlotSet("name"),
                ],
                False,
            ),
            (
                [
                    ActiveLoop("loop"),
                    ActionExecuted("action_listen"),
                    UserUttered("hi", intent={"name": "greet"}),
                    ActionExecutionRejected("loop"),
                ],
                False,
            ),
            (
                [
                    ActionExecuted("action_listen"),
                    UserUttered("hi", intent={"name": "greet"}),
                    ActionExecuted("utter_greet"),
                ],
                True,
            ),
        ],
    )
    def test_skip_predictions_to_prevent_loop(
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        default_domain: Domain,
        caplog: LogCaptureFixture,
        tracker_events: List[Event],
        should_skip: bool,
        tmp_path: Path,
    ):
        """Skips predictions to prevent loop."""
        precomputations = None
        tracker = DialogueStateTracker(sender_id="init", slots=default_domain.slots)
        tracker.update_with_events(tracker_events, default_domain)
        with caplog.at_level(logging.DEBUG):
            prediction = trained_policy.predict_action_probabilities(
                tracker, default_domain, precomputations
            )

        assert (
            "Skipping predictions for UnexpecTEDIntentPolicy" in caplog.text
        ) == should_skip

        if should_skip:
            assert prediction.probabilities == trained_policy._default_predictions(
                default_domain
            )

    @pytest.mark.parametrize(
        "tracker_events",
        [
            [
                ActionExecuted("action_listen"),
                UserUttered("hi", intent={"name": "inexistent_intent"}),
            ],
            [
                ActionExecuted("action_listen"),
                UserUttered("hi", intent={"name": "inexistent_intent"}),
                EntitiesAdded([{"name": "dummy"}]),
            ],
            [
                ActionExecuted("action_listen"),
                UserUttered("hi", intent={"name": "inexistent_intent"}),
                SlotSet("name"),
            ],
            [
                ActiveLoop("loop"),
                ActionExecuted("action_listen"),
                UserUttered("hi", intent={"name": "inexistent_intent"}),
                ActionExecutionRejected("loop"),
            ],
        ],
    )
    def test_skip_predictions_if_new_intent(
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        default_domain: Domain,
        caplog: LogCaptureFixture,
        tracker_events: List[Event],
    ):
        """Skips predictions if there's a new intent created."""
        tracker = DialogueStateTracker(sender_id="init", slots=default_domain.slots)
        tracker.update_with_events(tracker_events, default_domain)

        with caplog.at_level(logging.DEBUG):
            prediction = trained_policy.predict_action_probabilities(
                tracker, default_domain, precomputations=None
            )

        assert "Skipping predictions for UnexpecTEDIntentPolicy" in caplog.text

        assert prediction.probabilities == trained_policy._default_predictions(
            default_domain
        )

    @pytest.mark.parametrize(
        "tracker_events_with_action, tracker_events_without_action",
        [
            (
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                    ActionExecuted("utter_greet"),
                    UserUttered(text="sad", intent={"name": "thank_you"}),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    ActionExecuted("utter_greet"),
                    UserUttered(text="sad", intent={"name": "thank_you"}),
                ],
            ),
            (
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    EntitiesAdded(entities=[{"entity": "name", "value": "Peter"}]),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                    ActionExecuted("utter_greet"),
                    UserUttered(text="sad", intent={"name": "thank_you"}),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    EntitiesAdded(entities=[{"entity": "name", "value": "Peter"}]),
                    ActionExecuted("utter_greet"),
                    UserUttered(text="sad", intent={"name": "thank_you"}),
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
                    UserUttered(text="sad", intent={"name": "thank_you"}),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                    ActionExecuted("some_form"),
                    ActiveLoop("some_form"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="default", intent={"name": "default"}),
                    UserUttered(text="sad", intent={"name": "thank_you"}),
                ],
            ),
        ],
    )
    def test_ignore_action_unlikely_intent(
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        default_domain: Domain,
        tracker_events_with_action: List[Event],
        tracker_events_without_action: List[Event],
        tmp_path: Path,
    ):
        precomputations = None
        tracker_with_action = DialogueStateTracker.from_events(
            "test 1", evts=tracker_events_with_action
        )
        tracker_without_action = DialogueStateTracker.from_events(
            "test 2", evts=tracker_events_without_action
        )
        prediction_with_action = trained_policy.predict_action_probabilities(
            tracker_with_action, default_domain, precomputations
        )
        prediction_without_action = trained_policy.predict_action_probabilities(
            tracker_without_action, default_domain, precomputations
        )

        # If the weights didn't change then both trackers
        # should result in same prediction. For `UnexpecTEDIntentPolicy`, the real
        # prediction is inside action metadata.
        assert (
            prediction_with_action.action_metadata
            == prediction_without_action.action_metadata
        )

    def test_label_embedding_collection(self, trained_policy: UnexpecTEDIntentPolicy):
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

    @pytest.mark.parametrize(
        "query_intent_index, ranking_length", [(0, 0), (1, 3), (2, 1), (5, 0)]
    )
    def test_collect_action_metadata(
        self,
        trained_policy: UnexpecTEDIntentPolicy,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        default_domain: Domain,
        tmp_path: Path,
        query_intent_index: int,
        ranking_length: int,
    ):
        loaded_policy = self.persist_and_load_policy(
            trained_policy, model_storage, resource, execution_context
        )

        def test_individual_label_metadata(
            label_metadata: RankingCandidateMetadata,
            all_thresholds: Dict[int, float],
            all_similarities: np.array,
            label_index: int,
        ):

            expected_score = all_similarities[0][label_index]
            expected_threshold = (
                all_thresholds[label_index] if label_index in all_thresholds else None
            )
            expected_severity = (
                expected_threshold - expected_score if expected_threshold else None
            )

            assert label_metadata.score == expected_score
            assert label_metadata.threshold == expected_threshold
            assert label_metadata.severity == expected_severity

        # Monkey-patch certain attributes of the policy to make the testing easier.
        label_thresholds = {0: 1.2, 1: -0.3, 4: -2.3, 5: 0.2}
        loaded_policy.label_thresholds = label_thresholds
        loaded_policy.config[RANKING_LENGTH] = ranking_length

        # Some dummy similarities
        similarities = np.array([[3.2, 0.2, -1.2, -4.3, -5.1, 2.3]])

        query_intent = default_domain.intents[query_intent_index]

        metadata = loaded_policy._collect_action_metadata(
            default_domain, similarities, query_intent=query_intent
        )

        # Test all elements of metadata for query intent
        assert metadata.query_intent.name == query_intent
        test_individual_label_metadata(
            metadata.query_intent,
            label_thresholds,
            similarities,
            query_intent_index,
        )

        # Check if ranking is sorted correctly and truncated to `ranking_length`
        sorted_label_similarities = sorted(
            [(index, score) for index, score in enumerate(similarities[0])],
            key=lambda x: -x[1],
        )
        sorted_label_similarities = (
            sorted_label_similarities[:ranking_length]
            if ranking_length
            else sorted_label_similarities
        )
        expected_label_rankings = [
            default_domain.intents[index] for index, _ in sorted_label_similarities
        ]
        collected_label_rankings = [
            label_metadata.name for label_metadata in metadata.ranking
        ]
        assert collected_label_rankings == expected_label_rankings

        # Test all elements of metadata for all labels in ranking
        for label_metadata in metadata.ranking:
            label_index = default_domain.intents.index(label_metadata.name)
            test_individual_label_metadata(
                label_metadata, label_thresholds, similarities, label_index
            )

    @pytest.mark.parametrize(
        "tracker_events_for_training, expected_trackers_with_events",
        [
            # Filter because of no intent and action name
            (
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ],
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello"),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="happy to make it work"),
                        ActionExecuted(action_text="Great!"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ],
                ],
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ]
                ],
            ),
            # Filter because of no action name
            (
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ],
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello"),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted(action_text="Great!"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ],
                ],
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ]
                ],
            ),
            # Filter because of no intent
            (
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ],
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello"),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="happy to make it work"),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ],
                ],
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ]
                ],
            ),
            # No filter needed
            (
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ]
                ],
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted("utter_goodbye"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ]
                ],
            ),
            # Filter to return empty list of trackers
            (
                [
                    [
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(text="hello", intent={"name": "greet"}),
                        ActionExecuted("utter_greet"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                        UserUttered(
                            text="happy to make it work", intent={"name": "goodbye"}
                        ),
                        ActionExecuted(action_text="Great!"),
                        ActionExecuted(ACTION_LISTEN_NAME),
                    ]
                ],
                [],
            ),
        ],
    )
    def test_filter_training_trackers(
        self,
        tracker_events_for_training: List[List[Event]],
        expected_trackers_with_events: List[List[Event]],
        domain: Domain,
    ):
        trackers_for_training = [
            TrackerWithCachedStates.from_events(
                sender_id=f"{tracker_index}", evts=events, domain=domain
            )
            for tracker_index, events in enumerate(tracker_events_for_training)
        ]

        filtered_trackers = UnexpecTEDIntentPolicy._get_trackers_for_training(
            trackers_for_training
        )
        assert len(filtered_trackers) == len(expected_trackers_with_events)
        for collected_tracker, expected_tracker_events in zip(
            filtered_trackers, expected_trackers_with_events
        ):
            collected_tracker_events = list(collected_tracker.events)
            assert collected_tracker_events == expected_tracker_events


@pytest.mark.parametrize(
    "tracker_events, skip_training",
    [
        (
            [
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    ActionExecuted("utter_greet"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(
                        text="happy to make it work", intent={"name": "goodbye"}
                    ),
                    ActionExecuted("utter_goodbye"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello"),
                    ActionExecuted("utter_greet"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="happy to make it work"),
                    ActionExecuted(action_text="Great!"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                ],
            ],
            False,
        ),
        (
            [
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello"),
                    ActionExecuted("utter_greet"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="happy to make it work"),
                    ActionExecuted(action_text="Great!"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                ]
            ],
            True,
        ),
        (
            [
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello"),
                    ActionExecuted("utter_greet"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="happy to make it work"),
                    ActionExecuted("utter_goodbye"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                ]
            ],
            True,
        ),
        (
            [
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello"),
                    ActionExecuted("utter_greet"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(
                        text="happy to make it work", intent={"name": "goodbye"}
                    ),
                    ActionExecuted(action_text="Great!"),
                    ActionExecuted(ACTION_LISTEN_NAME),
                ]
            ],
            True,
        ),
    ],
)
def test_train_with_e2e_data(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    tracker_events: List[List[Event]],
    skip_training: bool,
    domain: Domain,
):
    policy = UnexpecTEDIntentPolicy(
        UnexpecTEDIntentPolicy.get_default_config(),
        default_model_storage,
        Resource("UnexpecTEDIntentPolicy"),
        default_execution_context,
        featurizer=IntentMaxHistoryTrackerFeaturizer(
            IntentTokenizerSingleStateFeaturizer()
        ),
    )
    trackers_for_training = [
        TrackerWithCachedStates.from_events(
            sender_id=f"{tracker_index}", evts=events, domain=domain
        )
        for tracker_index, events in enumerate(tracker_events)
    ]
    if skip_training:
        with pytest.warns(UserWarning):
            policy.train(trackers_for_training, domain, precomputations=None)
    else:
        policy.train(trackers_for_training, domain, precomputations=None)
