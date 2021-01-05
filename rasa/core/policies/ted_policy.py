import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

import rasa.shared.utils.io
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Any, List, Optional, Text, Dict, Tuple, Union, TYPE_CHECKING

import rasa.utils.io as io_utils
import rasa.core.actions.action
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.classifiers.diet_classifier import EntityTagSpec
from rasa.shared.core.domain import Domain
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.nlu.constants import (
    ACTION_TEXT,
    ACTION_NAME,
    INTENT,
    TEXT,
    ENTITIES,
    VALID_FEATURE_TYPES,
    FEATURE_TYPE_SENTENCE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_TAGS,
    EXTRACTOR,
)
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE
from rasa.shared.core.constants import ACTIVE_LOOP, SLOTS, ACTION_LISTEN_NAME
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
import rasa.utils.train_utils
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureSignature,
    FeatureArray,
    Data,
)
from rasa.utils.tensorflow.model_data_utils import convert_to_data_format
from rasa.utils.tensorflow.constants import (
    LABEL,
    IDS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    DROP_RATE_DIALOGUE,
    DROP_RATE_LABEL,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    WEIGHT_SPARSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    SOFTMAX,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CHECKPOINT_MODEL,
    ENCODING_DIMENSION,
    UNIDIRECTIONAL_ENCODER,
    SEQUENCE,
    SENTENCE,
    SEQUENCE_LENGTH,
    DENSE_DIMENSION,
    CONCAT_DIMENSION,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    MASK,
    HIDDEN_LAYERS_SIZES,
    FEATURIZERS,
    ENTITY_RECOGNITION,
)
from rasa.shared.core.events import EntitiesAdded, Event
from rasa.shared.nlu.training_data.message import Message

if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features


logger = logging.getLogger(__name__)

E2E_CONFIDENCE_THRESHOLD = "e2e_confidence_threshold"
LABEL_KEY = LABEL
LABEL_SUB_KEY = IDS
LENGTH = "length"
INDICES = "indices"
SENTENCE_FEATURES_TO_ENCODE = [INTENT, TEXT, ACTION_NAME, ACTION_TEXT]
SEQUENCE_FEATURES_TO_ENCODE = [TEXT, ACTION_TEXT, f"{LABEL}_{ACTION_TEXT}"]
LABEL_FEATURES_TO_ENCODE = [f"{LABEL}_{ACTION_NAME}", f"{LABEL}_{ACTION_TEXT}"]
STATE_LEVEL_FEATURES = [ENTITIES, SLOTS, ACTIVE_LOOP]
PREDICTION_FEATURES = STATE_LEVEL_FEATURES + SENTENCE_FEATURES_TO_ENCODE + [DIALOGUE]

SAVE_MODEL_FILE_NAME = "ted_policy"


class TEDPolicy(Policy):
    """Transformer Embedding Dialogue (TED) Policy is described in
    https://arxiv.org/abs/1910.00486.
    This policy has a pre-defined architecture, which comprises the
    following steps:
        - concatenate user input (user intent and entities), previous system actions,
          slots and active forms for each time step into an input vector to
          pre-transformer embedding layer;
        - feed it to transformer;
        - apply a dense layer to the output of the transformer to get embeddings of a
          dialogue for each time step;
        - apply a dense layer to create embeddings for system actions for each time
          step;
        - calculate the similarity between the dialogue embedding and embedded system
          actions. This step is based on the StarSpace
          (https://arxiv.org/abs/1709.03856) idea.
    """

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding list.
        HIDDEN_LAYERS_SIZES: {TEXT: [], ACTION_TEXT: [], f"{LABEL}_{ACTION_TEXT}": []},
        # Dense dimension to use for sparse features.
        DENSE_DIMENSION: {
            TEXT: 128,
            ACTION_TEXT: 128,
            f"{LABEL}_{ACTION_TEXT}": 128,
            INTENT: 20,
            ACTION_NAME: 20,
            f"{LABEL}_{ACTION_NAME}": 20,
            ENTITIES: 20,
            SLOTS: 20,
            ACTIVE_LOOP: 20,
        },
        # Default dimension to use for concatenating sequence and sentence features.
        CONCAT_DIMENSION: {TEXT: 128, ACTION_TEXT: 128, f"{LABEL}_{ACTION_TEXT}": 128},
        # Dimension size of embedding vectors before the dialogue transformer encoder.
        ENCODING_DIMENSION: 50,
        # Number of units in transformer encoders
        TRANSFORMER_SIZE: {
            TEXT: 128,
            ACTION_TEXT: 128,
            f"{LABEL}_{ACTION_TEXT}": 128,
            DIALOGUE: 128,
        },
        # Number of layers in transformer encoders
        NUM_TRANSFORMER_LAYERS: {
            TEXT: 1,
            ACTION_TEXT: 1,
            f"{LABEL}_{ACTION_TEXT}": 1,
            DIALOGUE: 1,
        },
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # Use a unidirectional or bidirectional encoder.
        UNIDIRECTIONAL_ENCODER: True,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 256],
        # Strategy used whenc creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
        # Number of epochs to train
        EPOCHS: 1,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # The number of incorrect labels. The algorithm will minimize
        # their similarity to the user input during training.
        NUM_NEG: 20,
        # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
        SIMILARITY_TYPE: AUTO,
        # The type of the loss function, either 'softmax' or 'margin'.
        LOSS_TYPE: SOFTMAX,
        # Number of top actions to normalize scores for loss type 'softmax'.
        # Set to 0 to turn off normalization.
        RANKING_LENGTH: 10,
        # Indicates how similar the algorithm should try to make embedding vectors
        # for correct labels.
        # Should be 0.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_POS_SIM: 0.8,
        # Maximum negative similarity for incorrect labels.
        # Should be -1.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_NEG_SIM: -0.2,
        # If 'True' the algorithm only minimizes maximum similarity over
        # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
        USE_MAX_NEG_SIM: True,
        # If 'True' scale loss inverse proportionally to the confidence
        # of the correct prediction
        SCALE_LOSS: True,
        # ## Regularization parameters
        # The scale of regularization
        REGULARIZATION_CONSTANT: 0.001,
        # The scale of how important is to minimize the maximum similarity
        # between embeddings of different labels,
        # used only if 'loss_type' is set to 'margin'.
        NEGATIVE_MARGIN_SCALE: 0.8,
        # Dropout rate for embedding layers of dialogue features.
        DROP_RATE_DIALOGUE: 0.1,
        # Dropout rate for embedding layers of utterance level features.
        DROP_RATE: 0.0,
        # Dropout rate for embedding layers of label, e.g. action, features.
        DROP_RATE_LABEL: 0.0,
        # Dropout rate for attention.
        DROP_RATE_ATTENTION: 0.0,
        # Sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
        # If 'True' apply dropout to sparse input tensors
        SPARSE_INPUT_DROPOUT: True,
        # If 'True' apply dropout to dense input tensors
        DENSE_INPUT_DROPOUT: True,
        # If 'True' random tokens of the input message will be masked and the model
        # should predict those tokens.
        MASKED_LM: False,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EXAMPLES: 0,
        # If you want to use tensorboard to visualize training and validation metrics,
        # set this option to a valid output directory.
        TENSORBOARD_LOG_DIR: None,
        # Define when training metrics for tensorboard should be logged.
        # Either after every epoch or for every training step.
        # Valid values: 'epoch' and 'minibatch'
        TENSORBOARD_LOG_LEVEL: "epoch",
        # Perform model checkpointing
        CHECKPOINT_MODEL: False,
        # Only pick e2e prediction if the policy is confident enough
        E2E_CONFIDENCE_THRESHOLD: 0.5,
        # Specify what features to use as sequence and sentence features.
        # By default all features in the pipeline are used.
        FEATURIZERS: [],
        # If set to true, entities are predicted in user utterances.
        ENTITY_RECOGNITION: True,
    }

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return MaxHistoryTrackerFeaturizer(
            SingleStateFeaturizer(), max_history=max_history
        )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        fake_features: Optional[Dict[Text, List["Features"]]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        should_finetune: bool = False,
        **kwargs: Any,
    ) -> None:
        """Declare instance variables with default values."""
        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super().__init__(
            featurizer, priority, should_finetune=should_finetune, **kwargs
        )
        if isinstance(featurizer, FullDialogueTrackerFeaturizer):
            self.is_full_dialogue_featurizer_used = True
        else:
            self.is_full_dialogue_featurizer_used = False

        self._load_params(**kwargs)

        self.model = model

        self._entity_tag_specs = entity_tag_specs

        self.fake_features = fake_features or defaultdict(list)
        # TED is only e2e if only text is present in fake features, which represent
        # all possible input features for current version of this trained ted
        self.only_e2e = TEXT in self.fake_features and INTENT not in self.fake_features

        self._label_data: Optional[RasaModelData] = None
        self.data_example: Optional[Dict[Text, List[np.ndarray]]] = None

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        new_config = rasa.utils.train_utils.check_core_deprecated_options(kwargs)
        self.config = rasa.utils.train_utils.override_defaults(
            self.defaults, new_config
        )
        self.config = rasa.utils.train_utils.update_similarity_type(self.config)
        self.config = rasa.utils.train_utils.update_evaluation_parameters(self.config)

    def _create_entity_tag_specs(self) -> List[EntityTagSpec]:
        """Create entity tag specifications with their respective tag id mappings."""
        _tag_specs = []

        tag_id_index_mapping = self.featurizer.state_featurizer.get_entity_tag_ids()

        if tag_id_index_mapping:
            _tag_specs.append(
                EntityTagSpec(
                    tag_name=ENTITY_ATTRIBUTE_TYPE,
                    tags_to_ids=tag_id_index_mapping,
                    ids_to_tags={
                        value: key for key, value in tag_id_index_mapping.items()
                    },
                    num_tags=len(tag_id_index_mapping),
                )
            )

        return _tag_specs

    def _create_label_data(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> Tuple[RasaModelData, List[Dict[Text, List["Features"]]]]:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        encoded_all_labels = state_featurizer.encode_all_actions(domain, interpreter)

        attribute_data, _ = convert_to_data_format(
            encoded_all_labels, featurizers=self.config[FEATURIZERS]
        )

        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")
        label_data.add_lengths(
            f"{LABEL}_{ACTION_TEXT}",
            SEQUENCE_LENGTH,
            f"{LABEL}_{ACTION_TEXT}",
            SEQUENCE,
        )

        label_ids = np.arange(domain.num_actions)
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )

        return label_data, encoded_all_labels

    def _create_data_for_entities(
        self, entity_tags: Optional[List[List[Dict[Text, List["Features"]]]]]
    ) -> Optional[Data]:
        if not self.config[ENTITY_RECOGNITION]:
            return

        # check that there are real entity tags
        if entity_tags and any([any(turn_tags) for turn_tags in entity_tags]):
            entity_tags_data, _ = convert_to_data_format(entity_tags)
            return entity_tags_data

        # there are no "real" entity tags
        logger.debug(
            f"Entity recognition cannot be performed, "
            f"set '{ENTITY_RECOGNITION}' config parameter to 'False'."
        )
        self.config[ENTITY_RECOGNITION] = False

    def _create_model_data(
        self,
        tracker_state_features: List[List[Dict[Text, List["Features"]]]],
        label_ids: Optional[np.ndarray] = None,
        entity_tags: Optional[List[List[Dict[Text, List["Features"]]]]] = None,
        encoded_all_labels: Optional[List[Dict[Text, List["Features"]]]] = None,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData.

        Args:
            tracker_state_features: a dictionary of attributes
                (INTENT, TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, ACTIVE_LOOP)
                to a list of features for all dialogue turns in all training trackers
            label_ids: the label ids (e.g. action ids) for every dialogue turn in all
                training trackers
            entity_tags: a dictionary of entity type (ENTITY_TAGS) to a list of features
                containing entity tag ids for text user inputs otherwise empty dict
                for all dialogue turns in all training trackers
            encoded_all_labels: a list of dictionaries containing attribute features
                for label ids

        Returns:
            RasaModelData
        """
        model_data = RasaModelData(label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY)

        if label_ids is not None and encoded_all_labels is not None:
            label_ids = np.array(
                [np.expand_dims(seq_label_ids, -1) for seq_label_ids in label_ids]
            )
            model_data.add_features(
                LABEL_KEY,
                LABEL_SUB_KEY,
                [FeatureArray(label_ids, number_of_dimensions=3)],
            )

            attribute_data, self.fake_features = convert_to_data_format(
                tracker_state_features, featurizers=self.config[FEATURIZERS]
            )

            entity_tags_data = self._create_data_for_entities(entity_tags)
            if entity_tags_data is not None:
                model_data.add_data(entity_tags_data)
        else:
            # method is called during prediction
            attribute_data, _ = convert_to_data_format(
                tracker_state_features,
                self.fake_features,
                featurizers=self.config[FEATURIZERS],
            )

        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)
        model_data.add_lengths(ACTION_TEXT, SEQUENCE_LENGTH, ACTION_TEXT, SEQUENCE)

        # add the dialogue lengths
        attribute_present = next(iter(list(attribute_data.keys())))
        dialogue_lengths = np.array(
            [
                np.size(np.squeeze(f, -1))
                for f in model_data.data[attribute_present][MASK][0]
            ]
        )
        model_data.data[DIALOGUE][LENGTH] = [
            FeatureArray(dialogue_lengths, number_of_dimensions=1)
        ]

        # make sure all keys are in the same order during training and prediction
        model_data.sort()

        return model_data

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Train the policy on given training trackers."""

        if not training_trackers:
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # dealing with training data
        tracker_state_features, label_ids, entity_tags = self.featurize_for_training(
            training_trackers, domain, interpreter, **kwargs
        )

        self._label_data, encoded_all_labels = self._create_label_data(
            domain, interpreter
        )

        # extract actual training data to feed to model
        model_data = self._create_model_data(
            tracker_state_features, label_ids, entity_tags, encoded_all_labels
        )
        if model_data.is_empty():
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        if self.config[ENTITY_RECOGNITION]:
            self._entity_tag_specs = self._create_entity_tag_specs()

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        if not self.finetune_mode:
            # This means the model wasn't loaded from a
            # previously trained model and hence needs
            # to be instantiated.
            self.model = TED(
                model_data.get_signature(),
                self.config,
                isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
                self._label_data,
                self._entity_tag_specs,
            )

        self.model.fit(
            model_data,
            self.config[EPOCHS],
            self.config[BATCH_SIZES],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[EVAL_NUM_EPOCHS],
            batch_strategy=self.config[BATCH_STRATEGY],
        )

    def _featurize_tracker_for_e2e(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        # construct two examples in the batch to be fed to the model -
        # one by featurizing last user text
        # and second - an optional one (see conditions below),
        # the first example in the constructed batch either does not contain user input
        # or uses intent or text based on whether TED is e2e only.
        tracker_state_features = self.featurizer.create_state_features(
            [tracker], domain, interpreter, use_text_for_last_user_input=self.only_e2e
        )
        # the second - text, but only after user utterance and if not only e2e
        if (
            tracker.latest_action_name == ACTION_LISTEN_NAME
            and TEXT in self.fake_features
            and not self.only_e2e
        ):
            tracker_state_features += self.featurizer.create_state_features(
                [tracker], domain, interpreter, use_text_for_last_user_input=True
            )
        return tracker_state_features

    def _pick_confidence(
        self, confidences: np.ndarray, similarities: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        # the confidences and similarities have shape (batch-size x number of actions)
        # batch-size can only be 1 or 2;
        # in the case batch-size==2, the first example contain user intent as features,
        # the second - user text as features
        if confidences.shape[0] > 2:
            raise ValueError(
                "We cannot pick prediction from batches of size more than 2."
            )
        # we use heuristic to pick correct prediction
        if confidences.shape[0] == 2:
            # we use similarities to pick appropriate input,
            # since it seems to be more accurate measure,
            # policy is trained to maximize the similarity not the confidence
            if (
                np.max(confidences[1]) > self.config[E2E_CONFIDENCE_THRESHOLD]
                # TODO maybe compare confidences is better
                and np.max(similarities[1]) > np.max(similarities[0])
            ):
                return confidences[1], True

            return confidences[0], False

        # by default the first example in a batch is the one to use for prediction
        return confidences[0], self.only_e2e

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take.

        See the docstring of the parent class `Policy` for more information.
        """
        if self.model is None:
            return self._prediction(self._default_predictions(domain))

        # create model data from tracker
        tracker_state_features = self._featurize_tracker_for_e2e(
            tracker, domain, interpreter
        )
        model_data = self._create_model_data(tracker_state_features)

        output = self.model.predict(model_data)

        # take the last prediction in the sequence
        similarities = output["similarities"].numpy()[:, -1, :]
        confidences = output["action_scores"].numpy()[:, -1, :]
        # take correct prediction from batch
        confidence, is_e2e_prediction = self._pick_confidence(confidences, similarities)

        if self.config[LOSS_TYPE] == SOFTMAX and self.config[RANKING_LENGTH] > 0:
            confidence = rasa.utils.train_utils.normalize(
                confidence, self.config[RANKING_LENGTH]
            )

        optional_events = self._create_optional_event_for_entities(
            output, is_e2e_prediction, interpreter, tracker
        )

        return self._prediction(
            confidence.tolist(),
            is_end_to_end_prediction=is_e2e_prediction,
            optional_events=optional_events,
        )

    def _create_optional_event_for_entities(
        self,
        prediction_output: Dict[Text, tf.Tensor],
        is_e2e_prediction: bool,
        interpreter: NaturalLanguageInterpreter,
        tracker: DialogueStateTracker,
    ) -> Optional[List[Event]]:
        if tracker.latest_action_name != ACTION_LISTEN_NAME or not is_e2e_prediction:
            # entities belong only to the last user message
            # and only if user text was used for prediction,
            # a user message always comes after action listen
            return

        if not self.config[ENTITY_RECOGNITION]:
            # entity recognition is not turned on, no entities can be predicted
            return

        # The batch dimension of entity prediction is not the same as batch size,
        # rather it is the number of last (if max history featurizer else all)
        # text inputs in the batch
        # therefore, in order to pick entities from the latest user message
        # we need to pick entities from the last batch dimension of entity prediction
        (
            predicted_tags,
            confidence_values,
        ) = rasa.utils.train_utils.entity_label_to_tags(
            prediction_output, self._entity_tag_specs, prediction_index=-1
        )

        if ENTITY_ATTRIBUTE_TYPE not in predicted_tags:
            # no entities detected
            return

        # entities belong to the last message of the tracker
        # convert the predicted tags to actual entities
        text = tracker.latest_message.text
        parsed_message = interpreter.featurize_message(Message(data={TEXT: text}))
        tokens = parsed_message.get(TOKENS_NAMES[TEXT])
        entities = EntityExtractor.convert_predictions_into_entities(
            text, tokens, predicted_tags, confidences=confidence_values
        )

        # add the extractor name
        for entity in entities:
            entity[EXTRACTOR] = "TEDPolicy"

        return [EntitiesAdded(entities)]

    def persist(self, path: Union[Text, Path]) -> None:
        """Persists the policy to a storage."""
        if self.model is None:
            logger.debug(
                "Method `persist(...)` was called without a trained model present. "
                "Nothing to persist then!"
            )
            return

        model_path = Path(path)
        tf_model_file = model_path / f"{SAVE_MODEL_FILE_NAME}.tf_model"

        rasa.shared.utils.io.create_directory_for_file(tf_model_file)

        self.featurizer.persist(path)

        if self.model.checkpoint_model:
            self.model.copy_best(str(tf_model_file))
        else:
            self.model.save(str(tf_model_file))

        io_utils.json_pickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.priority.pkl", self.priority
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.meta.pkl", self.config
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl", self.data_example
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.fake_features.pkl",
            self.fake_features,
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.label_data.pkl",
            dict(self._label_data.data),
        )

        entity_tag_specs = (
            [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
            if self._entity_tag_specs
            else []
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{SAVE_MODEL_FILE_NAME}.entity_tag_specs.json",
            entity_tag_specs,
        )

    @classmethod
    def load(
        cls,
        path: Union[Text, Path],
        should_finetune: bool = False,
        epoch_override: int = defaults[EPOCHS],
        **kwargs: Any,
    ) -> "TEDPolicy":
        """Loads a policy from the storage.

        **Needs to load its featurizer**
        """
        model_path = Path(path)

        if not model_path.exists():
            raise Exception(
                f"Failed to load TED policy model. Path "
                f"'{model_path.absolute()}' doesn't exist."
            )

        tf_model_file = model_path / f"{SAVE_MODEL_FILE_NAME}.tf_model"

        featurizer = TrackerFeaturizer.load(path)

        if not (model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl").is_file():
            return cls(featurizer=featurizer)

        loaded_data = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl"
        )
        label_data = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.label_data.pkl"
        )
        fake_features = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.fake_features.pkl"
        )
        label_data = RasaModelData(data=label_data)
        meta = io_utils.pickle_load(model_path / f"{SAVE_MODEL_FILE_NAME}.meta.pkl")
        priority = io_utils.json_unpickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.priority.pkl"
        )
        entity_tag_specs = rasa.shared.utils.io.read_json_file(
            model_path / f"{SAVE_MODEL_FILE_NAME}.entity_tag_specs.json"
        )
        entity_tag_specs = [
            EntityTagSpec(
                tag_name=tag_spec["tag_name"],
                ids_to_tags={
                    int(key): value for key, value in tag_spec["ids_to_tags"].items()
                },
                tags_to_ids={
                    key: int(value) for key, value in tag_spec["tags_to_ids"].items()
                },
                num_tags=tag_spec["num_tags"],
            )
            for tag_spec in entity_tag_specs
        ]

        model_data_example = RasaModelData(
            label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY, data=loaded_data
        )
        meta = rasa.utils.train_utils.update_similarity_type(meta)

        meta[EPOCHS] = epoch_override

        model = TED.load(
            str(tf_model_file),
            model_data_example,
            data_signature=model_data_example.get_signature(),
            config=meta,
            # during prediction we don't care about previous dialogue turns,
            # so to save computation time, use only the last one
            use_only_last_dialogue_turns=True,
            label_data=label_data,
            entity_tag_specs=entity_tag_specs,
            finetune_mode=should_finetune,
        )

        if not should_finetune:
            # build the graph for prediction
            predict_data_example = RasaModelData(
                label_key=LABEL_KEY,
                label_sub_key=LABEL_SUB_KEY,
                data={
                    feature_name: features
                    for feature_name, features in model_data_example.items()
                    if feature_name
                    # we need to remove label features for prediction if they are present
                    in PREDICTION_FEATURES
                },
            )
            model.build_for_predict(predict_data_example)

        return cls(
            featurizer=featurizer,
            priority=priority,
            model=model,
            fake_features=fake_features,
            entity_tag_specs=entity_tag_specs,
            should_finetune=should_finetune,
            **meta,
        )


class TED(TransformerRasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        config: Dict[Text, Any],
        use_only_last_dialogue_turns: bool,
        label_data: RasaModelData,
        entity_tag_specs: Optional[List[EntityTagSpec]],
    ) -> None:
        """Intializes the TED model.

        Args:
            data_signature: the data signature of the input data
            config: the model configuration
            use_only_last_dialogue_turns: if 'True' only the last dialogue turn will be used
            label_data: the label data
            entity_tag_specs: the entity tag specifications
        """
        super().__init__("TED", config, data_signature, label_data)

        self.use_only_last_dialogue_turns = use_only_last_dialogue_turns

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if feature_name in PREDICTION_FEATURES
        }

        self._entity_tag_specs = entity_tag_specs

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # metrics
        self.action_loss = tf.keras.metrics.Mean(name="loss")
        self.action_acc = tf.keras.metrics.Mean(name="acc")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")
        self.metrics_to_log += ["loss", "acc"]
        if self.config[ENTITY_RECOGNITION]:
            self.metrics_to_log += ["e_loss", "e_f1"]

        # needed for efficient prediction
        self.all_labels_embed: Optional[tf.Tensor] = None

        self._prepare_layers()

    def _check_data(self) -> None:
        if not any(key in [INTENT, TEXT] for key in self.data_signature.keys()):
            raise ValueError(
                f"No user features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

        if not any(
            key in [ACTION_NAME, ACTION_TEXT] for key in self.data_signature.keys()
        ):
            raise ValueError(
                f"No action features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL not in self.data_signature:
            raise ValueError(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    # ---CREATING LAYERS HELPERS---

    def _prepare_layers(self) -> None:
        for name in self.data_signature.keys():
            self._prepare_sparse_dense_layer_for(name, self.data_signature)
            if name in SEQUENCE_FEATURES_TO_ENCODE:
                self._prepare_sequence_layers(name)
            self._prepare_encoding_layers(name)

        for name in self.label_signature.keys():
            self._prepare_sparse_dense_layer_for(name, self.label_signature)
            if name in SEQUENCE_FEATURES_TO_ENCODE:
                self._prepare_sequence_layers(name)
            self._prepare_encoding_layers(name)

        self._prepare_transformer_layer(
            DIALOGUE,
            self.config[NUM_TRANSFORMER_LAYERS][DIALOGUE],
            self.config[TRANSFORMER_SIZE][DIALOGUE],
            self.config[DROP_RATE_DIALOGUE],
            self.config[DROP_RATE_ATTENTION],
        )

        self._prepare_embed_layers(DIALOGUE)
        self._prepare_embed_layers(LABEL)

        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_sparse_dense_layer_for(
        self, name: Text, signature: Dict[Text, Dict[Text, List[FeatureSignature]]]
    ) -> None:
        """Prepares the sparse dense layer for the given attribute name.

        It is used to combine the sparse and dense features of the attribute at the
        beginning of the model.

        Args:
            name: the attribute name
            signature: data signature
        """
        for feature_type in VALID_FEATURE_TYPES:
            if feature_type not in signature[name]:
                # features for feature type are not present
                continue

            self._prepare_sparse_dense_dropout_layers(
                f"{name}_{feature_type}", self.config[DROP_RATE]
            )

            # use the same configurable dense dimension for all sparse features
            self._prepare_sparse_dense_layers(
                signature[name][feature_type],
                f"{name}_{feature_type}",
                self.config[DENSE_DIMENSION][name],
            )

    def _prepare_encoding_layers(self, name: Text) -> None:
        """Create ffnn layer for given attribute name. The layer is used just before
        all dialogue features are combined.

        Args:
            name: attribute name
        """
        # create encoding layers only for the features which should be encoded;
        if name not in SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE:
            return
        # check that there are SENTENCE features for the attribute name in data
        if (
            name in SENTENCE_FEATURES_TO_ENCODE
            and FEATURE_TYPE_SENTENCE not in self.data_signature[name]
        ):
            return
        #  same for label_data
        if (
            name in LABEL_FEATURES_TO_ENCODE
            and FEATURE_TYPE_SENTENCE not in self.label_signature[name]
        ):
            return

        self._prepare_ffnn_layer(
            f"{name}",
            [self.config[ENCODING_DIMENSION]],
            self.config[DROP_RATE_DIALOGUE],
            prefix="encoding_layer",
        )

    # ---GRAPH BUILDING HELPERS---

    @staticmethod
    def _compute_dialogue_indices(
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> None:
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], dtype=tf.int32)
        # wrap in a list, because that's the structure of tf_batch_data
        tf_batch_data[DIALOGUE][INDICES] = [
            (
                tf.map_fn(
                    tf.range,
                    dialogue_lengths,
                    fn_output_signature=tf.RaggedTensorSpec(
                        shape=[None], dtype=tf.int32
                    ),
                )
            ).values
        ]

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]
        # labels cannot have all features "fake"
        all_labels_encoded = {}
        for key in self.tf_label_data.keys():
            if key != LABEL_KEY:
                attribute_features, _, _ = self._encode_real_features_per_attribute(
                    self.tf_label_data, key
                )
                all_labels_encoded[key] = attribute_features

        if (
            all_labels_encoded.get(f"{LABEL_KEY}_{ACTION_TEXT}") is not None
            and all_labels_encoded.get(f"{LABEL_KEY}_{ACTION_NAME}") is not None
        ):
            x = all_labels_encoded.pop(
                f"{LABEL_KEY}_{ACTION_TEXT}"
            ) + all_labels_encoded.pop(f"{LABEL_KEY}_{ACTION_NAME}")
        elif all_labels_encoded.get(f"{LABEL_KEY}_{ACTION_TEXT}") is not None:
            x = all_labels_encoded.pop(f"{LABEL_KEY}_{ACTION_TEXT}")
        else:
            x = all_labels_encoded.pop(f"{LABEL_KEY}_{ACTION_NAME}")

        # additional sequence axis is artifact of our RasaModelData creation
        # TODO check whether this should be solved in data creation
        x = tf.squeeze(x, axis=1)

        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    def _embed_dialogue(
        self,
        dialogue_in: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Create dialogue level embedding and mask."""
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], tf.int32)
        mask = self._compute_mask(dialogue_lengths)

        dialogue_transformed = self._tf_layers[f"transformer.{DIALOGUE}"](
            dialogue_in, 1 - mask, self._training
        )
        dialogue_transformed = tfa.activations.gelu(dialogue_transformed)

        if self.use_only_last_dialogue_turns:
            # pick last vector if max history featurizer is used
            dialogue_transformed = tf.expand_dims(
                self._last_token(dialogue_transformed, dialogue_lengths), 1
            )
            mask = tf.expand_dims(self._last_token(mask, dialogue_lengths), 1)

        dialogue_embed = self._tf_layers[f"embed.{DIALOGUE}"](dialogue_transformed)

        return dialogue_embed, mask, dialogue_transformed

    def _encode_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # The input is a representation of 4d tensor of
        # shape (batch-size x dialogue-len x sequence-len x units) in 3d of shape
        # (sum of dialogue history length for all tensors in the batch x
        # max sequence length x number of features).

        # However, some dialogue turns contain non existent state features,
        # e.g. `intent` and `text` features are mutually exclusive,
        # as well as `action_name` and `action_text` are mutually exclusive,
        # or some dialogue turns don't contain any `slots`.
        # In order to create 4d full tensors, we created "fake" zero features for
        # these non existent state features. And filtered them during batch generation.
        # Therefore the first dimensions for different attributes are different.
        # It could happen that some batches don't contain "real" features at all,
        # e.g. large number of stories don't contain any `slots`.
        # Therefore actual input tensors will be empty.
        # Since we need actual numbers to create dialogue turn features, we create
        # zero tensors in `_encode_fake_features_per_attribute` for these attributes.
        return tf.cond(
            tf.shape(tf_batch_data[attribute][SENTENCE][0])[0] > 0,
            lambda: self._encode_real_features_per_attribute(tf_batch_data, attribute),
            lambda: self._encode_fake_features_per_attribute(tf_batch_data, attribute),
        )

    def _get_dense_units(
        self, attribute_features_list: List[tf.Tensor], attribute: Text
    ) -> int:
        # TODO this should be done in corresponding layers once in init
        units = 0
        for f in attribute_features_list:
            if isinstance(f, tf.SparseTensor):
                units += self.config[DENSE_DIMENSION][attribute]
            else:
                units += f.shape[-1]
        return units

    def _get_concat_units(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> int:
        # TODO this should be done in corresponding layers once in init
        # calculate concat sequence sentence dim
        sentence_units = self._get_dense_units(
            tf_batch_data[attribute][SENTENCE], attribute
        )
        sequence_units = self._get_dense_units(
            tf_batch_data[attribute][SEQUENCE], attribute
        )

        if sequence_units and not sentence_units:
            return sequence_units

        if sentence_units and not sequence_units:
            return sentence_units

        if sentence_units != sequence_units:
            return self.config[CONCAT_DIMENSION][TEXT]

        return sentence_units

    def _encode_fake_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # we need to create real zero tensors with appropriate batch and dialogue dim
        # because they are passed to dialogue transformer
        attribute_mask = tf_batch_data[attribute][MASK][0]

        batch_dim = tf.shape(attribute_mask)[0]
        dialogue_dim = tf.shape(attribute_mask)[1]
        if attribute in set(SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE):
            units = self.config[ENCODING_DIMENSION]
        else:
            units = self._get_dense_units(tf_batch_data[attribute][SENTENCE], attribute)

        attribute_features = tf.zeros(
            (batch_dim, dialogue_dim, units), dtype=tf.float32
        )
        if attribute == TEXT:
            # if the input features are fake, we don't process them further,
            # but we need to calculate correct last dim (units) so that tf could infer
            # the last shape of the tensors
            if self.config[NUM_TRANSFORMER_LAYERS][TEXT] > 0:
                text_transformer_units = self.config[TRANSFORMER_SIZE][TEXT]
            elif self.config[HIDDEN_LAYERS_SIZES][TEXT]:
                text_transformer_units = self.config[HIDDEN_LAYERS_SIZES][TEXT][-1]
            else:
                text_transformer_units = self._get_concat_units(
                    tf_batch_data, attribute
                )

            text_transformer_output = tf.zeros(
                (0, 0, text_transformer_units), dtype=tf.float32
            )
            text_sequence_lengths = tf.zeros((0, 1), dtype=tf.int32)
        else:
            # simulate None with empty tensor of zeros
            text_transformer_output = tf.zeros((0,))
            text_sequence_lengths = tf.zeros((0,))

        return attribute_features, text_transformer_output, text_sequence_lengths

    @staticmethod
    def _create_last_dialogue_turns_mask(
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> tf.Tensor:
        # Since use_only_last_dialogue_turns is True,
        # we need to find the locations of last dialogue turns in
        # (combined batch dimension and dialogue length,) dimension,
        # so that we can use `_sequence_lengths` as a boolean  mask to pick
        # which ones are "real" textual input in these last dialogue turns.

        # In order to do that we can use given `dialogue_lengths`.
        # For example:
        # If we have `dialogue_lengths = [2, 1, 3]`, than
        # `dialogue_indices = [0, 1, 0, 0, 1, 2]` here we can spot that `0`
        # always indicates the first dialogue turn,
        # which means that previous dialogue turn is the last dialogue turn.
        # Combining this with the fact that the last element in
        # `dialogue_indices` is always the last dialogue turn, we can add
        # a `0` to the end, getting
        # `_dialogue_indices = [0, 1, 0, 0, 1, 2, 0]`.
        # Then removing the first element
        # `_last_dialogue_turn_inverse_indicator = [1, 0, 0, 1, 2, 0]`
        # we see that `0` points to the last dialogue turn.
        # We convert all positive numbers to `True` and take
        # the inverse mask to get
        # `last_dialogue_mask = [0, 1, 1, 0, 0, 1],
        # which precisely corresponds to the fact that first dialogue is of
        # length 2, the second 1 and the third 3.
        last_dialogue_turn_mask = tf.math.logical_not(
            tf.cast(
                tf.concat(
                    [
                        tf_batch_data[DIALOGUE][INDICES][0],
                        tf.zeros((1,), dtype=tf.int32),
                    ],
                    axis=0,
                )[1:],
                dtype=tf.bool,
            )
        )
        # get only the indices of real inputs
        return tf.boolean_mask(
            last_dialogue_turn_mask,
            tf.reshape(tf_batch_data[attribute][SEQUENCE_LENGTH][0], (-1,)),
        )

    def _encode_real_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Encodes features for a given attribute.

        Args:
            tf_batch_data: dictionary mapping every attribute to its features and masks
            attribute: the attribute we will encode features for
            (e.g., ACTION_NAME, INTENT)

        Returns:
            A tensor combining  all features for `attribute`
        """
        # simulate None with empty tensor of zeros
        text_transformer_output = tf.zeros((0,))
        text_sequence_lengths = tf.zeros((0,))

        if attribute in SEQUENCE_FEATURES_TO_ENCODE:
            # sequence_lengths contain `0` for "fake" features, while
            # tf_batch_data[attribute] contain only "real" features
            sequence_lengths = tf_batch_data[attribute][SEQUENCE_LENGTH][0]
            # extract only nonzero lengths and cast to int
            sequence_lengths = tf.cast(
                tf.boolean_mask(sequence_lengths, sequence_lengths), dtype=tf.int32
            )
            # boolean mask returns flat tensor
            sequence_lengths = tf.expand_dims(sequence_lengths, axis=-1)

            mask_sequence_text = tf.squeeze(
                self._compute_mask(sequence_lengths), axis=1
            )
            # add 1 to sequence lengths to account for sentence features
            sequence_lengths += 1
            mask_text = tf.squeeze(self._compute_mask(sequence_lengths), axis=1)

            attribute_features, _, _, _ = self._create_sequence(
                tf_batch_data[attribute][SEQUENCE],
                tf_batch_data[attribute][SENTENCE],
                mask_sequence_text,
                mask_text,
                attribute,
                sparse_dropout=self.config[SPARSE_INPUT_DROPOUT],
                dense_dropout=self.config[DENSE_INPUT_DROPOUT],
                masked_lm_loss=self.config[MASKED_LM],
                sequence_ids=False,
            )

            if attribute == TEXT:
                text_transformer_output = attribute_features
                text_sequence_lengths = sequence_lengths

                if self.use_only_last_dialogue_turns:
                    # get the location of all last dialogue inputs
                    last_dialogue_turns_mask = self._create_last_dialogue_turns_mask(
                        tf_batch_data, attribute
                    )
                    # pick outputs that correspond to the last dialogue turns
                    text_transformer_output = tf.boolean_mask(
                        text_transformer_output, last_dialogue_turns_mask
                    )
                    text_sequence_lengths = tf.boolean_mask(
                        text_sequence_lengths, last_dialogue_turns_mask
                    )

            # resulting attribute features will have shape
            # combined batch dimension and dialogue length x 1 x units
            attribute_features = tf.expand_dims(
                self._last_token(
                    attribute_features, tf.squeeze(sequence_lengths, axis=-1)
                ),
                axis=1,
            )

        else:
            # resulting attribute features will have shape
            # combined batch dimension and dialogue length x 1 x units
            attribute_features = self._combine_sparse_dense_features(
                tf_batch_data[attribute][SENTENCE], f"{attribute}_{SENTENCE}"
            )

        if attribute in SENTENCE_FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE:
            attribute_features = self._tf_layers[f"encoding_layer.{attribute}"](
                attribute_features
            )

        # attribute features have shape
        # (combined batch dimension and dialogue length x 1 x units)
        # convert them back to their original shape of
        # batch size x dialogue length x units
        attribute_features = self._convert_to_original_shape(
            attribute_features, tf_batch_data, attribute
        )

        return attribute_features, text_transformer_output, text_sequence_lengths

    @staticmethod
    def _convert_to_original_shape(
        attribute_features: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        attribute: Text,
    ) -> tf.Tensor:
        """Transform attribute features back to original shape.

        Given shape: (combined batch and dialogue dimension x 1 x units)
        Original shape: (batch x dialogue length x units)

        Args:
            attribute_features: the "real" features to convert
            attribute_mask:  the tensor containing the position of "real" features
                in the dialogue, shape is (batch-size x dialogue_len x 1)
            dialogue_lengths: the tensor containing the actual dialogue length,
                shape is (batch-size,)

        Returns:
            The converted attribute features
        """
        # in order to convert the attribute features with shape
        # (combined batch-size and dialogue length x 1 x units)
        # to a shape of (batch-size x dialogue length x units)
        # we use tf.scatter_nd. Therefore, we need the target shape and the indices
        # mapping the values of attribute features to the position in the resulting
        # tensor.

        # attribute_mask has shape batch x dialogue_len x 1
        attribute_mask = tf_batch_data[attribute][MASK][0]

        if attribute in SENTENCE_FEATURES_TO_ENCODE + STATE_LEVEL_FEATURES:
            dialogue_lengths = tf.cast(
                tf_batch_data[DIALOGUE][LENGTH][0], dtype=tf.int32
            )
            dialogue_indices = tf_batch_data[DIALOGUE][INDICES][0]
        else:
            # for labels, dialogue length is a fake dim and equal to 1
            dialogue_lengths = tf.ones((tf.shape(attribute_mask)[0],), dtype=tf.int32)
            dialogue_indices = tf.zeros((tf.shape(attribute_mask)[0],), dtype=tf.int32)

        batch_dim = tf.shape(attribute_mask)[0]
        dialogue_dim = tf.shape(attribute_mask)[1]
        units = attribute_features.shape[-1]

        # attribute_mask has shape (batch x dialogue_len x 1), remove last dimension
        attribute_mask = tf.cast(tf.squeeze(attribute_mask, axis=-1), dtype=tf.int32)
        # sum of attribute mask contains number of dialogue turns with "real" features
        non_fake_dialogue_lengths = tf.reduce_sum(attribute_mask, axis=-1)
        # create the batch indices
        batch_indices = tf.repeat(tf.range(batch_dim), non_fake_dialogue_lengths)

        # attribute_mask has shape (batch x dialogue_len x 1), while
        # dialogue_indices has shape (combined_dialogue_len,)
        # in order to find positions of real input we need to flatten
        # attribute mask to (combined_dialogue_len,)
        dialogue_indices_mask = tf.boolean_mask(
            attribute_mask, tf.sequence_mask(dialogue_lengths, dtype=tf.int32)
        )
        # pick only those indices that contain "real" input
        dialogue_indices = tf.boolean_mask(dialogue_indices, dialogue_indices_mask)

        indices = tf.stack([batch_indices, dialogue_indices], axis=1)

        shape = tf.convert_to_tensor([batch_dim, dialogue_dim, units])
        attribute_features = tf.squeeze(attribute_features, axis=1)

        return tf.scatter_nd(indices, attribute_features, shape)

    def _process_batch_data(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        """Encodes batch data.

        Combines intent and text and action name and action text if both are present.

        Args:
            tf_batch_data: dictionary mapping every attribute to its features and masks

        Returns:
             Tensor: encoding of all features in the batch, combined;
        """
        # encode each attribute present in tf_batch_data
        text_transformer_output = None
        text_sequence_lengths = None
        batch_encoded = {}
        for attribute in tf_batch_data.keys():
            if attribute in SENTENCE_FEATURES_TO_ENCODE + STATE_LEVEL_FEATURES:
                (
                    attribute_features,
                    _text_transformer_output,
                    _text_sequence_lengths,
                ) = self._encode_features_per_attribute(tf_batch_data, attribute)

                batch_encoded[attribute] = attribute_features
                if attribute == TEXT:
                    text_transformer_output = _text_transformer_output
                    text_sequence_lengths = _text_sequence_lengths

        # if both action text and action name are present, combine them; otherwise,
        # return the one which is present

        if (
            batch_encoded.get(ACTION_TEXT) is not None
            and batch_encoded.get(ACTION_NAME) is not None
        ):
            batch_action = batch_encoded.pop(ACTION_TEXT) + batch_encoded.pop(
                ACTION_NAME
            )
        elif batch_encoded.get(ACTION_TEXT) is not None:
            batch_action = batch_encoded.pop(ACTION_TEXT)
        else:
            batch_action = batch_encoded.pop(ACTION_NAME)
        # same for user input
        if (
            batch_encoded.get(INTENT) is not None
            and batch_encoded.get(TEXT) is not None
        ):
            batch_user = batch_encoded.pop(INTENT) + batch_encoded.pop(TEXT)
        elif batch_encoded.get(TEXT) is not None:
            batch_user = batch_encoded.pop(TEXT)
        else:
            batch_user = batch_encoded.pop(INTENT)

        batch_features = [batch_user, batch_action]
        # once we have user input and previous action,
        # add all other attributes (SLOTS, ACTIVE_LOOP, etc.) to batch_features;
        for key in batch_encoded.keys():
            batch_features.append(batch_encoded.get(key))

        batch_features = tf.concat(batch_features, axis=-1)

        return batch_features, text_transformer_output, text_sequence_lengths

    def _reshape_for_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_transformer_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # The first dim of the output of the text sequence transformer is the same
        # as number of "real" features for `text` at the last dialogue turns
        # (let's call it `N`),
        # which corresponds to the first dim of the tag ids tensor.
        # To calculate the loss for entities we need the output of the text
        # sequence transformer (shape: N x sequence length x units),
        # the output of the dialogue transformer
        # (shape: batch size x dialogue length x units) and the tag ids for the
        # entities (shape: N x sequence length - 1 x units)
        # In order to process the tensors, they need to have the same shape.
        # Convert the output of the dialogue transformer to shape
        # (N x 1 x units).

        # Note: The CRF layer cannot handle 4D tensors. E.g. we cannot use the shape
        # batch size x dialogue length x sequence length x units

        # convert the output of the dialogue transformer
        # to shape (real entity dim x 1 x units)
        attribute_mask = tf_batch_data[TEXT][MASK][0]
        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], tf.int32)

        if self.use_only_last_dialogue_turns:
            # pick outputs that correspond to the last dialogue turns
            attribute_mask = tf.expand_dims(
                self._last_token(attribute_mask, dialogue_lengths), axis=1
            )
        dialogue_transformer_output = tf.boolean_mask(
            dialogue_transformer_output, tf.squeeze(attribute_mask, axis=-1)
        )

        # boolean mask removed axis=1, add it back
        dialogue_transformer_output = tf.expand_dims(
            dialogue_transformer_output, axis=1
        )

        # broadcast the dialogue transformer output sequence-length-times to get the
        # same shape as the text sequence transformer output
        dialogue_transformer_output = tf.tile(
            dialogue_transformer_output, (1, tf.shape(text_transformer_output)[1], 1)
        )

        # concat the output of the dialogue transformer to the output of the text
        # sequence transformer (adding context)
        # resulting shape (N x sequence length x 2 units)
        # N = number of "real" features for `text` at the last dialogue turns
        text_transformed = tf.concat(
            [text_transformer_output, dialogue_transformer_output], axis=-1
        )

        text_mask = tf.squeeze(self._compute_mask(text_sequence_lengths), axis=1)
        # add zeros to match the shape of text_transformed, because
        # max sequence length might differ, since it is calculated dynamically
        # based on a subset of sequence lengths
        sequence_diff = tf.shape(text_transformed)[1] - tf.shape(text_mask)[1]
        text_mask = tf.pad(text_mask, [[0, 0], [0, sequence_diff], [0, 0]])

        # remove additional dims and sentence features
        text_sequence_lengths = tf.reshape(text_sequence_lengths, (-1,)) - 1

        return text_transformed, text_mask, text_sequence_lengths

    # ---TRAINING---

    def _batch_loss_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_transformer_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> tf.Tensor:
        # It could happen that some batches don't contain "real" features for `text`,
        # e.g. large number of stories are intent only.
        # Therefore actual `text_transformer_output` will be empty.
        # We cannot create a loss with empty tensors.
        # Since we need actual numbers to create a full loss, we output
        # zero in this case.
        return tf.cond(
            tf.shape(text_transformer_output)[0] > 0,
            lambda: self._real_batch_loss_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_transformer_output,
                text_sequence_lengths,
            ),
            lambda: tf.constant(0.0),
        )

    def _real_batch_loss_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_transformer_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> tf.Tensor:

        text_transformed, text_mask, text_sequence_lengths = self._reshape_for_entities(
            tf_batch_data,
            dialogue_transformer_output,
            text_transformer_output,
            text_sequence_lengths,
        )

        tag_ids = tf_batch_data[ENTITY_TAGS][IDS][0]
        # add a zero (no entity) for the sentence features to match the shape of inputs
        sequence_diff = tf.shape(text_transformed)[1] - tf.shape(tag_ids)[1]
        tag_ids = tf.pad(tag_ids, [[0, 0], [0, sequence_diff], [0, 0]])

        loss, f1, _ = self._calculate_entity_loss(
            text_transformed,
            tag_ids,
            text_mask,
            text_sequence_lengths,
            ENTITY_ATTRIBUTE_TYPE,
        )

        self.entity_loss.update_state(loss)
        self.entity_f1.update_state(f1)

        return loss

    @staticmethod
    def _get_labels_embed(
        label_ids: tf.Tensor, all_labels_embed: tf.Tensor
    ) -> tf.Tensor:
        # instead of processing labels again, gather embeddings from
        # all_labels_embed using label ids

        indices = tf.cast(label_ids[:, :, 0], tf.int32)
        labels_embed = tf.gather(all_labels_embed, indices)

        return labels_embed

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)
        self._compute_dialogue_indices(tf_batch_data)

        all_label_ids, all_labels_embed = self._create_all_labels_embed()

        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        labels_embed = self._get_labels_embed(label_ids, all_labels_embed)

        (
            dialogue_in,
            text_transformer_output,
            text_sequence_lengths,
        ) = self._process_batch_data(tf_batch_data)
        (
            dialogue_embed,
            dialogue_mask,
            dialogue_transformer_output,
        ) = self._embed_dialogue(dialogue_in, tf_batch_data)
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        losses = []

        loss, acc = self._tf_layers[f"loss.{LABEL}"](
            dialogue_embed,
            labels_embed,
            label_ids,
            all_labels_embed,
            all_label_ids,
            dialogue_mask,
        )
        losses.append(loss)

        if (
            self.config[ENTITY_RECOGNITION]
            and text_transformer_output is not None
            and text_sequence_lengths is not None
        ):
            losses.append(
                self._batch_loss_entities(
                    tf_batch_data,
                    dialogue_transformer_output,
                    text_transformer_output,
                    text_sequence_lengths,
                )
            )

        self.action_loss.update_state(loss)
        self.action_acc.update_state(acc)

        return tf.math.add_n(losses)

    # ---PREDICTION---

    def prepare_for_predict(self) -> None:
        """Prepares the model for prediction."""
        _, self.all_labels_embed = self._create_all_labels_embed()

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        if self.all_labels_embed is None:
            raise ValueError(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )
        self._compute_dialogue_indices(tf_batch_data)

        (
            dialogue_in,
            text_transformer_output,
            text_sequence_lengths,
        ) = self._process_batch_data(tf_batch_data)
        (
            dialogue_embed,
            dialogue_mask,
            dialogue_transformer_output,
        ) = self._embed_dialogue(dialogue_in, tf_batch_data)
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        sim_all = self._tf_layers[f"loss.{LABEL}"].sim(
            dialogue_embed[:, :, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            dialogue_mask,
        )

        scores = self._tf_layers[f"loss.{LABEL}"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )
        predictions = {"action_scores": scores, "similarities": sim_all}

        if (
            self.config[ENTITY_RECOGNITION]
            and text_transformer_output is not None
            and text_sequence_lengths is not None
        ):
            pred_ids, confidences = self._batch_predict_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_transformer_output,
                text_sequence_lengths,
            )
            name = ENTITY_ATTRIBUTE_TYPE
            predictions[f"e_{name}_ids"] = pred_ids
            predictions[f"e_{name}_scores"] = confidences

        return predictions

    def _batch_predict_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_transformer_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # It could happen that current prediction turn don't contain
        # "real" features for `text`,
        # Therefore actual `text_transformer_output` will be empty.
        # We cannot predict entities with empty tensors.
        # Since we need to output some tensors of the same shape, we output
        # zero tensors.
        return tf.cond(
            tf.shape(text_transformer_output)[0] > 0,
            lambda: self._real_batch_predict_entities(
                tf_batch_data,
                dialogue_transformer_output,
                text_transformer_output,
                text_sequence_lengths,
            ),
            lambda: (
                # the output is of shape (batch_size, max_seq_len)
                tf.zeros(tf.shape(text_transformer_output)[:2], dtype=tf.int32),
                tf.zeros(tf.shape(text_transformer_output)[:2], dtype=tf.float32),
            ),
        )

    def _real_batch_predict_entities(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        dialogue_transformer_output: tf.Tensor,
        text_transformer_output: tf.Tensor,
        text_sequence_lengths: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        text_transformed, _, text_sequence_lengths = self._reshape_for_entities(
            tf_batch_data,
            dialogue_transformer_output,
            text_transformer_output,
            text_sequence_lengths,
        )

        name = ENTITY_ATTRIBUTE_TYPE

        _logits = self._tf_layers[f"embed.{name}.logits"](text_transformed)

        return self._tf_layers[f"crf.{name}"](_logits, text_sequence_lengths)


# pytype: enable=key-error
