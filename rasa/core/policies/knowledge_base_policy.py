import copy
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy import sparse

from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io as io_utils
from rasa.core.knowledge_base.converter.sql_converter import SQLConverter
from rasa.core.knowledge_base.grammar.grammar import Grammar, GrammarRule
from rasa.core.knowledge_base.schema.database_featurizer import DatabaseSchemaFeaturizer
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
    E2ESingleStateFeaturizer,
)
from rasa.nlu.constants import ACTION_NAME, INTENT, ACTION_TEXT
from rasa.core.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.policies.policy import Policy
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE
from rasa.core.trackers import DialogueStateTracker
from rasa.utils import train_utils
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.transformer import TransformerEncoder, TransformerDecoder
from rasa.utils.tensorflow.models import RasaModel
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
from rasa.utils.tensorflow.constants import (
    LABEL,
    DATABASE,
    HIDDEN_LAYERS_SIZES,
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
)


logger = logging.getLogger(__name__)

DIALOGUE_FEATURES = f"{DIALOGUE}_features"
LABEL_FEATURES = f"{LABEL}_features"
LABEL_IDS = f"{LABEL}_ids"

SAVE_MODEL_FILE_NAME = "knowledge_base_policy"


class KnowledgeBasePolicy(TEDPolicy):

    SUPPORTS_ONLINE_TRAINING = True

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the dialogue and label embedding layers.
        # The number of hidden layers is equal to the length of the corresponding
        # list.
        HIDDEN_LAYERS_SIZES: {
            DIALOGUE: [],
            LABEL: [],
            f"{DIALOGUE}_name_text": [100],
            DATABASE: [128],
        },
        # Number of units in transformer
        TRANSFORMER_SIZE: 128,
        # Number of transformer layers
        NUM_TRANSFORMER_LAYERS: 1,
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [8, 32],
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
        # Dropout rate for embedding layers of label, e.g. action, features.
        DROP_RATE_LABEL: 0.0,
        # Dropout rate for attention.
        DROP_RATE_ATTENTION: 0,
        # Sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
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
    }

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        if max_history is None:
            return FullDialogueTrackerFeaturizer(E2ESingleStateFeaturizer())
        else:
            return MaxHistoryTrackerFeaturizer(
                E2ESingleStateFeaturizer(), max_history=max_history
            )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        rule_to_id_mapping: Optional[Dict[Text, int]] = None,
        id_to_rule_mapping: Optional[Dict[int, Text]] = None,
        **kwargs: Any,
    ) -> None:
        """Declare instance variables with default values."""

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super().__init__(featurizer, priority)
        if isinstance(featurizer, FullDialogueTrackerFeaturizer):
            self.is_full_dialogue_featurizer_used = True
        else:
            self.is_full_dialogue_featurizer_used = False

        self._load_params(**kwargs)

        self.model = model

        self.data_example: Optional[Dict[Text, List[np.ndarray]]] = None

        self.rule_to_id_mapping = rule_to_id_mapping
        self.id_to_rule_mapping = id_to_rule_mapping

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        self.config = copy.deepcopy(self.defaults)
        self.config.update(kwargs)

        self.config = train_utils.check_deprecated_options(self.config)

        self.config = train_utils.update_similarity_type(self.config)
        self.config = train_utils.update_evaluation_parameters(self.config)

    def _rule_id_mapping(self, domain: Domain) -> Dict[Text, int]:
        grammar = Grammar(domain.database_schema)
        rules = grammar.rules
        rules += grammar.build_instance_production()

        rule_to_feature_map = {}
        index = 0
        for rule in rules:
            if rule.nonterminal not in rule_to_feature_map:
                rule_to_feature_map[rule.nonterminal] = index
                index += 1
            rule_to_feature_map[rule.rule] = index
            index += 1
        rule_to_feature_map["C *"] = index

        return rule_to_feature_map

    @staticmethod
    def _invert_mapping(mapping: Dict) -> Dict:
        return {value: key for key, value in mapping.items()}

    # noinspection PyPep8Naming
    def _label_features_for_Y(
        self, label_ids: np.ndarray, domain: Domain
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
        """Prepare Y data for training:

        - feature vector for the output produced so far
        - feature vector for the next token to predict
        - possible options for the next token to predict
        """
        sql_converter = SQLConverter(domain.database_schema)

        rules_so_far = []
        next_rule_to_predict = []
        possible_rules_to_predict = []
        steps_per_dialogue = []

        for label_id in label_ids:
            query = domain.action_names[label_id]
            # ignore other actions like action listen
            if not query.startswith("SELECT"):
                continue

            # TODO why is it replaced in the first place?
            query = query.replace("<U>", "*")

            # TODO: check if SELECT * FROM x is converted correctly
            rules = sql_converter.convert_to_grammar_rules(query)

            steps_per_dialogue.append(len(rules))

            features = []
            for rule in rules:
                # output so far
                if rules_so_far:
                    f = rules_so_far[-1]
                else:
                    f = np.zeros(len(self.rule_to_id_mapping))
                idx = self.rule_to_id_mapping[rule.nonterminal]
                f[idx] = 1
                rules_so_far.append(f)
                # next rule to predict at this stage
                idx = self.rule_to_id_mapping[rule.rule]
                f = np.zeros(len(self.rule_to_id_mapping))
                f[idx] = 1
                next_rule_to_predict.append(f)
                # possible next actions
                next_rules = list(
                    GrammarRule.from_nonterminal(rule.nonterminal).grammar_dict.values()
                )
                f = np.zeros([len(next_rules), len(self.rule_to_id_mapping)])
                for i, r in enumerate(next_rules):
                    idx = self.rule_to_id_mapping[r]
                    f[i][idx] = 1
                possible_rules_to_predict.append(f)

        return (
            rules_so_far,
            next_rule_to_predict,
            possible_rules_to_predict,
            steps_per_dialogue,
        )

    def _create_model_data(
        self,
        data_X: np.ndarray,
        dialog_lengths: Optional[np.ndarray] = None,
        label_ids: Optional[np.ndarray] = None,
        domain: Optional[Domain] = None,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData."""

        if label_ids is not None:
            rules_so_far, next_rule_to_predict, possible_rules_to_predict, steps_per_dialogue = self._label_features_for_Y(
                label_ids, domain
            )
        else:
            rules_so_far, next_rule_to_predict, possible_rules_to_predict = np.array([])
            steps_per_dialogue = 1

        model_data = RasaModelData(label_key=LABEL_IDS)

        X_user_sparse = []
        X_user_dense = []
        X_intent = []
        X_user_if_text = []
        X_action_sparse = []
        X_action_dense = []
        X_action_name = []
        X_action_if_text = []
        X_entities = []

        for dial, steps in zip(data_X, steps_per_dialogue):
            (
                state_user_sparse,
                state_user_dense,
                state_intent,
                state_user_if_text,
            ) = self._process_user_and_action_features(dial[:, :4])
            (
                state_action_sparse,
                state_action_dense,
                state_action_name,
                state_action_if_text,
            ) = self._process_user_and_action_features(dial[:, 4:8])
            state_entites = self._process_entities(dial[:, 8])

            for _ in range(steps):
                X_user_sparse.append(state_user_sparse)
                X_user_dense.append(state_user_dense)
                X_intent.append(state_intent)
                X_user_if_text.append(state_user_if_text)
                X_action_sparse.append(state_action_sparse)
                X_action_dense.append(state_action_dense)
                X_action_name.append(state_action_name)
                X_action_if_text.append(state_action_if_text)
                X_entities.append(state_entites)

        model_data.add_features(
            f"{DIALOGUE_FEATURES}_user",
            [np.array(X_user_sparse), np.array(X_user_dense)],
        )
        model_data.add_features(f"{DIALOGUE_FEATURES}_entities", [np.array(X_entities)])
        model_data.add_features(f"{DIALOGUE_FEATURES}_user_name", [np.array(X_intent)])
        model_data.add_features(
            f"{DIALOGUE_FEATURES}_action",
            [np.array(X_action_sparse), np.array(X_action_dense)],
        )
        model_data.add_features(
            f"{DIALOGUE_FEATURES}_action_name", [np.array(X_action_name)]
        )
        model_data.add_features(
            f"{DIALOGUE_FEATURES}_action_if_text", [np.array(X_action_if_text)]
        )
        model_data.add_features(
            f"{DIALOGUE_FEATURES}_user_if_text", [np.array(X_user_if_text)]
        )

        model_data.add_features(f"{LABEL_FEATURES}_so_far", [np.array(rules_so_far)])
        model_data.add_features(
            f"{LABEL_FEATURES}_possible_next_rules",
            [np.array(possible_rules_to_predict)],
        )
        model_data.add_features(LABEL_IDS, [np.array(next_rule_to_predict)])

        if dialog_lengths is not None:
            final_dialogue_lengths = []
            for l, s in zip(dialog_lengths, steps_per_dialogue):
                for _ in range(s):
                    final_dialogue_lengths.append(l)
            model_data.add_features(
                "dialog_lengths", [np.array(final_dialogue_lengths)]
            )

        return model_data

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Train the policy on given training trackers."""

        database_features = DatabaseSchemaFeaturizer.featurize(domain.database_schema)

        # dealing with training data
        training_data = self.featurize_for_training(
            training_trackers, domain, interpreter, **kwargs
        )

        self.rule_to_id_mapping = self._rule_id_mapping(domain)
        self.id_to_rule_mapping = self._invert_mapping(self.rule_to_id_mapping)

        # extract actual training data to feed to model
        model_data = self._create_model_data(
            training_data.X,
            np.array(training_data.true_length),
            training_data.y,
            domain=domain,
        )
        if model_data.is_empty():
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        self.model = KnowledgeBaseModel(
            model_data.get_signature(),
            self.config,
            isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
            self.rule_to_id_mapping,
            self.id_to_rule_mapping,
            database_features,
        )

        self.model.fit(
            model_data,
            self.config[EPOCHS],
            self.config[BATCH_SIZES],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[EVAL_NUM_EPOCHS],
            batch_strategy=self.config[BATCH_STRATEGY],
        )

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter = RegexInterpreter(),
        **kwargs: Any,
    ) -> List[float]:
        """Predict the next action the bot should take.
        Return the list of probabilities for the next actions.
        """

        # TODO always return utter_result as action
        # TODO update the tracker state and set a QueryEvent

        return [0.0]

    def persist(self, path: Text) -> None:
        """Persists the policy to a storage."""

        if self.model is None:
            logger.debug(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )
            return

        model_path = Path(path)
        tf_model_file = model_path / f"{SAVE_MODEL_FILE_NAME}.tf_model"

        io_utils.create_directory_for_file(tf_model_file)

        self.featurizer.persist(path)

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
            model_path / f"{SAVE_MODEL_FILE_NAME}.label_data.pkl", self._label_data
        )

    @classmethod
    def load(cls, path: Text) -> "KnowledgeBasePolicy":
        """Loads a policy from the storage.
        **Needs to load its featurizer**
        """

        if not os.path.exists(path):
            raise Exception(
                f"Failed to load KnowledgeBase policy model. Path "
                f"'{os.path.abspath(path)}' doesn't exist."
            )

        model_path = Path(path)
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
        meta = io_utils.pickle_load(model_path / f"{SAVE_MODEL_FILE_NAME}.meta.pkl")
        priority = io_utils.json_unpickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.priority.pkl"
        )

        model_data_example = RasaModelData(label_key=LABEL_IDS, data=loaded_data)
        meta = train_utils.update_similarity_type(meta)

        model = KnowledgeBaseModel.load(
            str(tf_model_file),
            model_data_example,
            data_signature=model_data_example.get_signature(),
            config=meta,
            max_history_tracker_featurizer_used=isinstance(
                featurizer, MaxHistoryTrackerFeaturizer
            ),
            label_data=label_data,
        )

        # build the graph for prediction
        predict_data_example = RasaModelData(
            label_key=LABEL_IDS,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if DIALOGUE in feature_name
            },
        )
        model.build_for_predict(predict_data_example)

        return cls(featurizer=featurizer, priority=priority, model=model, **meta)


# accessing _tf_layers with any key results in key-error, disable it
# pytype: disable=key-error


class KnowledgeBaseModel(RasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
        max_history_tracker_featurizer_used: bool,
        rule_to_id_mapping: Dict[Text, int],
        id_to_rule_mapping: Dict[int, Text],
        database_features: np.ndarray,
    ) -> None:
        super().__init__(
            name="KnowledgeBase",
            random_seed=config[RANDOM_SEED],
            tensorboard_log_dir=config[TENSORBOARD_LOG_DIR],
            tensorboard_log_level=config[TENSORBOARD_LOG_LEVEL],
        )

        self.config = config
        self.max_history_tracker_featurizer_used = max_history_tracker_featurizer_used

        # data
        self.data_signature = data_signature

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if DIALOGUE in feature_name
        }

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        self.all_labels_embed = None

        self.id_to_rule_mapping = id_to_rule_mapping
        self.rule_to_id_mapping = rule_to_id_mapping

        self.database_features = database_features

        # metrics
        self.action_loss = tf.keras.metrics.Mean(name="loss")
        self.action_acc = tf.keras.metrics.Mean(name="acc")
        self.metrics_to_log += ["loss", "acc"]

        # set up tf layers
        self._tf_layers: Dict[Text : tf.keras.layers.Layer] = {}
        self._prepare_layers()

    def _prepare_sparse_dense_layers(
        self,
        data_signature: List[FeatureSignature],
        name: Text,
        reg_lambda: float,
        dense_dim: int,
    ) -> None:
        sparse = False
        dense = False
        for is_sparse, shape in data_signature:
            if is_sparse:
                sparse = True
            else:
                dense = True

        if sparse:
            self._tf_layers[f"sparse_to_dense.{name}"] = layers.DenseForSparse(
                units=dense_dim, reg_lambda=reg_lambda, name=name
            )
            if not dense:
                # create dense labels for the input to use in negative sampling
                self._tf_layers[f"sparse_to_dense_ids.{name}"] = layers.DenseForSparse(
                    units=2, trainable=False, name=f"sparse_to_dense_ids.{name}"
                )

    def _prepare_layers(self) -> None:
        self._tf_layers[f"loss.{LABEL}"] = layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MAX_POS_SIM],
            self.config[MAX_NEG_SIM],
            self.config[USE_MAX_NEG_SIM],
            self.config[NEGATIVE_MARGIN_SCALE],
            self.config[SCALE_LOSS],
            # set to 1 to get deterministic behaviour
            parallel_iterations=1 if self.random_seed is not None else 1000,
        )

        for feature_name in self.data_signature.keys():
            if feature_name.startswith(DIALOGUE_FEATURES) or feature_name.startswith(
                LABEL_FEATURES
            ):
                self._prepare_sparse_dense_layers(
                    self.data_signature[feature_name],
                    feature_name,
                    self.config[REGULARIZATION_CONSTANT],
                    100,
                )

            if feature_name.startswith(DIALOGUE_FEATURES) and not feature_name.endswith(
                "if_text"
            ):
                self._tf_layers[f"ffnn.{feature_name}"] = layers.Ffnn(
                    self.config[HIDDEN_LAYERS_SIZES][f"{DIALOGUE}_name_text"],
                    self.config[DROP_RATE_DIALOGUE],
                    self.config[REGULARIZATION_CONSTANT],
                    self.config[WEIGHT_SPARSITY],
                    layer_name_suffix=feature_name,
                )

        self._tf_layers[f"ffnn.{DIALOGUE}"] = layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][DIALOGUE],
            self.config[DROP_RATE_DIALOGUE],
            self.config[REGULARIZATION_CONSTANT],
            self.config[WEIGHT_SPARSITY],
            layer_name_suffix=DIALOGUE,
        )
        self._tf_layers[f"ffnn.{LABEL}"] = layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][LABEL],
            self.config[DROP_RATE_LABEL],
            self.config[REGULARIZATION_CONSTANT],
            self.config[WEIGHT_SPARSITY],
            layer_name_suffix=LABEL,
        )
        self._tf_layers[f"ffnn.{DATABASE}"] = layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][DATABASE],
            self.config[DROP_RATE_DIALOGUE],
            self.config[REGULARIZATION_CONSTANT],
            self.config[WEIGHT_SPARSITY],
            layer_name_suffix=DATABASE,
        )

        self._tf_layers["transformer_encoder"] = TransformerEncoder(
            self.config[NUM_TRANSFORMER_LAYERS],
            self.config[TRANSFORMER_SIZE],
            self.config[NUM_HEADS],
            self.config[TRANSFORMER_SIZE] * 4,
            self.config[REGULARIZATION_CONSTANT],
            dropout_rate=self.config[DROP_RATE_DIALOGUE],
            attention_dropout_rate=self.config[DROP_RATE_ATTENTION],
            sparsity=self.config[WEIGHT_SPARSITY],
            unidirectional=True,
            use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
            use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
            max_relative_position=self.config[MAX_RELATIVE_POSITION],
            name=DIALOGUE + "_encoder",
        )
        self._tf_layers["transformer_decoder"] = TransformerDecoder(
            self.config[NUM_TRANSFORMER_LAYERS],
            self.config[TRANSFORMER_SIZE],
            self.config[NUM_HEADS],
            self.config[TRANSFORMER_SIZE] * 4,
            self.config[REGULARIZATION_CONSTANT],
            dropout_rate=self.config[DROP_RATE_DIALOGUE],
            attention_dropout_rate=self.config[DROP_RATE_ATTENTION],
            sparsity=self.config[WEIGHT_SPARSITY],
            unidirectional=True,
            use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
            use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
            max_relative_position=self.config[MAX_RELATIVE_POSITION],
            name=DIALOGUE + "_decoder",
        )

        self._tf_layers[f"embed.{DIALOGUE}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            DIALOGUE,
            self.config[SIMILARITY_TYPE],
        )
        self._tf_layers[f"embed.{LABEL}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            LABEL,
            self.config[SIMILARITY_TYPE],
        )

    def _combine_sparse_dense_features(
        self,
        features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]],
        name: Text,
        sparse_dropout: bool = False,
    ) -> tf.Tensor:

        dense_features = []

        for f in features:
            if isinstance(f, tf.SparseTensor):
                if sparse_dropout:
                    _f = self._tf_layers[f"sparse_dropout.{name}"](f, self._training)
                else:
                    _f = f
                dense_features.append(self._tf_layers[f"sparse_to_dense.{name}"](_f))
            else:
                dense_features.append(f)

        return tf.concat(dense_features, axis=-1)

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_labels = []
        for key in self.tf_label_data.keys():
            if key.startswith(LABEL_FEATURES):
                all_labels.append(
                    self._combine_sparse_dense_features(self.tf_label_data[key], key)
                )

        all_labels = tf.concat(all_labels, axis=-1)
        all_labels = tf.squeeze(all_labels, axis=1)
        all_labels_embed = self._embed_label(all_labels)

        return all_labels, all_labels_embed

    @staticmethod
    def _compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        return mask

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        return tf.expand_dims(tf.gather_nd(x, indices), 1)

    def _encoded_dialogue(
        self, dialogue_in: tf.Tensor, sequence_lengths
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create dialogue level embedding and mask."""

        mask = self._compute_mask(sequence_lengths)

        dialogue = self._tf_layers[f"ffnn.{DIALOGUE}"](dialogue_in, self._training)
        dialogue_transformed = self._tf_layers["transformer_encoder"](
            dialogue, 1 - tf.expand_dims(mask, axis=-1), self._training
        )
        dialogue_transformed = tfa.activations.gelu(dialogue_transformed)

        if self.max_history_tracker_featurizer_used:
            # pick last label if max history featurizer is used
            # dialogue_transformed = dialogue_transformed[:, -1:, :]
            dialogue_transformed = self._last_token(
                dialogue_transformed, sequence_lengths
            )
            mask = self._last_token(mask, sequence_lengths)

        return dialogue_transformed, mask

    def _embed_label(self, label_in: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        label = self._tf_layers[f"ffnn.{LABEL}"](label_in, self._training)
        return self._tf_layers[f"embed.{LABEL}"](label)

    def _preprocess_batch(self, batch: Dict[Text, List[tf.Tensor]]) -> tf.Tensor:
        name_features = None
        text_features = None
        batch_features = []
        feats_to_combine = [f"{DIALOGUE_FEATURES}_user", f"{DIALOGUE_FEATURES}_action"]
        for feature in feats_to_combine:
            if not batch[feature] == []:
                text_features = self._combine_sparse_dense_features(
                    batch[feature], feature
                )
                text_features = self._tf_layers[f"ffnn.{feature}"](text_features)
                mask = tf.cast(
                    tf.math.equal(batch[f"{feature}_if_text"], 1), tf.float32
                )
                text_features = text_features * mask
            if not batch[f"{feature}_name"] == []:
                name_features = self._combine_sparse_dense_features(
                    batch[f"{feature}_name"], f"{feature}_name"
                )
                name_features = self._tf_layers[f"ffnn.{feature}_name"](name_features)
                mask = tf.cast(
                    tf.math.equal(batch[f"{feature}_if_text"], -1), tf.float32
                )
                name_features = name_features * mask

            if text_features is not None and name_features is not None:
                batch_features.append(text_features + name_features)
            else:
                batch_features.append(
                    text_features if text_features is not None else name_features
                )

        if not batch[f"{DIALOGUE_FEATURES}_entities"] == []:
            batch_features.append(batch[f"{DIALOGUE_FEATURES}_entities"])

        batch_features = tf.squeeze(tf.concat(batch_features, axis=-1), 0)
        return batch_features

    def _embed_label(self, label_in: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        label = self._tf_layers[f"ffnn.{LABEL}"](label_in, self._training)
        return self._tf_layers[f"embed.{LABEL}"](label)

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        batch = self.batch_to_model_data_format(batch_in, self.data_signature)

        sequence_lengths = tf.cast(
            tf.squeeze(batch["dialog_lengths"], axis=0), tf.int32
        )

        rule_to_predict = tf.cast(tf.squeeze(batch[f"{LABEL_IDS}"], axis=0), tf.float32)
        output_so_far = tf.cast(
            tf.squeeze(batch[f"{LABEL_FEATURES}_so_far"], axis=0), tf.float32
        )
        possible_next_rules = tf.cast(
            tf.squeeze(batch[f"{LABEL_FEATURES}_possible_next_rules"], axis=0),
            tf.float32,
        )

        dialogue_in = self._preprocess_batch(batch)

        dialogue_encoded, mask = self._encoded_dialogue(dialogue_in, sequence_lengths)

        database_features = tf.convert_to_tensor(
            self.database_features, dtype=tf.float32
        )
        database_features = self._tf_layers[f"ffnn.{DATABASE}"](
            database_features, self._training
        )

        # TODO
        in_features = tf.concat([database_features, dialogue_encoded], axis=0)

        output_so_far = tf.expand_dims(output_so_far, axis=1)
        output_so_far = self._tf_layers[f"ffnn.{LABEL}"](output_so_far, self._training)

        dialogue_transformed = self._tf_layers["transformer_decoder"](
            output_so_far,
            dialogue_encoded,
            1 - tf.expand_dims(mask, axis=-1),
            self._training,
        )
        dialogue_transformed = tfa.activations.gelu(dialogue_transformed)

        label_embed = self._embed_label(rule_to_predict)
        all_labels_embed = self._embed_label(possible_next_rules)

        dialogue_embed = self._tf_layers[f"embed.{DIALOGUE}"](dialogue_transformed)

        loss, acc = self._tf_layers[f"loss.{LABEL}"](
            dialogue_embed,
            label_embed,
            rule_to_predict,
            all_labels_embed,
            possible_next_rules,
            mask,
        )

        self.action_loss.update_state(loss)
        self.action_acc.update_state(acc)

        return loss

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        batch = self.batch_to_model_data_format(batch_in, self.predict_data_signature)

        dialogue_in = self._preprocess_batch(batch)
        sequence_lengths = tf.expand_dims(tf.shape(dialogue_in)[1], axis=0)

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels_embed()

        dialogue_embed, mask = self._encoded_dialogue(dialogue_in, sequence_lengths)

        sim_all = self._tf_layers[f"loss.{LABEL}"].sim(
            dialogue_embed[:, :, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            mask,
        )

        scores = self._tf_layers[f"loss.{LABEL}"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )

        return {"action_scores": scores}


# pytype: enable=key-error
