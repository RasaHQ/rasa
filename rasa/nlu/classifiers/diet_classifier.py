import logging

import numpy as np
import os
import pickle
import scipy.sparse
import warnings
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, Dict, List, Optional, Text, Tuple, Union

import rasa.utils.io
import rasa.nlu.utils.bilou_utils as bilou_utils
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.test import determine_token_labels
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.components import any_of
from rasa.utils import train_utils
from rasa.utils.tensorflow import tf_layers
from rasa.utils.tensorflow.tf_models import RasaModel
from rasa.utils.tensorflow.tf_model_data import RasaModelData, FeatureSignature
from rasa.nlu.constants import (
    INTENT,
    TEXT,
    ENTITIES,
    SPARSE_FEATURE_NAMES,
    DENSE_FEATURE_NAMES,
    TOKENS_NAMES,
)
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    SHARE_HIDDEN_LAYERS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    MAX_SEQ_LENGTH,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    DENSE_DIM,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    SPARSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROPRATE,
    C_EMB,
    C2,
    SCALE_LOSS,
    USE_MAX_SIM_NEG,
    MU_NEG,
    MU_POS,
    EMBED_DIM,
    BILOU_FLAG,
)


logger = logging.getLogger(__name__)


class DIETClassifier(EntityExtractor):
    """label classifier using supervised embeddings.

    The embedding intent classifier embeds user inputs
    and intent labels into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    It also provides rankings of the labels that did not "win".

    The embedding intent classifier needs to be preceded by
    a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``CountVectorsFeaturizer`` that
    can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
    However, in this implementation the `mu` parameter is treated differently
    and additional hidden layers are added together with dropout.
    """

    provides = ["intent", "intent_ranking", "entities"]

    requires = [any_of(DENSE_FEATURE_NAMES[TEXT], SPARSE_FEATURE_NAMES[TEXT])]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer
        # for input words and intent labels,
        # the number of hidden layers is thus equal to the length of this list
        HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []},
        # Whether to share the hidden layer weights between input words and labels
        SHARE_HIDDEN_LAYERS: False,
        # number of units in transformer
        TRANSFORMER_SIZE: 256,
        # number of transformer layers
        NUM_TRANSFORMER_LAYERS: 2,
        # number of attention heads in transformer
        NUM_HEADS: 4,
        # max sequence length if pos_encoding='emb'
        MAX_SEQ_LENGTH: 256,
        # training parameters
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        BATCH_SIZES: [64, 256],
        # how to create batches
        BATCH_STRATEGY: "balanced",  # string 'sequence' or 'balanced'
        # number of epochs
        EPOCHS: 300,
        # set random seed to any int to get reproducible results
        RANDOM_SEED: None,
        # optimizer
        LEARNING_RATE: 0.001,
        # embedding parameters
        # default dense dimension used if no dense features are present
        DENSE_DIM: {TEXT: 512, LABEL: 20},
        # dimension size of embedding vectors
        EMBED_DIM: 20,
        # the type of the similarity
        NUM_NEG: 20,
        # flag if minimize only maximum similarity over incorrect actions
        SIMILARITY_TYPE: "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        LOSS_TYPE: "softmax",  # string 'softmax' or 'margin'
        # number of top intents to normalize scores for softmax loss_type
        # set to 0 to turn off normalization
        RANKING_LENGTH: 10,
        # how similar the algorithm should try
        # to make embedding vectors for correct labels
        MU_POS: 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect labels
        MU_NEG: -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # flag: if true, only minimize the maximum similarity for incorrect labels
        USE_MAX_SIM_NEG: True,
        # scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # regularization parameters
        # the scale of L2 regularization
        C2: 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different labels
        C_EMB: 0.8,
        # dropout rate for rnn
        DROPRATE: 0.2,
        # use a unidirectional or bidirectional encoder
        UNIDIRECTIONAL_ENCODER: False,
        # if true apply dropout to sparse tensors
        SPARSE_INPUT_DROPOUT: True,
        # visualization of accuracy
        # how often to calculate training accuracy
        EVAL_NUM_EPOCHS: 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance
        # model config
        # if true intent classification is trained and intent predicted
        INTENT_CLASSIFICATION: True,
        # if true named entity recognition is trained and entities predicted
        ENTITY_RECOGNITION: True,
        # if true random tokens of the input message will be masked and the model
        # should predict those tokens
        MASKED_LM: False,
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        BILOU_FLAG: True,
    }
    # end default properties (DOC MARKER - don't remove)

    # init helpers
    def _check_config_parameters(self) -> None:
        self.component_config = train_utils.check_deprecated_options(
            self.component_config
        )

        if (
            self.component_config[MASKED_LM]
            and self.component_config[NUM_TRANSFORMER_LAYERS] == 0
        ):
            raise ValueError(
                f"If number of transformer layers is 0, "
                f"'{MASKED_LM}' option should be 'False'."
            )

        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            v1 = next(iter(self.component_config[HIDDEN_LAYERS_SIZES].values()))
            if any(
                v != v1 for v in self.component_config[HIDDEN_LAYERS_SIZES].values()
            ):
                raise ValueError(
                    f"If hidden layer weights are shared, "
                    f"{HIDDEN_LAYERS_SIZES} must coincide."
                )

        self.component_config = train_utils.update_similarity_type(
            self.component_config
        )

        if self.component_config[EVAL_NUM_EPOCHS] < 1:
            self.component_config[EVAL_NUM_EPOCHS] = self.component_config[EPOCHS]

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        inverted_tag_dict: Optional[Dict[int, Text]] = None,
        model: Optional[RasaModel] = None,
        batch_tuple_sizes: Optional[Dict] = None,
    ) -> None:
        """Declare instance variables with default values"""

        super().__init__(component_config)

        self._check_config_parameters()

        # transform numbers to labels
        self.inverted_label_dict = inverted_label_dict
        self.inverted_tag_dict = inverted_tag_dict

        self.model = model

        # encode all label_ids with numbers
        self._label_data = None

        # keep the input tuple sizes in self.batch_in
        self.batch_tuple_sizes = batch_tuple_sizes

        # number of entity tags
        self.num_tags = 0

        self.data_example = None

        self.label_key = (
            "label_ids" if self.component_config[INTENT_CLASSIFICATION] else None
        )

    @staticmethod
    def model_class() -> Any:
        return DIET

    # training data helpers:
    @staticmethod
    def _create_label_id_dict(
        training_data: TrainingData, attribute: Text
    ) -> Dict[Text, int]:
        """Create label_id dictionary"""

        distinct_label_ids = {
            example.get(attribute) for example in training_data.intent_examples
        } - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    def _create_tag_id_dict(self, training_data: TrainingData) -> Dict[Text, int]:
        """Create label_id dictionary"""

        if self.component_config[BILOU_FLAG]:
            return bilou_utils.build_tag_id_dict(training_data)

        distinct_tag_ids = set(
            [
                e["entity"]
                for example in training_data.entity_examples
                for e in example.get(ENTITIES)
            ]
        ) - {None}

        tag_id_dict = {
            tag_id: idx for idx, tag_id in enumerate(sorted(distinct_tag_ids), 1)
        }
        tag_id_dict["O"] = 0

        return tag_id_dict

    @staticmethod
    def _find_example_for_label(
        label: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    @staticmethod
    def _find_example_for_tag(
        tag: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        for ex in examples:
            for e in ex.get(attribute):
                if e["entity"] == tag:
                    return ex
        return None

    @staticmethod
    def _check_labels_features_exist(
        labels_example: List[Message], attribute: Text
    ) -> bool:
        """Check if all labels have features set"""

        for label_example in labels_example:
            if (
                label_example.get(SPARSE_FEATURE_NAMES[attribute]) is None
                and label_example.get(DENSE_FEATURE_NAMES[attribute]) is None
            ):
                return False
        return True

    @staticmethod
    def _extract_and_add_features(
        message: Message, attribute: Text
    ) -> Tuple[Optional[scipy.sparse.spmatrix], Optional[np.ndarray]]:
        sparse_features = None
        dense_features = None

        if message.get(SPARSE_FEATURE_NAMES[attribute]) is not None:
            sparse_features = message.get(SPARSE_FEATURE_NAMES[attribute])

        if message.get(DENSE_FEATURE_NAMES[attribute]) is not None:
            dense_features = message.get(DENSE_FEATURE_NAMES[attribute])

        if sparse_features is not None and dense_features is not None:
            if sparse_features.shape[0] != dense_features.shape[0]:
                raise ValueError(
                    f"Sequence dimensions for sparse and dense features "
                    f"don't coincide in '{message.text}' for attribute '{attribute}'."
                )

        return sparse_features, dense_features

    def check_input_dimension_consistency(self, model_data: RasaModelData):
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            num_text_features = model_data.get_feature_dimension("text_features")
            num_label_features = model_data.get_feature_dimension("label_features")

            if num_text_features != num_label_features:
                raise ValueError(
                    "If embeddings are shared text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def _extract_labels_precomputed_features(
        self, label_examples: List[Message], attribute: Text = INTENT
    ) -> List[np.ndarray]:
        """Collect precomputed encodings"""

        sparse_features = []
        dense_features = []

        for e in label_examples:
            _sparse, _dense = self._extract_and_add_features(e, attribute)
            if _sparse is not None:
                sparse_features.append(_sparse)
            if _dense is not None:
                dense_features.append(_dense)

        sparse_features = np.array(sparse_features)
        dense_features = np.array(dense_features)

        return [sparse_features, dense_features]

    @staticmethod
    def _compute_default_label_features(
        labels_example: List[Message],
    ) -> List[np.ndarray]:
        """Compute one-hot representation for the labels"""

        return [
            np.array(
                [
                    np.expand_dims(a, 0)
                    for a in np.eye(len(labels_example), dtype=np.float32)
                ]
            )
        ]

    def _create_label_data(
        self,
        training_data: TrainingData,
        label_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> RasaModelData:
        """Create matrix with label_ids encoded in rows as bag of words.

        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """

        # Collect one example for each label
        labels_idx_example = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_example.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_example = sorted(labels_idx_example, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_example]

        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            features = self._extract_labels_precomputed_features(
                labels_example, attribute
            )
        else:
            features = self._compute_default_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features("label_features", features)

        label_ids = np.array([idx for (idx, _) in labels_idx_example])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        label_data.add_features("label_ids", [np.expand_dims(label_ids, -1)])

        label_data.add_mask("label_mask", "label_features")

        return label_data

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[np.ndarray]:
        return [
            np.array(
                [
                    self._label_data.get("label_features")[0][label_id]
                    for label_id in label_ids
                ]
            )
        ]

    def _create_model_data(
        self,
        training_data: List[Message],
        label_id_dict: Optional[Dict[Text, int]] = None,
        tag_id_dict: Optional[Dict[Text, int]] = None,
        label_attribute: Optional[Text] = None,
    ) -> RasaModelData:
        """Prepare data for training and create a SessionDataType object"""

        X_sparse = []
        X_dense = []
        Y_sparse = []
        Y_dense = []
        label_ids = []
        tag_ids = []

        for e in training_data:
            if label_attribute is None or e.get(label_attribute):
                _sparse, _dense = self._extract_and_add_features(e, TEXT)
                if _sparse is not None:
                    X_sparse.append(_sparse)
                if _dense is not None:
                    X_dense.append(_dense)

            if e.get(label_attribute):
                _sparse, _dense = self._extract_and_add_features(e, label_attribute)
                if _sparse is not None:
                    Y_sparse.append(_sparse)
                if _dense is not None:
                    Y_dense.append(_dense)

                if label_id_dict:
                    label_ids.append(label_id_dict[e.get(label_attribute)])

            if self.component_config[ENTITY_RECOGNITION] and tag_id_dict:
                if self.component_config[BILOU_FLAG]:
                    _tags = bilou_utils.tags_to_ids(e, tag_id_dict)
                else:
                    _tags = []
                    for t in e.get(TOKENS_NAMES[TEXT]):
                        _tag = determine_token_labels(t, e.get(ENTITIES), None)
                        _tags.append(tag_id_dict[_tag])
                # transpose to have seq_len x 1
                tag_ids.append(np.array([_tags]).T)

        X_sparse = np.array(X_sparse)
        X_dense = np.array(X_dense)
        Y_sparse = np.array(Y_sparse)
        Y_dense = np.array(Y_dense)
        label_ids = np.array(label_ids)
        tag_ids = np.array(tag_ids)

        model_data = RasaModelData(label_key=self.label_key)
        model_data.add_features("text_features", [X_sparse, X_dense])
        model_data.add_features("label_features", [Y_sparse, Y_dense])
        if label_attribute and model_data.feature_not_exists("label_features"):
            # no label features are present, get default features from _label_data
            model_data.add_features(
                "label_features", self._use_default_label_features(label_ids)
            )

        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        model_data.add_features("label_ids", [np.expand_dims(label_ids, -1)])
        model_data.add_features("tag_ids", [tag_ids])

        model_data.add_mask("text_mask", "text_features")
        model_data.add_mask("label_mask", "label_features")

        return model_data

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """
        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)

        label_id_dict = self._create_label_id_dict(training_data, attribute=INTENT)
        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}

        self._label_data = self._create_label_data(
            training_data, label_id_dict, attribute=INTENT
        )

        tag_id_dict = self._create_tag_id_dict(training_data)
        self.inverted_tag_dict = {v: k for k, v in tag_id_dict.items()}

        label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )

        model_data = self._create_model_data(
            training_data.training_examples,
            label_id_dict,
            tag_id_dict,
            label_attribute=label_attribute,
        )

        self.num_tags = len(self.inverted_tag_dict)

        self.check_input_dimension_consistency(model_data)

        return model_data

    @staticmethod
    def _check_enough_labels(model_data: RasaModelData) -> bool:
        return len(np.unique(model_data.get("label_ids"))) >= 2

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        logger.debug("Started training embedding classifier.")

        # set numpy random seed
        np.random.seed(self.component_config[RANDOM_SEED])

        model_data = self.preprocess_train_data(training_data)
        if model_data.is_empty():
            logger.error(
                "Can not train DIET classifier. No data was provided. "
                "Skipping training of the classifier."
            )
            return

        if self.component_config[INTENT_CLASSIFICATION]:
            if not self._check_enough_labels(model_data):
                logger.error(
                    "Can not train intent classifier. "
                    "Need at least 2 different intent classes. "
                    "Skipping training of classifier."
                )
                return

        # keep one example for persisting and loading
        self.data_example = {k: [v[:1] for v in vs] for k, vs in model_data.items()}

        self.model = self.model_class()(
            model_data.get_signature(),
            self._label_data,
            self.inverted_tag_dict,
            self.component_config,
        )

        self.model.fit(
            model_data,
            self.component_config[EPOCHS],
            self.component_config[BATCH_SIZES],
            self.component_config[EVAL_NUM_EXAMPLES],
            self.component_config[EVAL_NUM_EPOCHS],
            self.component_config[BATCH_STRATEGY],
        )

    # process helpers
    def _predict(self, message: Message) -> Optional[Dict[Text, tf.Tensor]]:
        if self.model is None:
            logger.error(
                "There is no trained model: component is either not trained or "
                "didn't receive enough training data."
            )
            return

        # create session data from message and convert it into a batch of 1
        model_data = self._create_model_data([message])

        return self.model.predict(model_data)

    def _predict_label(
        self, predict_out: Optional[Dict[Text, tf.Tensor]]
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""

        label = {"name": None, "confidence": 0.0}
        label_ranking = []

        if predict_out is None:
            return label, label_ranking

        message_sim = predict_out["i_scores"].numpy()

        message_sim = message_sim.flatten()  # sim is a matrix

        label_ids = message_sim.argsort()[::-1]

        if (
            self.component_config[LOSS_TYPE] == "softmax"
            and self.component_config[RANKING_LENGTH] > 0
        ):
            message_sim = train_utils.normalize(
                message_sim, self.component_config[RANKING_LENGTH]
            )

        message_sim[::-1].sort()
        message_sim = message_sim.tolist()

        # if X contains all zeros do not predict some label
        if label_ids.size > 0:
            label = {
                "name": self.inverted_label_dict[label_ids[0]],
                "confidence": message_sim[0],
            }

            if (
                self.component_config[RANKING_LENGTH]
                and 0 < self.component_config[RANKING_LENGTH] < LABEL_RANKING_LENGTH
            ):
                output_length = self.component_config[RANKING_LENGTH]
            else:
                output_length = LABEL_RANKING_LENGTH

            ranking = list(zip(list(label_ids), message_sim))
            ranking = ranking[:output_length]
            label_ranking = [
                {"name": self.inverted_label_dict[label_idx], "confidence": score}
                for label_idx, score in ranking
            ]

        return label, label_ranking

    def _predict_entities(
        self, predict_out: Optional[Dict[Text, tf.Tensor]], message: Message
    ) -> List[Dict]:
        if predict_out is None:
            return []

        # load tf graph and session
        predictions = predict_out["e_ids"].numpy()

        tags = [self.inverted_tag_dict[p] for p in predictions[0]]

        if self.component_config[BILOU_FLAG]:
            tags = bilou_utils.remove_bilou_prefixes(tags)

        entities = self._convert_tags_to_entities(
            message.text, message.get("tokens", []), tags
        )

        extracted = self.add_extractor_name(entities)
        entities = message.get("entities", []) + extracted

        return entities

    @staticmethod
    def _convert_tags_to_entities(
        text: Text, tokens: List[Token], tags: List[Text]
    ) -> List[Dict[Text, Any]]:
        entities = []
        last_tag = "O"
        for token, tag in zip(tokens, tags):
            if tag == "O":
                last_tag = tag
                continue

            # new tag found
            if last_tag != tag:
                entity = {
                    "entity": tag,
                    "start": token.start,
                    "end": token.end,
                    "extractor": "DIET",
                }
                entities.append(entity)

            # belongs to last entity
            elif last_tag == tag:
                entities[-1]["end"] = token.end

            last_tag = tag

        for entity in entities:
            entity["value"] = text[entity["start"] : entity["end"]]

        return entities

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely label and its similarity to the input."""

        out = self._predict(message)

        if self.component_config[INTENT_CLASSIFICATION]:
            label, label_ranking = self._predict_label(out)

            message.set("intent", label, add_to_output=True)
            message.set("intent_ranking", label_ranking, add_to_output=True)

        if self.component_config[ENTITY_RECOGNITION]:
            entities = self._predict_entities(out, message)

            message.set("entities", entities, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        if self.model is None:
            return {"file": None}

        tf_model_file = os.path.join(model_dir, file_name + ".tf_model")

        rasa.utils.io.create_directory_for_file(tf_model_file)

        self.model.save(tf_model_file)

        with open(os.path.join(model_dir, file_name + ".data_example.pkl"), "wb") as f:
            pickle.dump(self.data_example, f)

        with open(os.path.join(model_dir, file_name + ".label_data.pkl"), "wb") as f:
            pickle.dump(self._label_data, f)

        with open(
            os.path.join(model_dir, file_name + ".inv_label_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.inverted_label_dict, f)

        with open(os.path.join(model_dir, file_name + ".inv_tag_dict.pkl"), "wb") as f:
            pickle.dump(self.inverted_tag_dict, f)

        with open(
            os.path.join(model_dir, file_name + ".batch_tuple_sizes.pkl"), "wb"
        ) as f:
            pickle.dump(self.batch_tuple_sizes, f)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["DIETClassifier"] = None,
        **kwargs: Any,
    ) -> "DIETClassifier":
        """Loads the trained model from the provided directory."""

        if not model_dir or not meta.get("file"):
            warnings.warn(
                f"Failed to load nlu model. "
                f"Maybe the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)

        (
            batch_tuple_sizes,
            inv_label_dict,
            inv_tag_dict,
            label_data,
            meta,
            data_example,
        ) = cls._load_from_files(meta, model_dir)

        meta = train_utils.update_similarity_type(meta)

        model = cls._load_model(inv_tag_dict, label_data, meta, data_example, model_dir)

        return cls(
            component_config=meta,
            inverted_label_dict=inv_label_dict,
            inverted_tag_dict=inv_tag_dict,
            model=model,
            batch_tuple_sizes=batch_tuple_sizes,
        )

    @classmethod
    def _load_from_files(cls, meta: Dict[Text, Any], model_dir: Text):
        file_name = meta.get("file")

        with open(os.path.join(model_dir, file_name + ".data_example.pkl"), "rb") as f:
            data_example = pickle.load(f)

        with open(os.path.join(model_dir, file_name + ".label_data.pkl"), "rb") as f:
            label_data = pickle.load(f)

        with open(
            os.path.join(model_dir, file_name + ".inv_label_dict.pkl"), "rb"
        ) as f:
            inv_label_dict = pickle.load(f)

        with open(os.path.join(model_dir, file_name + ".inv_tag_dict.pkl"), "rb") as f:
            inv_tag_dict = pickle.load(f)

        with open(
            os.path.join(model_dir, file_name + ".batch_tuple_sizes.pkl"), "rb"
        ) as f:
            batch_tuple_sizes = pickle.load(f)

        return (
            batch_tuple_sizes,
            inv_label_dict,
            inv_tag_dict,
            label_data,
            meta,
            data_example,
        )

    @classmethod
    def _load_model(
        cls,
        inv_tag_dict: Dict[int, Text],
        label_data: RasaModelData,
        meta: Dict[Text, Any],
        data_example: Dict[Text, List[np.ndarray]],
        model_dir: Text,
    ):
        file_name = meta.get("file")
        tf_model_file = os.path.join(model_dir, file_name + ".tf_model")

        label_key = "label_ids" if meta[INTENT_CLASSIFICATION] else None
        model_data_example = RasaModelData(label_key=label_key, data=data_example)

        model = cls.model_class().load(
            tf_model_file,
            model_data_example,
            data_signature=model_data_example.get_signature(),
            label_data=label_data,
            inverted_tag_dict=inv_tag_dict,
            config=meta,
        )

        # build the graph for prediction
        predict_data_example = RasaModelData(
            label_key=label_key,
            data={k: vs for k, vs in model_data_example.items() if "text" in k},
        )

        model.build_for_predict(predict_data_example)

        return model


# pytype: disable=key-error


class DIET(RasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, List[FeatureSignature]],
        label_data: RasaModelData,
        inverted_tag_dict: Optional[Dict[int, Text]],
        config: Dict[Text, Any],
    ) -> None:
        super().__init__(name="DIET", random_seed=config[RANDOM_SEED])

        # data
        self.data_signature = data_signature
        self._check_data()

        self.predict_data_signature = {
            k: vs for k, vs in data_signature.items() if "text" in k
        }

        label_batch = label_data.prepare_batch()
        self.tf_label_data = self.batch_to_model_data_format(
            label_batch, label_data.get_signature()
        )
        self._num_tags = len(inverted_tag_dict) if inverted_tag_dict is not None else 0

        self.config = config

        # tf objects
        self._tf_layers = {}
        self._prepare_layers()

        # tf training
        self._set_optimizer(tf.keras.optimizers.Adam(config[LEARNING_RATE]))
        self._create_metrics()
        self._update_metrics_to_log()

        self.all_labels_embed = None  # needed for efficient prediction

    def _check_data(self):
        if "text_features" not in self.data_signature:
            raise ValueError(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if self.config[INTENT_CLASSIFICATION]:
            if "label_features" not in self.data_signature:
                raise ValueError(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )
            if (
                self.config[SHARE_HIDDEN_LAYERS]
                and self.data_signature["text_features"]
                != self.data_signature["label_features"]
            ):
                raise ValueError(
                    "If hidden layer weights are shared, data signatures "
                    "for text_features and label_features must coincide."
                )

        if self.config[ENTITY_RECOGNITION] and "tag_ids" not in self.data_signature:
            raise ValueError(
                f"No tag ids present. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    def _create_metrics(self):
        # self.metrics preserve order
        # output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        # output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.response_acc = tf.keras.metrics.Mean(name="i_acc")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")

    def _update_metrics_to_log(self) -> None:
        if self.config[MASKED_LM]:
            self.metrics_to_log += ["m_loss", "m_acc"]
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log += ["i_loss", "i_acc"]
        if self.config[ENTITY_RECOGNITION]:
            self.metrics_to_log += ["e_loss", "e_f1"]

    def _prepare_layers(self) -> None:
        self.text_name = TEXT
        self._prepare_sequence_layers(self.text_name)
        if self.config[MASKED_LM]:
            self._prepare_mask_lm_layers(self.text_name)
        if self.config[INTENT_CLASSIFICATION]:
            self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL
            self._prepare_input_layers(self.label_name)
            self._prepare_label_classification_layers()
        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_sparse_dense_layers(
        self,
        feature_signatures: List[FeatureSignature],
        name: Text,
        reg_lambda: float,
        dense_dim: int,
    ) -> None:
        sparse = False
        dense = False
        for is_sparse, shape in feature_signatures:
            if is_sparse:
                sparse = True
            else:
                dense = True
                # if dense features are present
                # use the feature dimension of the dense features
                dense_dim = shape[-1]

        if sparse:
            self._tf_layers[f"sparse_to_dense.{name}"] = tf_layers.DenseForSparse(
                units=dense_dim, reg_lambda=reg_lambda, name=name
            )
            if not dense:
                # create dense labels for the input to use in negative sampling
                self._tf_layers[
                    f"sparse_to_dense_ids.{name}"
                ] = tf_layers.DenseForSparse(
                    units=2, trainable=False, name=f"sparse_to_dense_ids.{name}"
                )

    def _prepare_input_layers(self, name: Text) -> None:
        self._tf_layers[f"sparse_dropout.{name}"] = tf_layers.SparseDropout(
            rate=self.config[DROPRATE]
        )
        self._prepare_sparse_dense_layers(
            self.data_signature[f"{name}_features"],
            name,
            self.config[C2],
            self.config[DENSE_DIM][name],
        )
        self._tf_layers[f"ffnn.{name}"] = tf_layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][name],
            self.config[DROPRATE],
            self.config[C2],
            name,
        )

    def _prepare_sequence_layers(self, name: Text) -> None:
        self._prepare_input_layers(name)

        self._tf_layers[f"{name}_transformer"] = (
            tf_layers.TransformerEncoder(
                self.config[NUM_TRANSFORMER_LAYERS],
                self.config[TRANSFORMER_SIZE],
                self.config[NUM_HEADS],
                self.config[TRANSFORMER_SIZE] * 4,
                self.config[MAX_SEQ_LENGTH],
                self.config[C2],
                dropout_rate=self.config[DROPRATE],
                attention_dropout_rate=self.config[DROPRATE],
                unidirectional=self.config[UNIDIRECTIONAL_ENCODER],
                name=f"{name}_encoder",
            )
            if self.config[NUM_TRANSFORMER_LAYERS] > 0
            else lambda x, mask, training: x
        )

    def _prepare_mask_lm_layers(self, name: Text) -> None:
        self._tf_layers[f"{name}_input_mask"] = tf_layers.InputMask()
        self._tf_layers[f"embed.{name}_lm_mask"] = tf_layers.Embed(
            self.config[EMBED_DIM],
            self.config[C2],
            f"{name}_lm_mask",
            self.config[SIMILARITY_TYPE],
        )
        self._tf_layers[f"embed.{name}_golden_token"] = tf_layers.Embed(
            self.config[EMBED_DIM],
            self.config[C2],
            f"{name}_golden_token",
            self.config[SIMILARITY_TYPE],
        )
        self._tf_layers[f"loss.{name}_mask"] = tf_layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MU_POS],
            self.config[MU_NEG],
            self.config[USE_MAX_SIM_NEG],
            self.config[C_EMB],
            self.config[SCALE_LOSS],
            # set to 1 to get deterministic behaviour
            parallel_iterations=1 if self.random_seed is not None else 1000,
        )

    def _prepare_label_classification_layers(self) -> None:
        self._tf_layers["embed.text"] = tf_layers.Embed(
            self.config[EMBED_DIM],
            self.config[C2],
            "text",
            self.config[SIMILARITY_TYPE],
        )
        self._tf_layers["embed.label"] = tf_layers.Embed(
            self.config[EMBED_DIM],
            self.config[C2],
            "label",
            self.config[SIMILARITY_TYPE],
        )
        self._tf_layers["loss.label"] = tf_layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MU_POS],
            self.config[MU_NEG],
            self.config[USE_MAX_SIM_NEG],
            self.config[C_EMB],
            self.config[SCALE_LOSS],
            # set to 1 to get deterministic behaviour
            parallel_iterations=1 if self.random_seed is not None else 1000,
        )

    def _prepare_entity_recognition_layers(self) -> None:
        self._tf_layers["embed.logits"] = tf_layers.Embed(
            self._num_tags, self.config[C2], "logits"
        )
        self._tf_layers["crf"] = tf_layers.CRF(self._num_tags, self.config[C2])
        self._tf_layers["crf_f1_score"] = tfa.metrics.F1Score(
            num_classes=self._num_tags - 1,  # `0` prediction is not a prediction
            average="micro",
        )

    @staticmethod
    def _get_sequence_lengths(mask: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.reduce_sum(mask[:, :, 0], 1), tf.int32)

    def _combine_sparse_dense_features(
        self,
        features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]],
        mask: tf.Tensor,
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

        return tf.concat(dense_features, axis=-1) * mask

    def _features_as_seq_ids(
        self, features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]], name: Text,
    ) -> tf.Tensor:
        # if there are dense features it's enough
        for f in features:
            if not isinstance(f, tf.SparseTensor):
                return tf.stop_gradient(f)

        # we need dense labels for negative sampling
        for f in features:
            if isinstance(f, tf.SparseTensor):
                return tf.stop_gradient(
                    self._tf_layers[f"sparse_to_dense_ids.{name}"](f)
                )

    def _create_bow(
        self,
        features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask: tf.Tensor,
        name: Text,
        sparse_dropout: bool = False,
    ) -> tf.Tensor:

        x = self._combine_sparse_dense_features(features, mask, name, sparse_dropout)
        x = tf.reduce_sum(x, 1)  # convert to bag-of-words
        return self._tf_layers[f"ffnn.{name}"](x, self._training)

    def _create_sequence(
        self,
        features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask: tf.Tensor,
        name: Text,
        masked_lm_loss: bool = False,
        sequence_ids: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        if sequence_ids:
            x_seq_ids = self._features_as_seq_ids(features, name)
        else:
            x_seq_ids = None

        x = self._combine_sparse_dense_features(
            features, mask, name, sparse_dropout=self.config[SPARSE_INPUT_DROPOUT]
        )

        pre = self._tf_layers[f"ffnn.{name}"](x, self._training)

        if masked_lm_loss:
            pre, lm_mask_bool = self._tf_layers[f"{name}_input_mask"](
                pre, mask, self._training
            )
        else:
            lm_mask_bool = None

        transformed = self._tf_layers[f"{name}_transformer"](
            pre, 1 - mask, self._training
        )
        transformed = tfa.activations.gelu(transformed)

        return transformed, x, x_seq_ids, lm_mask_bool

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data["label_ids"][0]
        x = self._create_bow(
            self.tf_label_data["label_features"],
            self.tf_label_data["label_mask"][0],
            self.label_name,
        )
        all_labels_embed = self._tf_layers["embed.label"](x)

        return all_label_ids, all_labels_embed

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_index = tf.maximum(0, sequence_lengths - 1)
        idxs = tf.stack([tf.range(tf.shape(last_index)[0]), last_index], axis=1)
        return tf.gather_nd(x, idxs)

    def _mask_loss(
        self,
        a_transformed: tf.Tensor,
        a: tf.Tensor,
        a_seq_ids: tf.Tensor,
        lm_mask_bool: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:
        # make sure there is at least one element in the mask
        lm_mask_bool = tf.cond(
            tf.reduce_any(lm_mask_bool),
            lambda: lm_mask_bool,
            lambda: tf.scatter_nd([[0, 0, 0]], [True], tf.shape(lm_mask_bool)),
        )

        lm_mask_bool = tf.squeeze(lm_mask_bool, -1)
        a_t_masked = tf.boolean_mask(a_transformed, lm_mask_bool)
        a_masked = tf.boolean_mask(a, lm_mask_bool)
        a_masked_ids = tf.boolean_mask(a_seq_ids, lm_mask_bool)

        a_t_masked_embed = self._tf_layers[f"embed.{name}_lm_mask"](a_t_masked)
        a_masked_embed = self._tf_layers[f"embed.{name}_golden_token"](a_masked)

        return self._tf_layers[f"loss.{name}_mask"](
            a_t_masked_embed,
            a_masked_embed,
            a_masked_ids,
            a_masked_embed,
            a_masked_ids,
        )

    def _label_loss(
        self, a: tf.Tensor, b: tf.Tensor, label_ids: tf.Tensor
    ) -> tf.Tensor:
        all_label_ids, all_labels_embed = self._create_all_labels()

        a_embed = self._tf_layers["embed.text"](a)
        b_embed = self._tf_layers["embed.label"](b)

        return self._tf_layers["loss.label"](
            a_embed, b_embed, label_ids, all_labels_embed, all_label_ids
        )

    def _entity_loss(
        self, a: tf.Tensor, c: tf.Tensor, mask: tf.Tensor, sequence_lengths
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        # remove cls token
        sequence_lengths = sequence_lengths - 1
        c = tf.cast(c[:, :, 0], tf.int32)

        logits = self._tf_layers["embed.logits"](a)

        loss = self._tf_layers["crf"].loss(logits, c, sequence_lengths)
        pred_ids = self._tf_layers["crf"](logits, sequence_lengths)

        # TODO check that f1 calculation is correct
        # calculate f1 score for train predictions
        mask_bool = tf.cast(mask[:, :, 0], tf.bool)
        # pick only non padding values and flatten sequences
        c_masked = tf.boolean_mask(c, mask_bool)
        pred_ids_masked = tf.boolean_mask(pred_ids, mask_bool)
        # set `0` prediction to not a prediction
        c_masked_1 = tf.one_hot(c_masked - 1, self._num_tags - 1)
        pred_ids_masked_1 = tf.one_hot(pred_ids_masked - 1, self._num_tags - 1)

        f1 = self._tf_layers["crf_f1_score"](c_masked_1, pred_ids_masked_1)

        return loss, f1

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        mask_text = tf_batch_data["text_mask"][0]
        sequence_lengths = self._get_sequence_lengths(mask_text)

        (
            text_transformed,
            text_in,
            text_seq_ids,
            lm_mask_bool_text,
        ) = self._create_sequence(
            tf_batch_data["text_features"],
            mask_text,
            self.text_name,
            self.config[MASKED_LM],
            sequence_ids=True,
        )

        losses = []

        if self.config[MASKED_LM]:
            loss, acc = self._mask_loss(
                text_transformed, text_in, text_seq_ids, lm_mask_bool_text, "text"
            )
            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        if self.config[INTENT_CLASSIFICATION]:
            # get _cls_ vector for intent classification
            cls = self._last_token(text_transformed, sequence_lengths)

            label_ids = tf_batch_data["label_ids"][0]
            label = self._create_bow(
                tf_batch_data["label_features"],
                tf_batch_data["label_mask"][0],
                self.label_name,
            )
            loss, acc = self._label_loss(cls, label, label_ids)
            self.intent_loss.update_state(loss)
            self.response_acc.update_state(acc)
            losses.append(loss)

        if self.config[ENTITY_RECOGNITION]:
            tags = tf_batch_data["tag_ids"][0]

            loss, f1 = self._entity_loss(
                text_transformed, tags, mask_text, sequence_lengths
            )
            self.entity_loss.update_state(loss)
            self.entity_f1.update_state(f1)
            losses.append(loss)

        return tf.math.add_n(losses)

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        mask_text = tf_batch_data["text_mask"][0]
        sequence_lengths = self._get_sequence_lengths(mask_text)

        text_transformed, _, _, _ = self._create_sequence(
            tf_batch_data["text_features"], mask_text, self.text_name
        )

        out = {}
        if self.config[INTENT_CLASSIFICATION]:
            if self.all_labels_embed is None:
                _, self.all_labels_embed = self._create_all_labels()

            # get _cls_ vector for intent classification
            cls = self._last_token(text_transformed, sequence_lengths)
            cls_embed = self._tf_layers["embed.text"](cls)

            sim_all = self._tf_layers["loss.label"].sim(
                cls_embed[:, tf.newaxis, :], self.all_labels_embed[tf.newaxis, :, :]
            )
            scores = self._tf_layers["loss.label"].confidence_from_sim(
                sim_all, self.config[SIMILARITY_TYPE]
            )
            out["i_scores"] = scores

        if self.config[ENTITY_RECOGNITION]:
            logits = self._tf_layers["embed.logits"](text_transformed)
            pred_ids = self._tf_layers["crf"](logits, sequence_lengths - 1)
            out["e_ids"] = pred_ids

        return out


# pytype: enable=key-error
