import logging

import numpy as np
import tensorflow as tf

from typing import Any, Dict, List, Optional, Text, Tuple, Union

from rasa.nlu.training_data import TrainingData, Message
from rasa.nlu.classifiers.diet_classifier import DIETClassifier, DIET
from rasa.nlu.components import any_of
from rasa.utils.tensorflow.constants import (
    HIDDEN_LAYERS_SIZES_TEXT,
    HIDDEN_LAYERS_SIZES_LABEL,
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
from rasa.nlu.constants import (
    RESPONSE_ATTRIBUTE,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    DEFAULT_OPEN_UTTERANCE_TYPE,
    DENSE_FEATURE_NAMES,
    TEXT_ATTRIBUTE,
    SPARSE_FEATURE_NAMES,
)
from rasa.utils.tensorflow.tf_model_data import RasaModelData
from rasa.utils.tensorflow.tf_models import RasaModel

logger = logging.getLogger(__name__)


class ResponseSelector(DIETClassifier):
    """Response selector using supervised embeddings.

    The response selector embeds user inputs
    and candidate response into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    It also provides rankings of the response that did not "win".

    The supervised response selector needs to be preceded by
    a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``CountVectorsFeaturizer`` that
    can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
    However, in this implementation the `mu` parameter is treated differently
    and additional hidden layers are added together with dropout.
    """

    provides = [RESPONSE_ATTRIBUTE, "response_ranking"]

    requires = [
        any_of(
            DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE], SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]
        ),
        any_of(
            DENSE_FEATURE_NAMES[RESPONSE_ATTRIBUTE],
            SPARSE_FEATURE_NAMES[RESPONSE_ATTRIBUTE],
        ),
    ]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        HIDDEN_LAYERS_SIZES_TEXT: [],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        HIDDEN_LAYERS_SIZES_LABEL: [],
        # Whether to share the hidden layer weights between input words and intent labels
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
        DENSE_DIM: {"text": 512, "label": 20},
        # dimension size of embedding vectors
        EMBED_DIM: 20,
        # the type of the similarity
        NUM_NEG: 20,
        # flag if minimize only maximum similarity over incorrect actions
        SIMILARITY_TYPE: "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        LOSS_TYPE: "softmax",  # string 'softmax' or 'margin'
        # number of top responses to normalize scores for softmax loss_type
        # set to 0 to turn off normalization
        RANKING_LENGTH: 10,
        # how similar the algorithm should try
        # to make embedding vectors for correct intent labels
        MU_POS: 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        MU_NEG: -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        USE_MAX_SIM_NEG: True,
        # scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # regularization parameters
        # the scale of L2 regularization
        C2: 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
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
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance,
        # if true random tokens of the input message will be masked and the model
        # should predict those tokens
        MASKED_LM: False,
        # selector config
        # name of the intent for which this response selector is to be trained
        "retrieval_intent": None,
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        inverted_tag_dict: Optional[Dict[int, Text]] = None,
        model: Optional[RasaModel] = None,
        batch_tuple_sizes: Optional[Dict] = None,
    ):
        component_config = component_config or {}

        # the following properties don't exist for the ResponseSelector
        component_config[INTENT_CLASSIFICATION] = True
        component_config[ENTITY_RECOGNITION] = None
        component_config[BILOU_FLAG] = None

        super().__init__(
            component_config,
            inverted_label_dict,
            inverted_tag_dict,
            model,
            batch_tuple_sizes,
        )

    @staticmethod
    def model_name():
        return DIET2DIET

    def _load_selector_params(self, config: Dict[Text, Any]) -> None:
        self.retrieval_intent = config["retrieval_intent"]
        if not self.retrieval_intent:
            # retrieval intent was left to its default value
            logger.info(
                "Retrieval intent parameter was left to its default value. This "
                "response selector will be trained on training examples combining "
                "all retrieval intents."
            )

    def _check_config_parameters(self) -> None:
        super()._check_config_parameters()
        self._load_selector_params(self.component_config)

    @staticmethod
    def _set_message_property(
        message: Message, prediction_dict: Dict[Text, Any], selector_key: Text
    ) -> None:

        message_selector_properties = message.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})
        message_selector_properties[selector_key] = prediction_dict
        message.set(
            RESPONSE_SELECTOR_PROPERTY_NAME,
            message_selector_properties,
            add_to_output=True,
        )

    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Performs sanity checks on training data, extracts encodings for labels
        and prepares data for training"""
        if self.retrieval_intent:
            training_data = training_data.filter_by_intent(self.retrieval_intent)

        label_id_dict = self._create_label_id_dict(
            training_data, attribute=RESPONSE_ATTRIBUTE
        )
        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}

        self._label_data = self._create_label_data(
            training_data, label_id_dict, attribute=RESPONSE_ATTRIBUTE
        )

        model_data = self._create_model_data(
            training_data.intent_examples,
            label_id_dict,
            label_attribute=RESPONSE_ATTRIBUTE,
        )

        self.check_input_dimension_consistency(model_data)

        return model_data

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely response and its similarity to the input."""

        out = self._predict(message)
        label, label_ranking = self._predict_label(out)

        selector_key = (
            self.retrieval_intent
            if self.retrieval_intent
            else DEFAULT_OPEN_UTTERANCE_TYPE
        )

        logger.debug(
            f"Adding following selector key to message property: {selector_key}"
        )

        prediction_dict = {"response": label, "ranking": label_ranking}

        self._set_message_property(message, prediction_dict, selector_key)


class DIET2DIET(DIET):
    def _prepare_layers(self) -> None:
        self._prepare_sequence_layers(self.text_name)
        self._prepare_sequence_layers(self.label_name)
        if self.config[MASKED_LM]:
            self._prepare_mask_lm_layers(self.text_name)
            self._prepare_mask_lm_layers(self.label_name)
        self._prepare_label_classification_layers()

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data["label_ids"][0]

        mask_label = self.tf_label_data["label_mask"][0]
        sequence_lengths_label = self._get_sequence_lengths(mask_label)

        label_transformed, _, _, _ = self._create_sequence(
            self.tf_label_data["label_features"], mask_label, self.label_name,
        )
        cls_label = self._last_token(label_transformed, sequence_lengths_label)

        all_labels_embed = self._tf_layers["embed.label"](cls_label)

        return all_label_ids, all_labels_embed

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        mask_text = tf_batch_data["text_mask"][0]
        sequence_lengths_text = self._get_sequence_lengths(mask_text)

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

        mask_label = tf_batch_data["label_mask"][0]
        sequence_lengths_label = self._get_sequence_lengths(mask_label)

        label_transformed, _, _, _ = self._create_sequence(
            tf_batch_data["label_features"], mask_label, self.label_name,
        )

        losses = []

        if self.config[MASKED_LM]:
            loss, acc = self._mask_loss(
                text_transformed,
                text_in,
                text_seq_ids,
                lm_mask_bool_text,
                self.text_name,
            )

            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        # get _cls_ vector for label classification
        cls_text = self._last_token(text_transformed, sequence_lengths_text)
        cls_label = self._last_token(label_transformed, sequence_lengths_label)
        label_ids = tf_batch_data["label_ids"][0]

        loss, acc = self._label_loss(cls_text, cls_label, label_ids)
        self.intent_loss.update_state(loss)
        self.intent_acc.update_state(acc)
        losses.append(loss)

        return tf.math.add_n(losses)

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        mask_text = tf_batch_data["text_mask"][0]
        sequence_lengths_text = self._get_sequence_lengths(mask_text)

        text_transformed, _, _, _ = self._create_sequence(
            tf_batch_data["text_features"], mask_text, self.text_name
        )

        out = {}

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels()

        # get _cls_ vector for intent classification
        cls = self._last_token(text_transformed, sequence_lengths_text)
        cls_embed = self._tf_layers["embed.text"](cls)

        sim_all = self._tf_layers["loss.label"].sim(
            cls_embed[:, tf.newaxis, :], self.all_labels_embed[tf.newaxis, :, :]
        )
        scores = self._tf_layers["loss.label"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )
        out["i_scores"] = scores

        return out
