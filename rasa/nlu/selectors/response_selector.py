from __future__ import annotations
import copy
import logging
from rasa.nlu.featurizers.featurizer import Featurizer

import numpy as np
import tensorflow as tf

from typing import Any, Dict, Optional, Text, Tuple, Union, List, Type

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.nlu.training_data import util
import rasa.shared.utils.io
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.classifiers.diet_classifier import (
    DIET,
    LABEL_KEY,
    LABEL_SUB_KEY,
    SENTENCE,
    SEQUENCE,
    DIETClassifier,
)
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.utils.tensorflow import rasa_layers
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    SHARE_HIDDEN_LAYERS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    RANKING_LENGTH,
    RENORMALIZE_CONFIDENCES,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    CONNECTION_DENSITY,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    RETRIEVAL_INTENT,
    USE_TEXT_AS_LABEL,
    CROSS_ENTROPY,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CONCAT_DIMENSION,
    FEATURIZERS,
    CHECKPOINT_MODEL,
    DENSE_DIMENSION,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    SOFTMAX,
)
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_PROPERTY_NAME,
    RESPONSE_SELECTOR_RETRIEVAL_INTENTS,
    RESPONSE_SELECTOR_RESPONSES_KEY,
    RESPONSE_SELECTOR_PREDICTION_KEY,
    RESPONSE_SELECTOR_RANKING_KEY,
    RESPONSE_SELECTOR_UTTER_ACTION_KEY,
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    DEFAULT_TRANSFORMER_SIZE,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    RESPONSE,
    INTENT_RESPONSE_KEY,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)

from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.models import RasaModel

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
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

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Featurizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            **DIETClassifier.get_default_config(),
            # ## Architecture of the used neural network
            # Hidden layer sizes for layers before the embedding layers for user message
            # and labels.
            # The number of hidden layers is equal to the length of the corresponding
            # list.
            HIDDEN_LAYERS_SIZES: {TEXT: [256, 128], LABEL: [256, 128]},
            # Whether to share the hidden layer weights between input words
            # and responses
            SHARE_HIDDEN_LAYERS: False,
            # Number of units in transformer
            TRANSFORMER_SIZE: None,
            # Number of transformer layers
            NUM_TRANSFORMER_LAYERS: 0,
            # Number of attention heads in transformer
            NUM_HEADS: 4,
            # If 'True' use key relative embeddings in attention
            KEY_RELATIVE_ATTENTION: False,
            # If 'True' use key relative embeddings in attention
            VALUE_RELATIVE_ATTENTION: False,
            # Max position for relative embeddings. Only in effect if key-
            # or value relative attention are turned on
            MAX_RELATIVE_POSITION: 5,
            # Use a unidirectional or bidirectional encoder.
            UNIDIRECTIONAL_ENCODER: False,
            # ## Training parameters
            # Initial and final batch sizes:
            # Batch size will be linearly increased for each epoch.
            BATCH_SIZES: [64, 256],
            # Strategy used when creating batches.
            # Can be either 'sequence' or 'balanced'.
            BATCH_STRATEGY: BALANCED,
            # Number of epochs to train
            EPOCHS: 300,
            # Set random seed to any 'int' to get reproducible results
            RANDOM_SEED: None,
            # Initial learning rate for the optimizer
            LEARNING_RATE: 0.001,
            # ## Parameters for embeddings
            # Dimension size of embedding vectors
            EMBEDDING_DIMENSION: 20,
            # Default dense dimension to use if no dense features are present.
            DENSE_DIMENSION: {TEXT: 512, LABEL: 512},
            # Default dimension to use for concatenating sequence and sentence features.
            CONCAT_DIMENSION: {TEXT: 512, LABEL: 512},
            # The number of incorrect labels. The algorithm will minimize
            # their similarity to the user input during training.
            NUM_NEG: 20,
            # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
            SIMILARITY_TYPE: AUTO,
            # The type of the loss function, either 'cross_entropy' or 'margin'.
            LOSS_TYPE: CROSS_ENTROPY,
            # Number of top actions for which confidences should be predicted.
            # Set to 0 if confidences for all intents should be reported.
            RANKING_LENGTH: 10,
            # Determines whether the confidences of the chosen top actions should be
            # renormalized so that they sum up to 1. By default, we do not renormalize
            # and return the confidences for the top actions as is.
            # Note that renormalization only makes sense if confidences are generated
            # via `softmax`.
            RENORMALIZE_CONFIDENCES: False,
            # Indicates how similar the algorithm should try to make embedding vectors
            # for correct labels.
            # Should be 0.0 < ... < 1.0 for 'cosine' similarity type.
            MAX_POS_SIM: 0.8,
            # Maximum negative similarity for incorrect labels.
            # Should be -1.0 < ... < 1.0 for 'cosine' similarity type.
            MAX_NEG_SIM: -0.4,
            # If 'True' the algorithm only minimizes maximum similarity over
            # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
            USE_MAX_NEG_SIM: True,
            # Scale loss inverse proportionally to confidence of correct prediction
            SCALE_LOSS: True,
            # ## Regularization parameters
            # The scale of regularization
            REGULARIZATION_CONSTANT: 0.002,
            # Fraction of trainable weights in internal layers.
            CONNECTION_DENSITY: 1.0,
            # The scale of how important is to minimize the maximum similarity
            # between embeddings of different labels.
            NEGATIVE_MARGIN_SCALE: 0.8,
            # Dropout rate for encoder
            DROP_RATE: 0.2,
            # Dropout rate for attention
            DROP_RATE_ATTENTION: 0,
            # If 'True' apply dropout to sparse input tensors
            SPARSE_INPUT_DROPOUT: False,
            # If 'True' apply dropout to dense input tensors
            DENSE_INPUT_DROPOUT: False,
            # ## Evaluation parameters
            # How often calculate validation accuracy.
            # Small values may hurt performance, e.g. model accuracy.
            EVAL_NUM_EPOCHS: 20,
            # How many examples to use for hold out validation set
            # Large values may hurt performance, e.g. model accuracy.
            EVAL_NUM_EXAMPLES: 0,
            # ## Selector config
            # If 'True' random tokens of the input message will be masked and the model
            # should predict those tokens.
            MASKED_LM: False,
            # Name of the intent for which this response selector is to be trained
            RETRIEVAL_INTENT: None,
            # Boolean flag to check if actual text of the response
            # should be used as ground truth label for training the model.
            USE_TEXT_AS_LABEL: False,
            # If you want to use tensorboard to visualize training
            # and validation metrics,
            # set this option to a valid output directory.
            TENSORBOARD_LOG_DIR: None,
            # Define when training metrics for tensorboard should be logged.
            # Either after every epoch or for every training step.
            # Valid values: 'epoch' and 'batch'
            TENSORBOARD_LOG_LEVEL: "epoch",
            # Specify what features to use as sequence and sentence features
            # By default all features in the pipeline are used.
            FEATURIZERS: [],
            # Perform model checkpointing
            CHECKPOINT_MODEL: False,
            # if 'True' applies sigmoid on all similarity terms and adds it
            # to the loss function to ensure that similarity values are
            # approximately bounded. Used inside cross-entropy loss only.
            CONSTRAIN_SIMILARITIES: False,
            # Model confidence to be returned during inference. Currently, the only
            # possible value is `softmax`.
            MODEL_CONFIDENCE: SOFTMAX,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        model: Optional[RasaModel] = None,
        all_retrieval_intents: Optional[List[Text]] = None,
        responses: Optional[Dict[Text, List[Dict[Text, Any]]]] = None,
        sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None,
    ) -> None:
        """Declare instance variables with default values.

        Args:
            config: Configuration for the component.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            index_label_id_mapping: Mapping between label and index used for encoding.
            entity_tag_specs: Format specification all entity tags.
            model: Model architecture.
            all_retrieval_intents: All retrieval intents defined in the data.
            responses: All responses defined in the data.
            finetune_mode: If `True` loads the model with pre-trained weights,
                otherwise initializes it with random weights.
            sparse_feature_sizes: Sizes of the sparse features the model was trained on.
        """
        component_config = config

        # the following properties cannot be adapted for the ResponseSelector
        component_config[INTENT_CLASSIFICATION] = True
        component_config[ENTITY_RECOGNITION] = False
        component_config[BILOU_FLAG] = None

        # Initialize defaults
        self.responses = responses or {}
        self.all_retrieval_intents = all_retrieval_intents or []
        self.retrieval_intent = None
        self.use_text_as_label = False

        super().__init__(
            component_config,
            model_storage,
            resource,
            execution_context,
            index_label_id_mapping,
            entity_tag_specs,
            model,
            sparse_feature_sizes=sparse_feature_sizes,
        )

    @property
    def label_key(self) -> Text:
        """Returns label key."""
        return LABEL_KEY

    @property
    def label_sub_key(self) -> Text:
        """Returns label sub_key."""
        return LABEL_SUB_KEY

    @staticmethod
    def model_class(  # type: ignore[override]
        use_text_as_label: bool,
    ) -> Type[RasaModel]:
        """Returns model class."""
        if use_text_as_label:
            return DIET2DIET
        else:
            return DIET2BOW

    def _load_selector_params(self) -> None:
        self.retrieval_intent = self.component_config[RETRIEVAL_INTENT]
        self.use_text_as_label = self.component_config[USE_TEXT_AS_LABEL]

    def _warn_about_transformer_and_hidden_layers_enabled(
        self, selector_name: Text
    ) -> None:
        """Warns user if they enabled the transformer but didn't disable hidden layers.

        ResponseSelector defaults specify considerable hidden layer sizes, but
        this is for cases where no transformer is used. If a transformer exists,
        then, from our experience, the best results are achieved with no hidden layers
        used between the feature-combining layers and the transformer.
        """
        default_config = self.get_default_config()
        hidden_layers_is_at_default_value = (
            self.component_config[HIDDEN_LAYERS_SIZES]
            == default_config[HIDDEN_LAYERS_SIZES]
        )
        config_for_disabling_hidden_layers: Dict[Text, List[Any]] = {
            k: [] for k, _ in default_config[HIDDEN_LAYERS_SIZES].items()
        }
        # warn if the hidden layers aren't disabled
        if (
            self.component_config[HIDDEN_LAYERS_SIZES]
            != config_for_disabling_hidden_layers
        ):
            # make the warning text more contextual by explaining what the user did
            # to the hidden layers' config (i.e. what it is they should change)
            if hidden_layers_is_at_default_value:
                what_user_did = "left the hidden layer sizes at their default value:"
            else:
                what_user_did = "set the hidden layer sizes to be non-empty by setting"

            rasa.shared.utils.io.raise_warning(
                f"You have enabled a transformer inside {selector_name} by"
                f" setting a positive value for `{NUM_TRANSFORMER_LAYERS}`, but you "
                f"{what_user_did} `{HIDDEN_LAYERS_SIZES}="
                f"{self.component_config[HIDDEN_LAYERS_SIZES]}`. We recommend to "
                f"disable the hidden layers when using a transformer, by specifying "
                f"`{HIDDEN_LAYERS_SIZES}={config_for_disabling_hidden_layers}`.",
                category=UserWarning,
            )

    def _warn_and_correct_transformer_size(self, selector_name: Text) -> None:
        """Corrects transformer size so that training doesn't break; informs the user.

        If a transformer is used, the default `transformer_size` breaks things.
        We need to set a reasonable default value so that the model works fine.
        """
        if (
            self.component_config[TRANSFORMER_SIZE] is None
            or self.component_config[TRANSFORMER_SIZE] < 1
        ):
            rasa.shared.utils.io.raise_warning(
                f"`{TRANSFORMER_SIZE}` is set to "
                f"`{self.component_config[TRANSFORMER_SIZE]}` for "
                f"{selector_name}, but a positive size is required when using "
                f"`{NUM_TRANSFORMER_LAYERS} > 0`. {selector_name} will proceed, using "
                f"`{TRANSFORMER_SIZE}={DEFAULT_TRANSFORMER_SIZE}`. "
                f"Alternatively, specify a different value in the component's config.",
                category=UserWarning,
            )
            self.component_config[TRANSFORMER_SIZE] = DEFAULT_TRANSFORMER_SIZE

    def _check_config_params_when_transformer_enabled(self) -> None:
        """Checks & corrects config parameters when the transformer is enabled.

        This is needed because the defaults for individual config parameters are
        interdependent and some defaults should change when the transformer is enabled.
        """
        if self.component_config[NUM_TRANSFORMER_LAYERS] > 0:
            selector_name = "ResponseSelector" + (
                f"({self.retrieval_intent})" if self.retrieval_intent else ""
            )
            self._warn_about_transformer_and_hidden_layers_enabled(selector_name)
            self._warn_and_correct_transformer_size(selector_name)

    def _check_config_parameters(self) -> None:
        """Checks that component configuration makes sense; corrects it where needed."""
        super()._check_config_parameters()
        self._load_selector_params()
        # Once general DIET-related parameters have been checked, check also the ones
        # specific to ResponseSelector.
        self._check_config_params_when_transformer_enabled()

    def _set_message_property(
        self, message: Message, prediction_dict: Dict[Text, Any], selector_key: Text
    ) -> None:
        message_selector_properties = message.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})
        message_selector_properties[
            RESPONSE_SELECTOR_RETRIEVAL_INTENTS
        ] = self.all_retrieval_intents
        message_selector_properties[selector_key] = prediction_dict
        message.set(
            RESPONSE_SELECTOR_PROPERTY_NAME,
            message_selector_properties,
            add_to_output=True,
        )

    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.

        Args:
            training_data: training data to preprocessed.
        """
        # Collect all retrieval intents present in the data before filtering
        self.all_retrieval_intents = list(training_data.retrieval_intents)

        if self.retrieval_intent:
            training_data = training_data.filter_training_examples(
                lambda ex: self.retrieval_intent == ex.get(INTENT)
            )
        else:
            # retrieval intent was left to its default value
            logger.info(
                "Retrieval intent parameter was left to its default value. This "
                "response selector will be trained on training examples combining "
                "all retrieval intents."
            )

        label_attribute = RESPONSE if self.use_text_as_label else INTENT_RESPONSE_KEY

        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=label_attribute
        )

        self.responses = training_data.responses

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=label_attribute
        )

        model_data = self._create_model_data(
            training_data.intent_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
        )

        self._check_input_dimension_consistency(model_data)

        return model_data

    def _resolve_intent_response_key(
        self, label: Dict[Text, Optional[Text]]
    ) -> Optional[Text]:
        """Given a label, return the response key based on the label id.

        Args:
            label: predicted label by the selector

        Returns:
            The match for the label that was found in the known responses.
            It is always guaranteed to have a match, otherwise that case should have
            been caught earlier and a warning should have been raised.
        """
        for key, responses in self.responses.items():

            # First check if the predicted label was the key itself
            search_key = util.template_key_to_intent_response_key(key)
            if search_key == label.get("name"):
                return search_key

            # Otherwise loop over the responses to check if the text has a direct match
            for response in responses:
                if response.get(TEXT, "") == label.get("name"):
                    return search_key
        return None

    def process(self, messages: List[Message]) -> List[Message]:
        """Selects most like response for message.

        Args:
            messages: List containing latest user message.

        Returns:
            List containing the message augmented with the most likely response,
            the associated intent_response_key and its similarity to the input.
        """
        for message in messages:
            out = self._predict(message)
            top_label, label_ranking = self._predict_label(out)

            # Get the exact intent_response_key and the associated
            # responses for the top predicted label
            label_intent_response_key = (
                self._resolve_intent_response_key(top_label)
                or top_label[INTENT_NAME_KEY]
            )
            label_responses = self.responses.get(
                util.intent_response_key_to_template_key(label_intent_response_key)
            )

            if label_intent_response_key and not label_responses:
                # responses seem to be unavailable,
                # likely an issue with the training data
                # we'll use a fallback instead
                rasa.shared.utils.io.raise_warning(
                    f"Unable to fetch responses for {label_intent_response_key} "
                    f"This means that there is likely an issue with the training data."
                    f"Please make sure you have added responses for this intent."
                )
                label_responses = [{TEXT: label_intent_response_key}]

            for label in label_ranking:
                label[INTENT_RESPONSE_KEY] = (
                    self._resolve_intent_response_key(label) or label[INTENT_NAME_KEY]
                )
                # Remove the "name" key since it is either the same as
                # "intent_response_key" or it is the response text which
                # is not needed in the ranking.
                label.pop(INTENT_NAME_KEY)

            selector_key = (
                self.retrieval_intent
                if self.retrieval_intent
                else RESPONSE_SELECTOR_DEFAULT_INTENT
            )

            logger.debug(
                f"Adding following selector key to message property: {selector_key}"
            )

            utter_action_key = util.intent_response_key_to_template_key(
                label_intent_response_key
            )
            prediction_dict = {
                RESPONSE_SELECTOR_PREDICTION_KEY: {
                    RESPONSE_SELECTOR_RESPONSES_KEY: label_responses,
                    PREDICTED_CONFIDENCE_KEY: top_label[PREDICTED_CONFIDENCE_KEY],
                    INTENT_RESPONSE_KEY: label_intent_response_key,
                    RESPONSE_SELECTOR_UTTER_ACTION_KEY: utter_action_key,
                },
                RESPONSE_SELECTOR_RANKING_KEY: label_ranking,
            }

            self._set_message_property(message, prediction_dict, selector_key)

            if (
                self._execution_context.should_add_diagnostic_data
                and out
                and DIAGNOSTIC_DATA in out
            ):
                message.add_diagnostic_data(
                    self._execution_context.node_name, out.get(DIAGNOSTIC_DATA)
                )

        return messages

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        if self.model is None:
            return None

        with self._model_storage.write_to(self._resource) as model_path:
            file_name = self.__class__.__name__

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.responses.json", self.responses
            )

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.retrieval_intents.json",
                self.all_retrieval_intents,
            )

        super().persist()

    @classmethod
    def _load_model_class(
        cls,
        tf_model_file: Text,
        model_data_example: RasaModelData,
        label_data: RasaModelData,
        entity_tag_specs: List[EntityTagSpec],
        config: Dict[Text, Any],
        finetune_mode: bool = False,
    ) -> "RasaModel":

        predict_data_example = RasaModelData(
            label_key=model_data_example.label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )
        return cls.model_class(config[USE_TEXT_AS_LABEL]).load(
            tf_model_file,
            model_data_example,
            predict_data_example,
            data_signature=model_data_example.get_signature(),
            label_data=label_data,
            entity_tag_specs=entity_tag_specs,
            config=copy.deepcopy(config),
            finetune_mode=finetune_mode,
        )

    def _instantiate_model_class(self, model_data: RasaModelData) -> "RasaModel":
        return self.model_class(self.use_text_as_label)(
            data_signature=model_data.get_signature(),
            label_data=self._label_data,
            entity_tag_specs=self._entity_tag_specs,
            config=self.component_config,
        )

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> ResponseSelector:
        """Loads the trained model from the provided directory."""
        model = super().load(
            config, model_storage, resource, execution_context, **kwargs
        )

        try:
            with model_storage.read_from(resource) as model_path:
                file_name = cls.__name__
                responses = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.responses.json"
                )
                all_retrieval_intents = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.retrieval_intents.json"
                )
                model.responses = responses
                model.all_retrieval_intents = all_retrieval_intents
                return model
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource, execution_context)


class DIET2BOW(DIET):
    """DIET2BOW transformer implementation."""

    def _create_metrics(self) -> None:
        # self.metrics preserve order
        # output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.response_loss = tf.keras.metrics.Mean(name="r_loss")
        # output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.response_acc = tf.keras.metrics.Mean(name="r_acc")

    def _update_metrics_to_log(self) -> None:
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG

        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            if debug_log_level:
                self.metrics_to_log.append("m_loss")

        self.metrics_to_log.append("r_acc")
        if debug_log_level:
            self.metrics_to_log.append("r_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {"t": "total", "m": "mask", "r": "response"}
        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _update_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:

        self.response_loss.update_state(loss)
        self.response_acc.update_state(acc)


class DIET2DIET(DIET):
    """Diet 2 Diet transformer implementation."""

    def _check_data(self) -> None:
        if TEXT not in self.data_signature:
            raise InvalidConfigException(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL not in self.data_signature:
            raise InvalidConfigException(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if (
            self.config[SHARE_HIDDEN_LAYERS]
            and self.data_signature[TEXT][SENTENCE]
            != self.data_signature[LABEL][SENTENCE]
        ):
            raise ValueError(
                "If hidden layer weights are shared, data signatures "
                "for text_features and label_features must coincide."
            )

    def _create_metrics(self) -> None:
        # self.metrics preserve order
        # output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.response_loss = tf.keras.metrics.Mean(name="r_loss")
        # output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.response_acc = tf.keras.metrics.Mean(name="r_acc")

    def _update_metrics_to_log(self) -> None:
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG

        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            if debug_log_level:
                self.metrics_to_log.append("m_loss")

        self.metrics_to_log.append("r_acc")
        if debug_log_level:
            self.metrics_to_log.append("r_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {"t": "total", "m": "mask", "r": "response"}
        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _prepare_layers(self) -> None:
        self.text_name = TEXT
        self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL

        # For user text and response text, prepare layers that combine different feature
        # types, embed everything using a transformer and optionally also do masked
        # language modeling. Omit input dropout for label features.
        label_config = self.config.copy()
        label_config.update({SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False})
        for attribute, config in [
            (self.text_name, self.config),
            (self.label_name, label_config),
        ]:
            self._tf_layers[
                f"sequence_layer.{attribute}"
            ] = rasa_layers.RasaSequenceLayer(
                attribute, self.data_signature[attribute], config
            )

        if self.config[MASKED_LM]:
            self._prepare_mask_lm_loss(self.text_name)

        self._prepare_label_classification_layers(predictor_attribute=self.text_name)

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            self.tf_label_data, LABEL
        )

        # Combine all feature types into one and embed using a transformer.
        label_transformed, _, _, _, _, _ = self._tf_layers[
            f"sequence_layer.{self.label_name}"
        ](
            (
                self.tf_label_data[LABEL][SEQUENCE],
                self.tf_label_data[LABEL][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )

        # Last token is taken from the last position with real features, determined
        # - by the number of real tokens, i.e. by the sequence length of sequence-level
        #   features, and
        # - by the presence or absence of sentence-level features (reflected in the
        #   effective sequence length of these features being 1 or 0.
        # We need to combine the two lengths to correctly get the last position.
        sentence_feature_lengths = self._get_sentence_feature_lengths(
            self.tf_label_data, LABEL
        )
        sentence_label = self._last_token(
            label_transformed, sequence_feature_lengths + sentence_feature_lengths
        )

        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](sentence_label)

        return all_label_ids, all_labels_embed

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        # Process all features for text.
        sequence_feature_lengths_text = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )
        (
            text_transformed,
            text_in,
            _,
            text_seq_ids,
            mlm_mask_booleanean_text,
            _,
        ) = self._tf_layers[f"sequence_layer.{self.text_name}"](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths_text,
            ),
            training=self._training,
        )

        # Process all features for labels.
        sequence_feature_lengths_label = self._get_sequence_feature_lengths(
            tf_batch_data, LABEL
        )
        label_transformed, _, _, _, _, _ = self._tf_layers[
            f"sequence_layer.{self.label_name}"
        ](
            (
                tf_batch_data[LABEL][SEQUENCE],
                tf_batch_data[LABEL][SENTENCE],
                sequence_feature_lengths_label,
            ),
            training=self._training,
        )

        losses = []

        if self.config[MASKED_LM]:
            loss, acc = self._mask_loss(
                text_transformed,
                text_in,
                text_seq_ids,
                mlm_mask_booleanean_text,
                self.text_name,
            )

            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        # Get sentence feature vector for label classification. The vector is extracted
        # from the last position with real features. To determine this position, we
        # combine the sequence lengths of sequence- and sentence-level features.
        sentence_feature_lengths_text = self._get_sentence_feature_lengths(
            tf_batch_data, TEXT
        )
        sentence_vector_text = self._last_token(
            text_transformed,
            sequence_feature_lengths_text + sentence_feature_lengths_text,
        )

        # Extract sentence vector for the label attribute in the same way.
        sentence_feature_lengths_label = self._get_sentence_feature_lengths(
            tf_batch_data, LABEL
        )
        sentence_vector_label = self._last_token(
            label_transformed,
            sequence_feature_lengths_label + sentence_feature_lengths_label,
        )
        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]

        loss, acc = self._calculate_label_loss(
            sentence_vector_text, sentence_vector_label, label_ids
        )
        self.response_loss.update_state(loss)
        self.response_acc.update_state(acc)
        losses.append(loss)

        return tf.math.add_n(losses)

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        sequence_feature_lengths = self._get_sequence_feature_lengths(
            tf_batch_data, TEXT
        )
        text_transformed, _, _, _, _, attention_weights = self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )

        predictions = {
            DIAGNOSTIC_DATA: {
                "attention_weights": attention_weights,
                "text_transformed": text_transformed,
            }
        }

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels()

        # get sentence feature vector for intent classification
        sentence_vector = self._last_token(text_transformed, sequence_feature_lengths)
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)

        _, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
        )
        predictions["i_scores"] = scores

        return predictions
