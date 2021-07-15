import tensorflow as tf
import numpy as np
import logging
import random
from collections import defaultdict
from typing import (
    List,
    Text,
    Dict,
    Tuple,
    Union,
    Optional,
    Any,
)

from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.utils.tensorflow.constants import (
    LABEL,
    IDS,
    INTENT_CLASSIFICATION,
    SENTENCE,
    SEQUENCE_LENGTH,
    RANDOM_SEED,
    EMBEDDING_DIMENSION,
    REGULARIZATION_CONSTANT,
    SIMILARITY_TYPE,
    CONNECTION_DENSITY,
    NUM_NEG,
    LOSS_TYPE,
    MAX_POS_SIM,
    MAX_NEG_SIM,
    USE_MAX_NEG_SIM,
    NEGATIVE_MARGIN_SCALE,
    SCALE_LOSS,
    LEARNING_RATE,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
)
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureSignature,
    FeatureArray,
)
import rasa.utils.train_utils
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow import rasa_layers
from rasa.utils.tensorflow.temp_keras_modules import TmpKerasModel
from rasa.utils.tensorflow.data_generator import (
    RasaDataGenerator,
    RasaBatchDataGenerator,
)
from tensorflow.python.keras.utils import tf_utils
from rasa.shared.nlu.constants import TEXT
from rasa.shared.exceptions import RasaException

logger = logging.getLogger(__name__)

LABEL_KEY = LABEL
LABEL_SUB_KEY = IDS


# noinspection PyMethodOverriding
class RasaModel(TmpKerasModel):
    """Abstract custom Keras model.

     This model overwrites the following methods:
    - train_step
    - test_step
    - predict_step
    - save
    - load
    Cannot be used as tf.keras.Model.
    """

    def __init__(self, random_seed: Optional[int] = None, **kwargs: Any) -> None:
        """Initialize the RasaModel.

        Args:
            random_seed: set the random seed to get reproducible results
        """
        # make sure that keras releases resources from previously trained model
        tf.keras.backend.clear_session()
        super().__init__(**kwargs)

        self.total_loss = tf.keras.metrics.Mean(name="t_loss")
        self.metrics_to_log = ["t_loss"]

        self._training = None  # training phase should be defined when building a graph

        self.random_seed = random_seed
        self._set_random_seed()

        self._tf_predict_step = None
        self.prepared_for_prediction = False

    def _set_random_seed(self) -> None:
        random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        raise NotImplementedError

    def prepare_for_predict(self) -> None:
        """Prepares tf graph fpr prediction.

        This method should contain necessary tf calculations
        and set self variables that are used in `batch_predict`.
        For example, pre calculation of `self.all_labels_embed`.
        """
        pass

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        raise NotImplementedError

    def train_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, float]:
        """Performs a train step using the given batch.

        Args:
            batch_in: The batch input.

        Returns:
            Training metrics.
        """
        self._training = True

        # calculate supervision and regularization losses separately
        with tf.GradientTape(persistent=True) as tape:
            prediction_loss = self.batch_loss(batch_in)
            regularization_loss = tf.math.add_n(self.losses)
            total_loss = prediction_loss + regularization_loss

        self.total_loss.update_state(total_loss)

        # calculate the gradients that come from supervision signal
        prediction_gradients = tape.gradient(prediction_loss, self.trainable_variables)
        # calculate the gradients that come from regularization
        regularization_gradients = tape.gradient(
            regularization_loss, self.trainable_variables
        )
        # delete gradient tape manually
        # since it was created with `persistent=True` option
        del tape

        gradients = []
        for pred_grad, reg_grad in zip(prediction_gradients, regularization_gradients):
            if pred_grad is not None and reg_grad is not None:
                # remove regularization gradient for variables
                # that don't have prediction gradient
                gradients.append(
                    pred_grad
                    + tf.where(pred_grad > 0, reg_grad, tf.zeros_like(reg_grad))
                )
            else:
                gradients.append(pred_grad)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._training = None

        return self._get_metric_results()

    def test_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, float]:
        """Tests the model using the given batch.

        This method is used during validation.

        Args:
            batch_in: The batch input.

        Returns:
            Testing metrics.
        """
        self._training = False

        prediction_loss = self.batch_loss(batch_in)
        regularization_loss = tf.math.add_n(self.losses)
        total_loss = prediction_loss + regularization_loss
        self.total_loss.update_state(total_loss)

        self._training = None

        return self._get_metric_results()

    def predict_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        """Predicts the output for the given batch.

        Args:
            batch_in: The batch to predict.

        Returns:
            Prediction output.
        """
        self._training = False

        if not self.prepared_for_prediction:
            # in case the model is used for prediction without loading, e.g. directly
            # after training, we need to prepare the model for prediction once
            self.prepare_for_predict()
            self.prepared_for_prediction = True

        return self.batch_predict(batch_in)

    @staticmethod
    def _dynamic_signature(
        batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> List[List[tf.TensorSpec]]:
        element_spec = []
        for tensor in batch_in:
            if len(tensor.shape) > 1:
                shape = [None] * (len(tensor.shape) - 1) + [tensor.shape[-1]]
            else:
                shape = [None]
            element_spec.append(tf.TensorSpec(shape, tensor.dtype))
        # batch_in is a list of tensors, therefore we need to wrap element_spec into
        # the list
        return [element_spec]

    def _rasa_predict(
        self, batch_in: Tuple[np.ndarray]
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]:
        """Custom prediction method that builds tf graph on the first call.

        Args:
            batch_in: Prepared batch ready for input to `predict_step` method of model.

        Return:
            Prediction output, including diagnostic data.
        """
        self._training = False
        if not self.prepared_for_prediction:
            # in case the model is used for prediction without loading, e.g. directly
            # after training, we need to prepare the model for prediction once
            self.prepare_for_predict()
            self.prepared_for_prediction = True

        if self._run_eagerly:
            outputs = tf_utils.to_numpy_or_python_type(self.predict_step(batch_in))
            if DIAGNOSTIC_DATA in outputs:
                outputs[DIAGNOSTIC_DATA] = self._empty_lists_to_none_in_dict(
                    outputs[DIAGNOSTIC_DATA]
                )
            return outputs

        if self._tf_predict_step is None:
            self._tf_predict_step = tf.function(
                self.predict_step, input_signature=self._dynamic_signature(batch_in)
            )

        outputs = tf_utils.to_numpy_or_python_type(self._tf_predict_step(batch_in))
        if DIAGNOSTIC_DATA in outputs:
            outputs[DIAGNOSTIC_DATA] = self._empty_lists_to_none_in_dict(
                outputs[DIAGNOSTIC_DATA]
            )
        return outputs

    def run_inference(
        self,
        model_data: RasaModelData,
        batch_size: Union[int, List[int]] = 1,
        output_keys_expected: Optional[List[Text]] = None,
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]:
        """Implements bulk inferencing through the model.

        Args:
            model_data: Input data to be fed to the model.
            batch_size: Size of batches that the generator should create.
            output_keys_expected: Keys which are expected in the output.
                The output should be filtered to have only these keys before
                merging it with the output across all batches.

        Returns:
            Model outputs corresponding to the inputs fed.
        """
        outputs = {}
        (data_generator, _,) = rasa.utils.train_utils.create_data_generators(
            model_data=model_data, batch_sizes=batch_size, epochs=1, shuffle=False,
        )
        data_iterator = iter(data_generator)
        while True:
            try:
                # data_generator is a tuple of 2 elements - input and output.
                # We only need input, since output is always None and not
                # consumed by our TF graphs.
                batch_in = next(data_iterator)[0]
                batch_out: Dict[
                    Text, Union[np.ndarray, Dict[Text, Any]]
                ] = self._rasa_predict(batch_in)
                if output_keys_expected:
                    batch_out = {
                        key: output
                        for key, output in batch_out.items()
                        if key in output_keys_expected
                    }
                outputs = self._merge_batch_outputs(outputs, batch_out)
            except StopIteration:
                # Generator ran out of batches, time to finish inferencing
                break
        return outputs

    @staticmethod
    def _merge_batch_outputs(
        all_outputs: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
        batch_output: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]]:
        """Merges a batch's output into the output for all batches.

        Function assumes that the schema of batch output remains the same,
        i.e. keys and their value types do not change from one batch's
        output to another.

        Args:
            all_outputs: Existing output for all previous batches.
            batch_output: Output for a batch.

        Returns:
            Merged output with the output for current batch stacked
            below the output for all previous batches.
        """
        if not all_outputs:
            return batch_output
        for key, val in batch_output.items():
            if isinstance(val, np.ndarray):
                all_outputs[key] = np.concatenate(
                    [all_outputs[key], batch_output[key]], axis=0
                )

            elif isinstance(val, dict):
                # recurse and merge the inner dict first
                all_outputs[key] = RasaModel._merge_batch_outputs(all_outputs[key], val)

        return all_outputs

    @staticmethod
    def _empty_lists_to_none_in_dict(input_dict: Dict[Text, Any]) -> Dict[Text, Any]:
        """Recursively replaces empty list or np array with None in a dictionary."""

        def _recurse(x: Union[Dict[Text, Any], List[Any], np.ndarray]) -> Optional[Any]:
            if isinstance(x, dict):
                return {k: _recurse(v) for k, v in x.items()}
            elif (isinstance(x, list) or isinstance(x, np.ndarray)) and np.size(x) == 0:
                return None
            return x

        return _recurse(input_dict)

    def _get_metric_results(self, prefix: Optional[Text] = "") -> Dict[Text, float]:
        return {
            f"{prefix}{metric.name}": metric.result()
            for metric in self.metrics
            if metric.name in self.metrics_to_log
        }

    def save(self, model_file_name: Text, overwrite: bool = True) -> None:
        """Save the model to the given file.

        Args:
            model_file_name: The file name to save the model to.
            overwrite: If 'True' an already existing model with the same file name will
                       be overwritten.
        """
        self.save_weights(model_file_name, overwrite=overwrite, save_format="tf")

    @classmethod
    def load(
        cls,
        model_file_name: Text,
        model_data_example: RasaModelData,
        predict_data_example: Optional[RasaModelData] = None,
        finetune_mode: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> "RasaModel":
        """Loads a model from the given weights.

        Args:
            model_file_name: Path to file containing model weights.
            model_data_example: Example data point to construct the model architecture.
            predict_data_example: Example data point to speed up prediction during
              inference.
            finetune_mode: Indicates whether to load the model for further finetuning.
            *args: Any other non key-worded arguments.
            **kwargs: Any other key-worded arguments.

        Returns:
            Loaded model with weights appropriately set.
        """
        logger.debug(
            f"Loading the model from {model_file_name} "
            f"with finetune_mode={finetune_mode}..."
        )
        # create empty model
        model = cls(*args, **kwargs)
        learning_rate = kwargs.get("config", {}).get(LEARNING_RATE, 0.001)
        # need to train on 1 example to build weights of the correct size
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
        data_generator = RasaBatchDataGenerator(model_data_example, batch_size=1)
        model.fit(data_generator, verbose=False)
        # load trained weights
        model.load_weights(model_file_name)

        # predict on one data example to speed up prediction during inference
        # the first prediction always takes a bit longer to trace tf function
        if not finetune_mode and predict_data_example:
            model.run_inference(predict_data_example)

        logger.debug("Finished loading the model.")
        return model

    @staticmethod
    def batch_to_model_data_format(
        batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]],
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
    ) -> Dict[Text, Dict[Text, List[tf.Tensor]]]:
        """Convert input batch tensors into batch data format.

        Batch contains any number of batch data. The order is equal to the
        key-value pairs in session data. As sparse data were converted into (indices,
        data, shape) before, this method converts them into sparse tensors. Dense
        data is kept.
        """
        # during training batch is a tuple of input and target data
        # as our target data is inside the input data, we are just interested in the
        # input data
        if isinstance(batch[0], Tuple):
            batch = batch[0]

        batch_data = defaultdict(lambda: defaultdict(list))

        idx = 0
        for key, values in data_signature.items():
            for sub_key, signature in values.items():
                for is_sparse, feature_dimension, number_of_dimensions in signature:
                    # we converted all 4D features to 3D features before
                    number_of_dimensions = (
                        number_of_dimensions if number_of_dimensions != 4 else 3
                    )
                    if is_sparse:
                        tensor, idx = RasaModel._convert_sparse_features(
                            batch, feature_dimension, idx, number_of_dimensions
                        )
                    else:
                        tensor, idx = RasaModel._convert_dense_features(
                            batch, feature_dimension, idx, number_of_dimensions
                        )
                    batch_data[key][sub_key].append(tensor)

        return batch_data

    @staticmethod
    def _convert_dense_features(
        batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]],
        feature_dimension: int,
        idx: int,
        number_of_dimensions: int,
    ) -> Tuple[tf.Tensor, int]:
        if isinstance(batch[idx], tf.Tensor):
            # explicitly substitute last dimension in shape with known
            # static value
            if number_of_dimensions > 1 and (
                batch[idx].shape is None or batch[idx].shape[-1] is None
            ):
                shape: List[Optional[int]] = [None] * (number_of_dimensions - 1)
                shape.append(feature_dimension)
                batch[idx].set_shape(shape)

            return batch[idx], idx + 1

        # convert to Tensor
        return (
            tf.constant(batch[idx], dtype=tf.float32, shape=batch[idx].shape),
            idx + 1,
        )

    @staticmethod
    def _convert_sparse_features(
        batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]],
        feature_dimension: int,
        idx: int,
        number_of_dimensions: int,
    ) -> Tuple[tf.SparseTensor, int]:
        # explicitly substitute last dimension in shape with known
        # static value
        shape = [batch[idx + 2][i] for i in range(number_of_dimensions - 1)] + [
            feature_dimension
        ]
        return tf.SparseTensor(batch[idx], batch[idx + 1], shape), idx + 3

    def call(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]],
        training: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Calls the model on new inputs.

        Arguments:
            inputs: A tensor or list of tensors.
            training: Boolean or boolean scalar tensor, indicating whether to run
              the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        # This method needs to be implemented, otherwise the super class is raising a
        # NotImplementedError('When subclassing the `Model` class, you should
        #   implement a `call` method.')
        pass


# noinspection PyMethodOverriding
class TransformerRasaModel(RasaModel):
    def __init__(
        self,
        name: Text,
        config: Dict[Text, Any],
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        label_data: RasaModelData,
    ) -> None:
        super().__init__(
            name=name, random_seed=config[RANDOM_SEED],
        )

        self.config = config
        self.data_signature = data_signature
        self.label_signature = label_data.get_signature()
        self._check_data()

        label_batch = RasaDataGenerator.prepare_batch(label_data.data)
        self.tf_label_data = self.batch_to_model_data_format(
            label_batch, self.label_signature
        )

        # set up tf layers
        self._tf_layers: Dict[Text, tf.keras.layers.Layer] = {}

    def adjust_for_incremental_training(
        self,
        data_example: Dict[Text, Dict[Text, List[FeatureArray]]],
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    ) -> None:
        """Adjusts the model for incremental training.

        First we should check if any of the sparse feature sizes has decreased
        and raise an exception if this happens.
        If none of them have decreased and any of them has increased, then the
        function updates `DenseForSparse` layers, compiles the model, fits a sample
        data on it to activate adjusted layer(s) and updates the data signatures.

        New and old sparse feature sizes could look like this:
        {TEXT: {FEATURE_TYPE_SEQUENCE: [4, 24, 128], FEATURE_TYPE_SENTENCE: [4, 128]}}

        Args:
            data_example: a data example that is stored with the ML component.
            new_sparse_feature_sizes: sizes of current sparse features.
            old_sparse_feature_sizes: sizes of sparse features the model was
                                      previously trained on.
        """
        self._check_if_sparse_feature_sizes_decreased(
            new_sparse_feature_sizes=new_sparse_feature_sizes,
            old_sparse_feature_sizes=old_sparse_feature_sizes,
        )
        if self._sparse_feature_sizes_have_increased(
            new_sparse_feature_sizes=new_sparse_feature_sizes,
            old_sparse_feature_sizes=old_sparse_feature_sizes,
        ):
            self._update_dense_for_sparse_layers(
                new_sparse_feature_sizes, old_sparse_feature_sizes
            )
            self._compile_and_fit(data_example)

    @staticmethod
    def _check_if_sparse_feature_sizes_decreased(
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    ) -> None:
        """Checks if the sizes of sparse features have decreased during fine-tuning.

        Sparse feature sizes might decrease after changing the training data.
        This can happen for example with `LexicalSyntacticFeaturizer`.
        We don't support this behaviour and we raise an exception if this happens.

        Args:
            new_sparse_feature_sizes: sizes of current sparse features.
            old_sparse_feature_sizes: sizes of sparse features the model was
                                      previously trained on.

        Raises:
            RasaException: When any of the sparse feature sizes decrease
                           from the last time training was run.
        """
        for attribute, new_feature_sizes in new_sparse_feature_sizes.items():
            old_feature_sizes = old_sparse_feature_sizes[attribute]
            for feature_type, new_sizes in new_feature_sizes.items():
                old_sizes = old_feature_sizes[feature_type]
                for new_size, old_size in zip(new_sizes, old_sizes):
                    if new_size < old_size:
                        raise RasaException(
                            "Sparse feature sizes have decreased from the last time "
                            "training was run. The training data was changed in a way "
                            "that resulted in some features not being present in the "
                            "data anymore. This can happen if you had "
                            "`LexicalSyntacticFeaturizer` in your pipeline. "
                            "The pipeline cannot support incremental training "
                            "in this setting. We recommend you to retrain "
                            "the model from scratch."
                        )

    @staticmethod
    def _sparse_feature_sizes_have_increased(
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    ) -> bool:
        """Checks if the sizes of sparse features have increased during fine-tuning.

        If there's any sparse feature size that has increased after changing the
        training data, we need to look for the corresponding `DenseForSparse` layer
        and adjust it. On the other hand, if none of them have increased, we don't
        need to change anything. This function helps us with making the decision.

        Note that the function assumes that none of the sparse feature sizes
        have decreased. In other words, it should get valid arguments in order
        to function well.

        Args:
            new_sparse_feature_sizes: sizes of current sparse features.
            old_sparse_feature_sizes: sizes of sparse features the model was
                                      previously trained on.

        Returns:
            `True` if any of the sparse feature sizes has increased, `False` otherwise.
        """
        for attribute, new_feature_sizes in new_sparse_feature_sizes.items():
            old_feature_sizes = old_sparse_feature_sizes[attribute]
            for feature_type, new_sizes in new_feature_sizes.items():
                old_sizes = old_feature_sizes[feature_type]
                if sum(new_sizes) > sum(old_sizes):
                    return True
        return False

    def _update_dense_for_sparse_layers(
        self,
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    ) -> None:
        """Updates `DenseForSparse` layers.

        Updates sizes of `DenseForSparse` layers by comparing current sparse feature
        sizes to old ones. This must be done before fine-tuning starts to account
        for any change in the size of sparse features that might have happened
        because of addition of new data.

        Args:
            new_sparse_feature_sizes: sizes of current sparse features.
            old_sparse_feature_sizes: sizes of sparse features the model was
                                      previously trained on.
        """
        for name, layer in self._tf_layers.items():
            # `if` condition is necessary because only `RasaCustomLayer`
            # can adjust sparse layers for incremental training by default.
            if isinstance(layer, rasa_layers.RasaCustomLayer):
                layer.adjust_sparse_layers_for_incremental_training(
                    new_sparse_feature_sizes,
                    old_sparse_feature_sizes,
                    self.config[REGULARIZATION_CONSTANT],
                )

    def _compile_and_fit(
        self, data_example: Dict[Text, Dict[Text, List[FeatureArray]]]
    ) -> None:
        """Compiles modified model and fits a sample data on it.

        Args:
            data_example: a data example that is stored with the ML component.
        """
        self.compile(optimizer=tf.keras.optimizers.Adam(self.config[LEARNING_RATE]))
        label_key = LABEL_KEY if self.config[INTENT_CLASSIFICATION] else None
        label_sub_key = LABEL_SUB_KEY if self.config[INTENT_CLASSIFICATION] else None

        model_data = RasaModelData(
            label_key=label_key, label_sub_key=label_sub_key, data=data_example
        )
        self._update_data_signatures(model_data)
        data_generator = RasaBatchDataGenerator(model_data, batch_size=1)
        self.fit(data_generator, verbose=False)

    def _update_data_signatures(self, model_data: RasaModelData) -> None:
        self.data_signature = model_data.get_signature()
        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in self.data_signature.items()
            if TEXT in feature_name
        }

    def _check_data(self) -> None:
        raise NotImplementedError

    def _prepare_layers(self) -> None:
        raise NotImplementedError

    def _prepare_label_classification_layers(self, predictor_attribute: Text) -> None:
        """Prepares layers & loss for the final label prediction step."""
        self._prepare_embed_layers(predictor_attribute)
        self._prepare_embed_layers(LABEL)
        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

    def _prepare_embed_layers(self, name: Text, prefix: Text = "embed") -> None:
        self._tf_layers[f"{prefix}.{name}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            name,
        )

    def _prepare_ffnn_layer(
        self,
        name: Text,
        layer_sizes: List[int],
        drop_rate: float,
        prefix: Text = "ffnn",
    ) -> None:
        self._tf_layers[f"{prefix}.{name}"] = layers.Ffnn(
            layer_sizes,
            drop_rate,
            self.config[REGULARIZATION_CONSTANT],
            self.config[CONNECTION_DENSITY],
            layer_name_suffix=name,
        )

    def _prepare_dot_product_loss(
        self, name: Text, scale_loss: bool, prefix: Text = "loss",
    ) -> None:
        self._tf_layers[f"{prefix}.{name}"] = self.dot_product_loss_layer(
            self.config[NUM_NEG],
            loss_type=self.config[LOSS_TYPE],
            mu_pos=self.config[MAX_POS_SIM],
            mu_neg=self.config[MAX_NEG_SIM],
            use_max_sim_neg=self.config[USE_MAX_NEG_SIM],
            neg_lambda=self.config[NEGATIVE_MARGIN_SCALE],
            scale_loss=scale_loss,
            similarity_type=self.config[SIMILARITY_TYPE],
            constrain_similarities=self.config[CONSTRAIN_SIMILARITIES],
            model_confidence=self.config[MODEL_CONFIDENCE],
        )

    @property
    def dot_product_loss_layer(self) -> tf.keras.layers.Layer:
        """Returns the dot-product loss layer to use.

        Returns:
            The loss layer that is used by `_prepare_dot_product_loss`.
        """
        return layers.SingleLabelDotProductLoss

    def _prepare_entity_recognition_layers(self) -> None:
        for tag_spec in self._entity_tag_specs:
            name = tag_spec.tag_name
            num_tags = tag_spec.num_tags
            self._tf_layers[f"embed.{name}.logits"] = layers.Embed(
                num_tags, self.config[REGULARIZATION_CONSTANT], f"logits.{name}"
            )
            self._tf_layers[f"crf.{name}"] = layers.CRF(
                num_tags, self.config[REGULARIZATION_CONSTANT], self.config[SCALE_LOSS]
            )
            self._tf_layers[f"embed.{name}.tags"] = layers.Embed(
                self.config[EMBEDDING_DIMENSION],
                self.config[REGULARIZATION_CONSTANT],
                f"tags.{name}",
            )

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        return tf.gather_nd(x, indices)

    def _get_mask_for(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        key: Text,
        sub_key: Text,
    ) -> Optional[tf.Tensor]:
        if key not in tf_batch_data or sub_key not in tf_batch_data[key]:
            return None

        sequence_lengths = tf.cast(tf_batch_data[key][sub_key][0], dtype=tf.int32)
        return rasa_layers.compute_mask(sequence_lengths)

    def _get_sequence_feature_lengths(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], key: Text
    ) -> tf.Tensor:
        """Fetches the sequence lengths of real tokens per input example.

        The number of real tokens for an example is the same as the length of the
        sequence of the sequence-level (token-level) features for that input example.
        """
        if key in tf_batch_data and SEQUENCE_LENGTH in tf_batch_data[key]:
            return tf.cast(tf_batch_data[key][SEQUENCE_LENGTH][0], dtype=tf.int32)

        batch_dim = self._get_batch_dim(tf_batch_data[key])
        return tf.zeros([batch_dim], dtype=tf.int32)

    def _get_sentence_feature_lengths(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], key: Text,
    ) -> tf.Tensor:
        """Fetches the sequence lengths of sentence-level features per input example.

        This is needed because we treat sentence-level features as token-level features
        with 1 token per input example. Hence, the sequence lengths returned by this
        function are all 1s if sentence-level features are present, and 0s otherwise.
        """
        batch_dim = self._get_batch_dim(tf_batch_data[key])

        if key in tf_batch_data and SENTENCE in tf_batch_data[key]:
            return tf.ones([batch_dim], dtype=tf.int32)

        return tf.zeros([batch_dim], dtype=tf.int32)

    @staticmethod
    def _get_batch_dim(attribute_data: Dict[Text, List[tf.Tensor]]) -> int:
        # All the values in the attribute_data dict should be lists of tensors, each
        # tensor of the shape (batch_dim, ...). So we take the first non-empty list we
        # encounter and infer the batch size from its first tensor.
        for key, data in attribute_data.items():
            if data:
                return tf.shape(data[0])[0]
        return None

    def _calculate_entity_loss(
        self,
        inputs: tf.Tensor,
        tag_ids: tf.Tensor,
        mask: tf.Tensor,
        sequence_lengths: tf.Tensor,
        tag_name: Text,
        entity_tags: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        tag_ids = tf.cast(tag_ids[:, :, 0], tf.int32)

        if entity_tags is not None:
            _tags = self._tf_layers[f"embed.{tag_name}.tags"](entity_tags)
            inputs = tf.concat([inputs, _tags], axis=-1)

        logits = self._tf_layers[f"embed.{tag_name}.logits"](inputs)

        # should call first to build weights
        pred_ids, _ = self._tf_layers[f"crf.{tag_name}"](logits, sequence_lengths)
        loss = self._tf_layers[f"crf.{tag_name}"].loss(
            logits, tag_ids, sequence_lengths
        )
        f1 = self._tf_layers[f"crf.{tag_name}"].f1_score(tag_ids, pred_ids, mask)

        return loss, f1, logits

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        raise NotImplementedError

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        raise NotImplementedError
