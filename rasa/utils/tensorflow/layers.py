import logging
from typing import List, Optional, Text, Tuple, Callable, Union, Any, Dict
import tensorflow as tf
import tensorflow_addons as tfa
import rasa.utils.tensorflow.crf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K
from rasa.utils.tensorflow.constants import SOFTMAX, MARGIN, COSINE, INNER

logger = logging.getLogger(__name__)

# https://github.com/tensorflow/addons#gpu-and-cpu-custom-ops-1
tfa.options.TF_ADDONS_PY_OPS = True


class SparseDropout(tf.keras.layers.Dropout):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Arguments:
        rate: Float between 0 and 1; fraction of the input units to drop.
    """

    def call(
        self, inputs: tf.SparseTensor, training: Optional[Union[tf.Tensor, bool]] = None
    ) -> tf.SparseTensor:
        """Apply dropout to sparse inputs.

        Arguments:
            inputs: Input sparse tensor (of any rank).
            training: Python boolean indicating whether the layer should behave in
                training mode (adding dropout) or in inference mode (doing nothing).

        Returns:
            Output of dropout layer.

        Raises:
            A ValueError if inputs is not a sparse tensor
        """

        if not isinstance(inputs, tf.SparseTensor):
            raise ValueError("Input tensor should be sparse.")

        if training is None:
            training = K.learning_phase()

        def dropped_inputs() -> tf.SparseTensor:
            to_retain_prob = tf.random.uniform(
                tf.shape(inputs.values), 0, 1, inputs.values.dtype
            )
            to_retain = tf.greater_equal(to_retain_prob, self.rate)
            return tf.sparse.retain(inputs, to_retain)

        outputs = tf_utils.smart_cond(
            training, dropped_inputs, lambda: tf.identity(inputs)
        )
        # need to explicitly recreate sparse tensor, because otherwise the shape
        # information will be lost after `retain`
        # noinspection PyProtectedMember
        return tf.SparseTensor(outputs.indices, outputs.values, inputs._dense_shape)


class DenseForSparse(tf.keras.layers.Dense):
    """Dense layer for sparse input tensor.

    Just your regular densely-connected NN layer but for sparse tensors.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    Arguments:
        units: Positive integer, dimensionality of the output space.
            activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        reg_lambda: Float, regularization factor.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, reg_lambda: float = 0, **kwargs: Any) -> None:
        if reg_lambda > 0:
            regularizer = tf.keras.regularizers.l2(reg_lambda)
        else:
            regularizer = None

        super().__init__(kernel_regularizer=regularizer, **kwargs)

    def call(self, inputs: tf.SparseTensor) -> tf.Tensor:
        """Apply dense layer to sparse inputs.

        Arguments:
            inputs: Input sparse tensor (of any rank).

        Returns:
            Output of dense layer.

        Raises:
            A ValueError if inputs is not a sparse tensor
        """
        if not isinstance(inputs, tf.SparseTensor):
            raise ValueError("Input tensor should be sparse.")

        # outputs will be 2D
        outputs = tf.sparse.sparse_dense_matmul(
            tf.sparse.reshape(inputs, [-1, tf.shape(inputs)[-1]]), self.kernel
        )

        if len(inputs.shape) == 3:
            # reshape back
            outputs = tf.reshape(
                outputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], self.units)
            )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class DenseWithSparseWeights(tf.keras.layers.Dense):
    """Just your regular densely-connected NN layer but with sparse weights.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    It creates `kernel_mask` to set fraction of the `kernel` weights to zero.

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    Arguments:
        sparsity: Float between 0 and 1. Fraction of the `kernel`
            weights to set to zero.
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, sparsity: float = 0.8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sparsity = sparsity

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        # create random mask to set fraction of the `kernel` weights to zero
        kernel_mask = tf.random.uniform(tf.shape(self.kernel), 0, 1)
        kernel_mask = tf.cast(
            tf.greater_equal(kernel_mask, self.sparsity), self.kernel.dtype
        )
        self.kernel_mask = tf.Variable(
            initial_value=kernel_mask, trainable=False, name="kernel_mask"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # set fraction of the `kernel` weights to zero according to precomputed mask
        self.kernel.assign(self.kernel * self.kernel_mask)
        return super().call(inputs)


from rasa.utils.tensorflow.model_data import FeatureSignature

from rasa.utils.tensorflow.constants import (
    REGULARIZATION_CONSTANT,
    WEIGHT_SPARSITY,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    NUM_HEADS,
    UNIDIRECTIONAL_ENCODER,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    HIDDEN_LAYERS_SIZES,
    DROP_RATE,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    DENSE_DIMENSION,
    CONCAT_DIMENSION,
    DROP_RATE_ATTENTION,
)
from rasa.shared.nlu.constants import TEXT


class ConcatenateSparseDenseFeatures(tf.keras.layers.Layer):
    # TODO add docstring
    def __init__(
        self,
        attribute: Text,
        feature_type: Text,
        data_signature: List[FeatureSignature],
        dropout_rate: float,
        sparse_dropout: bool,
        dense_dropout: bool,
        dense_concat_dimension: int,
        sparse_to_dense_kw: Dict[Text, Any] = {},
    ) -> None:
        super().__init__(
            name=f"concatenate_sparse_dense_features_{attribute}_{feature_type}"
        )
        self.have_sparse_features = any(
            [signature.is_sparse for signature in data_signature]
        )
        self.have_dense_features = any(
            [not signature.is_sparse for signature in data_signature]
        )

        all_sparse_units = sum(
            [
                dense_concat_dimension
                for signature in data_signature
                if signature.is_sparse
            ]
        )
        all_dense_units = sum(
            [signature.units for signature in data_signature if not signature.is_sparse]
        )
        self.output_units = all_sparse_units + all_dense_units

        self.use_sparse_dropout = sparse_dropout
        self.use_dense_dropout = dense_dropout

        if self.have_sparse_features:
            if "name" not in sparse_to_dense_kw:
                sparse_to_dense_kw[
                    "name"
                ] = f"sparse_to_dense.{attribute}_{feature_type}"
            self._sparse_to_dense = DenseForSparse(**sparse_to_dense_kw)

            if self.use_sparse_dropout:
                self._sparse_dropout = SparseDropout(rate=dropout_rate)

        if self.use_dense_dropout:
            self._dense_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(
        self,
        features: List[Union[tf.Tensor, tf.SparseTensor]],
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Optional[tf.Tensor]:
        dense_features = []
        for f in features:
            if isinstance(f, tf.SparseTensor):
                if self.use_sparse_dropout:
                    _f = self._sparse_dropout(f, training)
                else:
                    _f = f

                dense_f = self._sparse_to_dense(_f)

                if self.use_dense_dropout:
                    dense_f = self._dense_dropout(dense_f, training)

                dense_features.append(dense_f)
            else:
                dense_features.append(f)

        return tf.concat(dense_features, axis=-1)


from rasa.utils.tensorflow.constants import SEQUENCE, SENTENCE


class ConcatenateSequenceSentenceFeatures(tf.keras.layers.Layer):
    # TODO add docstring
    def __init__(
        self,
        layer_name_suffix: Text,
        concat_dimension: int,
        sequence_signature: FeatureSignature,
        sentence_signature: FeatureSignature,
        concat_layers_kwargs: Dict[Text, Any] = {},
    ) -> None:
        super().__init__(
            name=f"concatenate_sequence_sentence_features_{layer_name_suffix}"
        )
        if sequence_signature and sentence_signature:
            self.do_concatenation = True
            if sequence_signature.units != sentence_signature.units:
                self.unify_dimensions_before_concat = True
                self.output_units = concat_dimension
                self.unify_dimensions_layers = {}
                for feature_type in [SEQUENCE, SENTENCE]:
                    if "layer_name_suffix" not in concat_layers_kwargs:
                        concat_layers_kwargs[
                            "layer_name_suffix"
                        ] = f"unify_dimensions_before_concat.{layer_name_suffix}_{feature_type}"
                    self.unify_dimensions_layers[feature_type] = Ffnn(
                        **concat_layers_kwargs
                    )
            else:
                self.unify_dimensions_before_concat = False
                self.output_units = sequence_signature.units
        else:
            self.do_concatenation = False
            if sequence_signature and not sentence_signature:
                self.return_just = SEQUENCE
                self.output_units = sequence_signature.units
            elif sentence_signature and not sequence_signature:
                self.return_just = SENTENCE
                self.output_units = sentence_signature.units

    def call(
        self, sequence_x: tf.Tensor, sentence_x: tf.Tensor, mask_text: tf.Tensor,
    ) -> tf.Tensor:
        if self.do_concatenation:
            if self.unify_dimensions_before_concat:
                sequence_x = self.unify_dimensions_layers[SEQUENCE](sequence_x)
                sentence_x = self.unify_dimensions_layers[SENTENCE](sentence_x)

            # we need to concatenate the sequence features with the sentence features
            # we cannot use tf.concat as the sequence features are padded

            # (1) get position of sentence features in mask
            last = mask_text * tf.math.cumprod(
                1 - mask_text, axis=1, exclusive=True, reverse=True
            )
            # (2) multiply by sentence features so that we get a matrix of
            #     batch-dim x seq-dim x feature-dim with zeros everywhere except for
            #     for the sentence features
            sentence_x = last * sentence_x

            # (3) add a zero to the end of sequence matrix to match the final shape
            sequence_x = tf.pad(sequence_x, [[0, 0], [0, 1], [0, 0]])

            # (4) sum up sequence features and sentence features
            return sequence_x + sentence_x
        else:
            if self.return_just == SEQUENCE:
                return sequence_x
            elif self.return_just == SENTENCE:
                return sentence_x


# does:
# 1. sparse+dense
# 2. seq+sent
class RasaInputLayer(tf.keras.layers.Layer):
    # TODO add docstring
    def __init__(
        self,
        name: Text,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        super().__init__(name=f"rasa_input_layer_{name}")
        # SPARSE + DENSE
        self.concat_sparse_dense = {}
        for feature_type in [SENTENCE, SEQUENCE]:
            if feature_type in data_signature:
                sparse_to_dense_layer_options = {
                    "units": config[DENSE_DIMENSION][name],
                    "reg_lambda": config[REGULARIZATION_CONSTANT],
                    "name": f"sparse_to_dense.{name}_{feature_type}",
                }
                self.concat_sparse_dense[feature_type] = ConcatenateSparseDenseFeatures(
                    attribute=name,
                    feature_type=feature_type,
                    data_signature=data_signature.get(feature_type, []),
                    dropout_rate=config[DROP_RATE],
                    sparse_dropout=config[SPARSE_INPUT_DROPOUT],
                    dense_dropout=config[DENSE_INPUT_DROPOUT],
                    dense_concat_dimension=config[DENSE_DIMENSION][name],
                    sparse_to_dense_kw=sparse_to_dense_layer_options,
                )
            else:
                self.concat_sparse_dense[feature_type] = lambda features, training: None

        # SEQUENCE + SENTENCE
        self.do_seq_sent_concat = all(
            [feature_type in data_signature for feature_type in [SEQUENCE, SENTENCE]]
        )
        seq_sent_data_signatures = {}
        for feature_type in [SEQUENCE, SENTENCE]:
            if feature_type in data_signature:
                signature_existing = data_signature[feature_type][0]
                signature_new = FeatureSignature(
                    is_sparse=False,
                    units=self.concat_sparse_dense[feature_type].output_units,
                    number_of_dimensions=signature_existing.number_of_dimensions,
                )
                seq_sent_data_signatures[feature_type] = signature_new
            else:
                seq_sent_data_signatures[feature_type] = None

        if self.do_seq_sent_concat:
            concat_layers_kwargs = {
                "layer_sizes": [config[CONCAT_DIMENSION][name]],
                "dropout_rate": config[DROP_RATE],
                "reg_lambda": config[REGULARIZATION_CONSTANT],
                "sparsity": config[WEIGHT_SPARSITY],
            }
        else:
            concat_layers_kwargs = {}

        self.concat_seq_sent = ConcatenateSequenceSentenceFeatures(
            sequence_signature=seq_sent_data_signatures[SEQUENCE],
            sentence_signature=seq_sent_data_signatures[SENTENCE],
            concat_dimension=config[CONCAT_DIMENSION].get(name, None),
            concat_layers_kwargs=concat_layers_kwargs,
            layer_name_suffix=name,
        )

        self.output_units = self.concat_seq_sent.output_units

    def call(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask_sequence: tf.Tensor = None,
        mask_text: tf.Tensor = None,
        training: bool = True,
    ) -> tf.Tensor:
        sequence_x = self.concat_sparse_dense[SEQUENCE](sequence_features, training)
        if sequence_x is not None and mask_sequence is not None:
            sequence_x = sequence_x * mask_sequence
        sentence_x = self.concat_sparse_dense[SENTENCE](sentence_features, training)

        return self.concat_seq_sent(sequence_x, sentence_x, mask_text)


from rasa.utils.tensorflow.transformer import TransformerEncoder
from rasa.core.constants import DIALOGUE


# does:
# 1. input_layer
# 2. ffnn
# [3. MLM: masking & creating dense labels to sample from]
# 4. transformer
class RasaSequenceLayer(tf.keras.layers.Layer):
    # TODO add docstring
    def __init__(
        self,
        name: Text,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        super().__init__(name=f"rasa_input_layer_{name}")
        self.config = config

        # RASA INPUT LAYER
        self.input_layer = RasaInputLayer(name, data_signature, config)

        # FFNN
        self.ffnn = Ffnn(
            config[HIDDEN_LAYERS_SIZES][name],
            config[DROP_RATE],
            config[REGULARIZATION_CONSTANT],
            config[WEIGHT_SPARSITY],
            layer_name_suffix=name,
        )

        # MLM
        # for sequential text features prepare the logic for producing dense token embeddings
        # to be used as labels in MLM. these will be sampled from for negative sampling.
        if name == TEXT and SEQUENCE in data_signature:
            self.input_mask_layer = InputMask()

            self.produce_dense_token_ids = True
            has_sparse = any(
                [signature.is_sparse for signature in data_signature[SEQUENCE]]
            )
            has_dense = any(
                [not signature.is_sparse for signature in data_signature[SEQUENCE]]
            )
            # if dense features are present, we use those as unique token-level embeddings,
            # otherwise we create these from the sparse features by using a simple layer.
            if has_sparse and not has_dense:
                self.sparse_to_dense_token_ids = DenseForSparse(
                    units=2,
                    use_bias=False,
                    trainable=False,
                    name=f"sparse_to_dense_token_ids.{name}",
                )
        else:
            self.produce_dense_token_ids = False

        # TRANSFORMER
        num_layers = config[NUM_TRANSFORMER_LAYERS]
        if isinstance(num_layers, dict):
            num_layers = num_layers[name]
        size = config[TRANSFORMER_SIZE]
        if isinstance(size, dict):
            size = size[name]

        if num_layers > 0:
            self.transformer = TransformerEncoder(
                num_layers=num_layers,
                units=size,
                num_heads=config[NUM_HEADS],
                filter_units=size * 4,
                reg_lambda=config[REGULARIZATION_CONSTANT],
                dropout_rate=config[DROP_RATE],
                attention_dropout_rate=config[DROP_RATE_ATTENTION],
                sparsity=config[WEIGHT_SPARSITY],
                unidirectional=config[UNIDIRECTIONAL_ENCODER],
                use_key_relative_position=config[KEY_RELATIVE_ATTENTION],
                use_value_relative_position=config[VALUE_RELATIVE_ATTENTION],
                max_relative_position=config[MAX_RELATIVE_POSITION],
                name=f"{name}_encoder",
            )
        else:
            self.transformer = lambda x, mask, training: x

        # TODO: should this simply use NUM_TRANSFORMER_LAYERS?
        # if config[f"{DIALOGUE}_{NUM_TRANSFORMER_LAYERS}"] > 0:
        if num_layers > 0:
            self.output_units = size
        elif config[HIDDEN_LAYERS_SIZES][TEXT]:
            self.output_units = config[HIDDEN_LAYERS_SIZES][TEXT][-1]
        else:
            self.output_units = self.input_layer.concat_seq_sent.output_units

    def _features_as_seq_ids(
        self, features: List[Union[tf.Tensor, tf.SparseTensor]]
    ) -> Optional[tf.Tensor]:
        """Creates dense labels for negative sampling."""

        # if there are dense features - we can use them
        for f in features:
            if not isinstance(f, tf.SparseTensor):
                seq_ids = tf.stop_gradient(f)
                # add a zero to the seq dimension for the sentence features
                seq_ids = tf.pad(seq_ids, [[0, 0], [0, 1], [0, 0]])
                return seq_ids

        # use additional sparse to dense layer
        for f in features:
            if isinstance(f, tf.SparseTensor):
                seq_ids = tf.stop_gradient(self.sparse_to_dense_token_ids(f))
                # add a zero to the seq dimension for the sentence features
                seq_ids = tf.pad(seq_ids, [[0, 0], [0, 1], [0, 0]])
                return seq_ids

        return None

    def call(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask_sequence: tf.Tensor,
        mask: tf.Tensor,
        name: Text,
        training: bool,
        masked_lm_loss: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:

        inputs = self.input_layer(
            sequence_features, sentence_features, mask_sequence, mask
        )

        inputs = self.ffnn(inputs, training)

        if self.produce_dense_token_ids:
            seq_ids = self._features_as_seq_ids(sequence_features)
        else:
            seq_ids = None

        # TODO unify this with self.produce_dense_token_ids?
        if masked_lm_loss:
            transformer_inputs, lm_mask_bool = self.input_mask_layer(
                inputs, mask, training
            )
        else:
            transformer_inputs = inputs
            lm_mask_bool = None

        outputs = self.transformer(transformer_inputs, 1 - mask, training)

        num_layers = self.config[NUM_TRANSFORMER_LAYERS]
        if isinstance(num_layers, dict):
            num_layers = num_layers[name]
        if num_layers > 0:
            # apply activation
            outputs = tfa.activations.gelu(outputs)

        return outputs, inputs, seq_ids, lm_mask_bool


class Ffnn(tf.keras.layers.Layer):
    """Feed-forward network layer.

    Arguments:
        layer_sizes: List of integers with dimensionality of the layers.
        dropout_rate: Float between 0 and 1; fraction of the input units to drop.
        reg_lambda: Float, regularization factor.
        sparsity: Float between 0 and 1. Fraction of the `kernel`
            weights to set to zero.
        layer_name_suffix: Text added to the name of the layers.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., layer_sizes[-1])`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, layer_sizes[-1])`.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        dropout_rate: float,
        reg_lambda: float,
        sparsity: float,
        layer_name_suffix: Text,
    ) -> None:
        super().__init__(name=f"ffnn_{layer_name_suffix}")

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._ffn_layers = []
        for i, layer_size in enumerate(layer_sizes):
            self._ffn_layers.append(
                DenseWithSparseWeights(
                    units=layer_size,
                    sparsity=sparsity,
                    activation=tfa.activations.gelu,
                    kernel_regularizer=l2_regularizer,
                    name=f"hidden_layer_{layer_name_suffix}_{i}",
                )
            )
            self._ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))

    def call(
        self, x: tf.Tensor, training: Optional[Union[tf.Tensor, bool]] = None
    ) -> tf.Tensor:
        for layer in self._ffn_layers:
            x = layer(x, training=training)

        return x


class Embed(tf.keras.layers.Layer):
    """Dense embedding layer.

    Arguments:
        embed_dim: Positive integer, dimensionality of the output space.
        reg_lambda: Float; regularization factor.
        layer_name_suffix: Text added to the name of the layers.
        similarity_type: Optional type of similarity measure to use,
            either 'cosine' or 'inner'.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., embed_dim)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, embed_dim)`.
    """

    def __init__(
        self,
        embed_dim: int,
        reg_lambda: float,
        layer_name_suffix: Text,
        similarity_type: Optional[Text] = None,
    ) -> None:
        super().__init__(name=f"embed_{layer_name_suffix}")

        self.similarity_type = similarity_type
        if self.similarity_type and self.similarity_type not in {COSINE, INNER}:
            raise ValueError(
                f"Wrong similarity type '{self.similarity_type}', "
                f"should be '{COSINE}' or '{INNER}'."
            )

        regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._dense = tf.keras.layers.Dense(
            units=embed_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name=f"embed_layer_{layer_name_suffix}",
        )

    # noinspection PyMethodOverriding
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self._dense(x)
        if self.similarity_type == COSINE:
            x = tf.nn.l2_normalize(x, axis=-1)

        return x


class InputMask(tf.keras.layers.Layer):
    """The layer that masks 15% of the input.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, input_dim)`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._masking_prob = 0.85
        self._mask_vector_prob = 0.7
        self._random_vector_prob = 0.1

    def build(self, input_shape: tf.TensorShape) -> None:
        self.mask_vector = self.add_weight(
            shape=(1, 1, input_shape[-1]), name="mask_vector"
        )
        self.built = True

    # noinspection PyMethodOverriding
    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor,
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly mask input sequences.

        Arguments:
            x: Input sequence tensor of rank 3.
            mask: A tensor representing sequence mask,
                contains `1` for inputs and `0` for padding.
            training: Python boolean indicating whether the layer should behave in
                training mode (mask inputs) or in inference mode (doing nothing).

        Returns:
            A tuple of masked inputs and boolean mask.
        """

        if training is None:
            training = K.learning_phase()

        lm_mask_prob = tf.random.uniform(tf.shape(mask), 0, 1, mask.dtype) * mask
        lm_mask_bool = tf.greater_equal(lm_mask_prob, self._masking_prob)

        def x_masked() -> tf.Tensor:
            x_random_pad = tf.random.uniform(
                tf.shape(x), tf.reduce_min(x), tf.reduce_max(x), x.dtype
            ) * (1 - mask)
            # shuffle over batch dim
            x_shuffle = tf.random.shuffle(x * mask + x_random_pad)

            # shuffle over sequence dim
            x_shuffle = tf.transpose(x_shuffle, [1, 0, 2])
            x_shuffle = tf.random.shuffle(x_shuffle)
            x_shuffle = tf.transpose(x_shuffle, [1, 0, 2])

            # shuffle doesn't support backprop
            x_shuffle = tf.stop_gradient(x_shuffle)

            mask_vector = tf.tile(self.mask_vector, (tf.shape(x)[0], tf.shape(x)[1], 1))

            other_prob = tf.random.uniform(tf.shape(mask), 0, 1, mask.dtype)
            other_prob = tf.tile(other_prob, (1, 1, x.shape[-1]))
            x_other = tf.where(
                other_prob < self._mask_vector_prob,
                mask_vector,
                tf.where(
                    other_prob < self._mask_vector_prob + self._random_vector_prob,
                    x_shuffle,
                    x,
                ),
            )

            return tf.where(tf.tile(lm_mask_bool, (1, 1, x.shape[-1])), x_other, x)

        return (
            tf_utils.smart_cond(training, x_masked, lambda: tf.identity(x)),
            lm_mask_bool,
        )


def _scale_loss(log_likelihood: tf.Tensor) -> tf.Tensor:
    """Creates scaling loss coefficient depending on the prediction probability.

    Arguments:
        log_likelihood: a tensor, log-likelihood of prediction

    Returns:
        Scaling tensor.
    """

    p = tf.math.exp(log_likelihood)
    # only scale loss if some examples are already learned
    return tf.cond(
        tf.reduce_max(p) > 0.5,
        lambda: tf.stop_gradient(tf.pow((1 - p) / 0.5, 4)),
        lambda: tf.ones_like(p),
    )


class CRF(tf.keras.layers.Layer):
    """CRF layer.

    Arguments:
        num_tags: Positive integer, number of tags.
        reg_lambda: Float; regularization factor.
        name: Optional name of the layer.
    """

    def __init__(
        self,
        num_tags: int,
        reg_lambda: float,
        scale_loss: bool,
        name: Optional[Text] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_tags = num_tags
        self.scale_loss = scale_loss
        self.transition_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self.f1_score_metric = tfa.metrics.F1Score(
            num_classes=num_tags - 1,  # `0` prediction is not a prediction
            average="micro",
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        # the weights should be created in `build` to apply random_seed
        self.transition_params = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            regularizer=self.transition_regularizer,
            name="transitions",
        )
        self.built = True

    # noinspection PyMethodOverriding
    def call(
        self, logits: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Decodes the highest scoring sequence of tags.

        Arguments:
            logits: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
            sequence_lengths: A [batch_size] vector of true sequence lengths.

        Returns:
            A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
            Contains the highest scoring tag indices.
            A [batch_size, max_seq_len] matrix, with dtype `tf.float32`.
            Contains the confidence values of the highest scoring tag indices.
        """
        predicted_ids, scores, _ = rasa.utils.tensorflow.crf.crf_decode(
            logits, self.transition_params, sequence_lengths
        )
        # set prediction index for padding to `0`
        mask = tf.sequence_mask(
            sequence_lengths,
            maxlen=tf.shape(predicted_ids)[1],
            dtype=predicted_ids.dtype,
        )

        confidence_values = scores * tf.cast(mask, tf.float32)
        predicted_ids = predicted_ids * mask

        return predicted_ids, confidence_values

    def loss(
        self, logits: tf.Tensor, tag_indices: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> tf.Tensor:
        """Computes the log-likelihood of tag sequences in a CRF.

        Arguments:
            logits: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
                to use as input to the CRF layer.
            tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
                we compute the log-likelihood.
            sequence_lengths: A [batch_size] vector of true sequence lengths.

        Returns:
            Negative mean log-likelihood of all examples,
            given the sequence of tag indices.
        """

        log_likelihood, _ = tfa.text.crf.crf_log_likelihood(
            logits, tag_indices, sequence_lengths, self.transition_params
        )
        loss = -log_likelihood
        if self.scale_loss:
            loss *= _scale_loss(log_likelihood)

        return tf.reduce_mean(loss)

    def f1_score(
        self, tag_ids: tf.Tensor, pred_ids: tf.Tensor, mask: tf.Tensor
    ) -> tf.Tensor:
        """Calculates f1 score for train predictions"""

        mask_bool = tf.cast(mask[:, :, 0], tf.bool)

        # pick only non padding values and flatten sequences
        tag_ids_flat = tf.boolean_mask(tag_ids, mask_bool)
        pred_ids_flat = tf.boolean_mask(pred_ids, mask_bool)

        # set `0` prediction to not a prediction
        num_tags = self.num_tags - 1

        tag_ids_flat_one_hot = tf.one_hot(tag_ids_flat - 1, num_tags)
        pred_ids_flat_one_hot = tf.one_hot(pred_ids_flat - 1, num_tags)

        return self.f1_score_metric(tag_ids_flat_one_hot, pred_ids_flat_one_hot)


class DotProductLoss(tf.keras.layers.Layer):
    """Dot-product loss layer.

    Arguments:
        num_neg: Positive integer, the number of incorrect labels;
            the algorithm will minimize their similarity to the input.
        loss_type: The type of the loss function, either 'softmax' or 'margin'.
        mu_pos: Float, indicates how similar the algorithm should
            try to make embedding vectors for correct labels;
            should be 0.0 < ... < 1.0 for 'cosine' similarity type.
        mu_neg: Float, maximum negative similarity for incorrect labels,
            should be -1.0 < ... < 1.0 for 'cosine' similarity type.
        use_max_sim_neg: Boolean, if 'True' the algorithm only minimizes
            maximum similarity over incorrect intent labels,
            used only if 'loss_type' is set to 'margin'.
        neg_lambda: Float, the scale of how important is to minimize
            the maximum similarity between embeddings of different labels,
            used only if 'loss_type' is set to 'margin'.
        scale_loss: Boolean, if 'True' scale loss inverse proportionally to
            the confidence of the correct prediction.
        name: Optional name of the layer.
        parallel_iterations: Positive integer, the number of iterations allowed
            to run in parallel.
        same_sampling: Boolean, if 'True' sample same negative labels
            for the whole batch.
    """

    def __init__(
        self,
        num_neg: int,
        loss_type: Text,
        mu_pos: float,
        mu_neg: float,
        use_max_sim_neg: bool,
        neg_lambda: float,
        scale_loss: bool,
        name: Optional[Text] = None,
        parallel_iterations: int = 1000,
        same_sampling: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.num_neg = num_neg
        self.loss_type = loss_type
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        self.use_max_sim_neg = use_max_sim_neg
        self.neg_lambda = neg_lambda
        self.scale_loss = scale_loss
        self.parallel_iterations = parallel_iterations
        self.same_sampling = same_sampling

    @staticmethod
    def _make_flat(x: tf.Tensor) -> tf.Tensor:
        """Make tensor 2D."""

        return tf.reshape(x, (-1, x.shape[-1]))

    def _random_indices(
        self, batch_size: tf.Tensor, total_candidates: tf.Tensor
    ) -> tf.Tensor:
        def rand_idxs() -> tf.Tensor:
            """Create random tensor of indices"""

            # (1, num_neg)
            return tf.expand_dims(
                tf.random.shuffle(tf.range(total_candidates))[: self.num_neg], 0
            )

        if self.same_sampling:
            return tf.tile(rand_idxs(), (batch_size, 1))

        def cond(idx: tf.Tensor, out: tf.Tensor) -> tf.Tensor:
            """Condition for while loop"""
            return idx < batch_size

        def body(idx: tf.Tensor, out: tf.Tensor) -> List[tf.Tensor]:
            """Body of the while loop"""
            return [
                # increment counter
                idx + 1,
                # add random indices
                tf.concat([out, rand_idxs()], 0),
            ]

        # first tensor already created
        idx1 = tf.constant(1)
        # create first random array of indices
        out1 = rand_idxs()  # (1, num_neg)

        return tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(
                cond,
                body,
                loop_vars=[idx1, out1],
                shape_invariants=[idx1.shape, tf.TensorShape([None, self.num_neg])],
                parallel_iterations=self.parallel_iterations,
            ),
        )[1]

    @staticmethod
    def _sample_idxs(batch_size: tf.Tensor, x: tf.Tensor, idxs: tf.Tensor) -> tf.Tensor:
        """Sample negative examples for given indices"""

        tiled = tf.tile(tf.expand_dims(x, 0), (batch_size, 1, 1))

        return tf.gather(tiled, idxs, batch_dims=1)

    def _get_bad_mask(
        self, labels: tf.Tensor, target_labels: tf.Tensor, idxs: tf.Tensor
    ) -> tf.Tensor:
        """Calculate bad mask for given indices.

        Checks that input features are different for positive negative samples.
        """

        pos_labels = tf.expand_dims(target_labels, axis=-2)
        neg_labels = self._sample_idxs(tf.shape(target_labels)[0], labels, idxs)

        return tf.cast(
            tf.reduce_all(tf.equal(neg_labels, pos_labels), axis=-1), pos_labels.dtype
        )

    def _get_negs(
        self, embeds: tf.Tensor, labels: tf.Tensor, target_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get negative examples from given tensor."""

        embeds_flat = self._make_flat(embeds)
        labels_flat = self._make_flat(labels)
        target_labels_flat = self._make_flat(target_labels)

        total_candidates = tf.shape(embeds_flat)[0]
        target_size = tf.shape(target_labels_flat)[0]

        neg_ids = self._random_indices(target_size, total_candidates)

        neg_embeds = self._sample_idxs(target_size, embeds_flat, neg_ids)
        bad_negs = self._get_bad_mask(labels_flat, target_labels_flat, neg_ids)

        # check if inputs have sequence dimension
        if len(target_labels.shape) == 3:
            # tensors were flattened for sampling, reshape back
            # add sequence dimension if it was present in the inputs
            target_shape = tf.shape(target_labels)
            neg_embeds = tf.reshape(
                neg_embeds, (target_shape[0], target_shape[1], -1, embeds.shape[-1])
            )
            bad_negs = tf.reshape(bad_negs, (target_shape[0], target_shape[1], -1))

        return neg_embeds, bad_negs

    def _sample_negatives(
        self,
        inputs_embed: tf.Tensor,
        labels_embed: tf.Tensor,
        labels: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_labels: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Sample negative examples."""

        pos_inputs_embed = tf.expand_dims(inputs_embed, axis=-2)
        pos_labels_embed = tf.expand_dims(labels_embed, axis=-2)

        # sample negative inputs
        neg_inputs_embed, inputs_bad_negs = self._get_negs(inputs_embed, labels, labels)
        # sample negative labels
        neg_labels_embed, labels_bad_negs = self._get_negs(
            all_labels_embed, all_labels, labels
        )
        return (
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
        )

    @staticmethod
    def sim(a: tf.Tensor, b: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Calculate similarity between given tensors."""

        sim = tf.reduce_sum(a * b, axis=-1)
        if mask is not None:
            sim *= tf.expand_dims(mask, 2)

        return sim

    @staticmethod
    def confidence_from_sim(sim: tf.Tensor, similarity_type: Text) -> tf.Tensor:
        if similarity_type == COSINE:
            # clip negative values to zero
            return tf.nn.relu(sim)
        else:
            # normalize result to [0, 1] with softmax
            return tf.nn.softmax(sim)

    def _train_sim(
        self,
        pos_inputs_embed: tf.Tensor,
        pos_labels_embed: tf.Tensor,
        neg_inputs_embed: tf.Tensor,
        neg_labels_embed: tf.Tensor,
        inputs_bad_negs: tf.Tensor,
        labels_bad_negs: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Define similarity."""

        # calculate similarity with several
        # embedded actions for the loss
        neg_inf = tf.constant(-1e9)

        sim_pos = self.sim(pos_inputs_embed, pos_labels_embed, mask)
        sim_neg_il = (
            self.sim(pos_inputs_embed, neg_labels_embed, mask)
            + neg_inf * labels_bad_negs
        )
        sim_neg_ll = (
            self.sim(pos_labels_embed, neg_labels_embed, mask)
            + neg_inf * labels_bad_negs
        )
        sim_neg_ii = (
            self.sim(pos_inputs_embed, neg_inputs_embed, mask)
            + neg_inf * inputs_bad_negs
        )
        sim_neg_li = (
            self.sim(pos_labels_embed, neg_inputs_embed, mask)
            + neg_inf * inputs_bad_negs
        )

        # output similarities between user input and bot actions
        # and similarities between bot actions and similarities between user inputs
        return sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li

    @staticmethod
    def _calc_accuracy(sim_pos: tf.Tensor, sim_neg: tf.Tensor) -> tf.Tensor:
        """Calculate accuracy."""

        max_all_sim = tf.reduce_max(tf.concat([sim_pos, sim_neg], axis=-1), axis=-1)
        return tf.reduce_mean(
            tf.cast(
                tf.math.equal(max_all_sim, tf.squeeze(sim_pos, axis=-1)), tf.float32
            )
        )

    def _loss_margin(
        self,
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Define max margin loss."""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0.0, self.mu_pos - tf.squeeze(sim_pos, axis=-1))

        # loss for minimizing similarity with `num_neg` incorrect actions
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg_il = tf.reduce_max(sim_neg_il, axis=-1)
            loss += tf.maximum(0.0, self.mu_neg + max_sim_neg_il)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0.0, self.mu_neg + sim_neg_il)
            loss += tf.reduce_sum(max_margin, axis=-1)

        # penalize max similarity between pos bot and neg bot embeddings
        max_sim_neg_ll = tf.maximum(
            0.0, self.mu_neg + tf.reduce_max(sim_neg_ll, axis=-1)
        )
        loss += max_sim_neg_ll * self.neg_lambda

        # penalize max similarity between pos dial and neg dial embeddings
        max_sim_neg_ii = tf.maximum(
            0.0, self.mu_neg + tf.reduce_max(sim_neg_ii, axis=-1)
        )
        loss += max_sim_neg_ii * self.neg_lambda

        # penalize max similarity between pos bot and neg dial embeddings
        max_sim_neg_li = tf.maximum(
            0.0, self.mu_neg + tf.reduce_max(sim_neg_li, axis=-1)
        )
        loss += max_sim_neg_li * self.neg_lambda

        if mask is not None:
            # mask loss for different length sequences
            loss *= mask
            # average the loss over sequence length
            loss = tf.reduce_sum(loss, axis=-1) / tf.reduce_sum(mask, axis=1)

        # average the loss over the batch
        loss = tf.reduce_mean(loss)

        return loss

    def _loss_softmax(
        self,
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Define softmax loss."""

        logits = tf.concat(
            [sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li], axis=-1
        )

        # create label_ids for softmax
        label_ids = tf.zeros_like(logits[..., 0], tf.int32)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_ids, logits=logits
        )

        if self.scale_loss:
            # in case of cross entropy log_likelihood = -loss
            loss *= _scale_loss(-loss)

        if mask is not None:
            loss *= mask

        if len(loss.shape) == 2:
            # average over the sequence
            if mask is not None:
                loss = tf.reduce_sum(loss, axis=-1) / tf.reduce_sum(mask, axis=-1)
            else:
                loss = tf.reduce_mean(loss, axis=-1)

        # average the loss over the batch
        return tf.reduce_mean(loss)

    @property
    def _chosen_loss(self) -> Callable:
        """Use loss depending on given option."""

        if self.loss_type == MARGIN:
            return self._loss_margin
        elif self.loss_type == SOFTMAX:
            return self._loss_softmax
        else:
            raise ValueError(
                f"Wrong loss type '{self.loss_type}', "
                f"should be '{MARGIN}' or '{SOFTMAX}'"
            )

    # noinspection PyMethodOverriding
    def call(
        self,
        inputs_embed: tf.Tensor,
        labels_embed: tf.Tensor,
        labels: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_labels: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate loss and accuracy.

        Arguments:
            inputs_embed: Embedding tensor for the batch inputs.
            labels_embed: Embedding tensor for the batch labels.
            labels: Tensor representing batch labels.
            all_labels_embed: Embedding tensor for the all labels.
            all_labels: Tensor representing all labels.
            mask: Optional tensor representing sequence mask,
                contains `1` for inputs and `0` for padding.

        Returns:
            loss: Total loss.
            accuracy: Training accuracy.
        """
        (
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
        ) = self._sample_negatives(
            inputs_embed, labels_embed, labels, all_labels_embed, all_labels
        )

        # calculate similarities
        sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li = self._train_sim(
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
            mask,
        )

        accuracy = self._calc_accuracy(sim_pos, sim_neg_il)

        loss = self._chosen_loss(
            sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li, mask
        )

        return loss, accuracy
