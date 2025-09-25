import logging
from typing import Any, Callable, List, Optional, Text, Tuple, Union

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils.control_flow_util import smart_cond

import rasa.utils.tensorflow.crf
import rasa.utils.tensorflow.layers_utils as layers_utils
from rasa.core.constants import DIALOGUE
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    ACTION_TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    INTENT,
    TEXT,
)
from rasa.utils.tensorflow.constants import (
    COSINE,
    CROSS_ENTROPY,
    INNER,
    LABEL,
    LABEL_PAD_ID,
    MARGIN,
    SOFTMAX,
)
from rasa.utils.tensorflow.crf import crf_log_likelihood
from rasa.utils.tensorflow.exceptions import TFLayerConfigException
from rasa.utils.tensorflow.metrics import F1Score

logger = logging.getLogger(__name__)


POSSIBLE_ATTRIBUTES = [
    TEXT,
    INTENT,
    LABEL,
    DIALOGUE,
    ACTION_NAME,
    ACTION_TEXT,
    f"{LABEL}_{ACTION_NAME}",
    f"{LABEL}_{ACTION_TEXT}",
]


class SparseDropout(tf.keras.layers.Dropout):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Arguments:
        rate: Fraction of the input units to drop (between 0 and 1).
    """

    def call(
        self, inputs: tf.SparseTensor, training: Optional[Union[tf.Tensor, bool]] = None
    ) -> tf.SparseTensor:
        """Apply dropout to sparse inputs.

        Arguments:
            inputs: Input sparse tensor (of any rank).
            training: Indicates whether the layer should behave in
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

        outputs = smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))
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
        use_bias: Indicates whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        reg_lambda: regularization factor
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

    def get_units(self) -> int:
        """Returns number of output units."""
        return self.units

    def get_kernel(self) -> tf.Tensor:
        """Returns kernel tensor."""
        return self.kernel

    def get_bias(self) -> Union[tf.Tensor, None]:
        """Returns bias tensor."""
        if self.use_bias:
            return self.bias
        return None

    def get_feature_type(self) -> Union[Text, None]:
        """Returns a feature type of the data that's fed to the layer.

        In order to correctly return a feature type, the function heavily relies
        on the name of `DenseForSparse` layer to contain the feature type.
        Acceptable values of feature types are `FEATURE_TYPE_SENTENCE`
        and `FEATURE_TYPE_SEQUENCE`.

        Returns:
            feature type of dense layer.
        """
        for feature_type in [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE]:
            if feature_type in self.name:
                return feature_type
        return None

    def get_attribute(self) -> Union[Text, None]:
        """Returns the attribute for which this layer was constructed.

        For example: TEXT, LABEL, etc.

        In order to correctly return an attribute, the function heavily relies
        on the name of `DenseForSparse` layer being in the following format:
        f"sparse_to_dense.{attribute}_{feature_type}".

        Returns:
            attribute of the layer.
        """
        metadata = self.name.split(".")
        if len(metadata) > 1:
            attribute_splits = metadata[1].split("_")[:-1]
            attribute = "_".join(attribute_splits)
            if attribute in POSSIBLE_ATTRIBUTES:
                return attribute
        return None

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


class RandomlyConnectedDense(tf.keras.layers.Dense):
    """Layer with dense ouputs that are connected to a random subset of inputs.

    `RandomlyConnectedDense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    It creates `kernel_mask` to set a fraction of the `kernel` weights to zero.

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    The output is guaranteed to be dense (each output is connected to at least one
    input), and no input is disconnected (each input is connected to at least one
    output).

    At `density = 0.0` the number of trainable weights is `max(input_size, units)`. At
    `density = 1.0` this layer is equivalent to `tf.keras.layers.Dense`.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, density: float = 0.2, **kwargs: Any) -> None:
        """Declares instance variables with default values.

        Args:
            density: Approximate fraction of trainable weights (between 0 and 1).
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Indicates whether the layer uses a bias vector.
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
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        if density < 0.0 or density > 1.0:
            raise TFLayerConfigException("Layer density must be in [0, 1].")

        self.density = density

    def build(self, input_shape: tf.TensorShape) -> None:
        """Prepares the kernel mask.

        Args:
            input_shape: Shape of the inputs to this layer
        """
        super().build(input_shape)

        if self.density == 1.0:
            self.kernel_mask = None
            return

        # Use callable initializer for TensorFlow 2.19.1 compatibility
        def kernel_mask_initializer() -> tf.Tensor:
            # Construct mask with given density and guarantee that every output is
            # connected to at least one input
            kernel_mask = self._minimal_mask() + self._random_mask()

            # We might accidently have added a random connection on top of
            # a fixed connection
            kernel_mask = tf.clip_by_value(kernel_mask, 0, 1)
            return kernel_mask

        self.kernel_mask = tf.Variable(
            initial_value=kernel_mask_initializer, trainable=False, name="kernel_mask"
        )

    def _random_mask(self) -> tf.Tensor:
        """Creates a random matrix with `num_ones` 1s and 0s otherwise.

        Returns:
            A random mask matrix
        """
        mask = tf.random.uniform(tf.shape(self.kernel), 0, 1)
        mask = tf.cast(tf.math.less(mask, self.density), self.kernel.dtype)
        return mask

    def _minimal_mask(self) -> tf.Tensor:
        """Creates a matrix with a minimal number of 1s to connect everythinig.

        If num_rows == num_cols, this creates the identity matrix.
        If num_rows > num_cols, this creates
            1 0 0 0
            0 1 0 0
            0 0 1 0
            0 0 0 1
            1 0 0 0
            0 1 0 0
            0 0 1 0
            . . . .
            . . . .
            . . . .
        If num_rows < num_cols, this creates
            1 0 0 1 0 0 1 ...
            0 1 0 0 1 0 0 ...
            0 0 1 0 0 1 0 ...

        Returns:
            A tiled and croped identity matrix.
        """
        kernel_shape = tf.shape(self.kernel)
        num_rows = kernel_shape[0]
        num_cols = kernel_shape[1]
        short_dimension = tf.minimum(num_rows, num_cols)

        mask = tf.tile(
            tf.eye(short_dimension, dtype=self.kernel.dtype),
            [
                tf.math.ceil(num_rows / short_dimension),
                tf.math.ceil(num_cols / short_dimension),
            ],
        )[:num_rows, :num_cols]

        return mask

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Processes the given inputs.

        Args:
            inputs: What goes into this layer

        Returns:
            The processed inputs.
        """
        # Apply kernel masking if needed (Keras 3.x compatibility check)
        if (
            self.density < 1.0
            and hasattr(self, "kernel_mask")
            and self.kernel_mask is not None
        ):
            # Set fraction of the `kernel` weights to zero according to precomputed mask
            self.kernel.assign(self.kernel * self.kernel_mask)
        return super().call(inputs)


class Ffnn(tf.keras.layers.Layer):
    """Feed-forward network layer.

    Arguments:
        layer_sizes: List of integers with dimensionality of the layers.
        dropout_rate: Fraction of the input units to drop (between 0 and 1).
        reg_lambda: regularization factor.
        density: Approximate fraction of trainable weights (between 0 and 1).
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
        density: float,
        layer_name_suffix: Text,
    ) -> None:
        super().__init__(name=f"ffnn_{layer_name_suffix}")

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._ffn_layers = []
        for i, layer_size in enumerate(layer_sizes):
            self._ffn_layers.append(
                RandomlyConnectedDense(
                    units=layer_size,
                    density=density,
                    activation=tf.nn.gelu,
                    kernel_regularizer=l2_regularizer,
                    name=f"hidden_layer_{layer_name_suffix}_{i}",
                )
            )
            self._ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))

    def call(
        self, x: tf.Tensor, training: Optional[Union[tf.Tensor, bool]] = None
    ) -> tf.Tensor:
        """Apply feed-forward network layer."""
        for layer in self._ffn_layers:
            x = layer(x, training=training)

        return x


class Embed(tf.keras.layers.Layer):
    """Dense embedding layer.

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
        self, embed_dim: int, reg_lambda: float, layer_name_suffix: Text
    ) -> None:
        """Initialize layer.

        Args:
            embed_dim: Dimensionality of the output space.
            reg_lambda: Regularization factor.
            layer_name_suffix: Text added to the name of the layers.
        """
        super().__init__(name=f"embed_{layer_name_suffix}")

        regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._dense = tf.keras.layers.Dense(
            units=embed_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name=f"embed_layer_{layer_name_suffix}",
        )

    # noinspection PyMethodOverriding
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply dense layer."""
        x = self._dense(x)
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
            training: Indicates whether the layer should run in
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

        return (smart_cond(training, x_masked, lambda: tf.identity(x)), lm_mask_bool)


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
        reg_lambda: regularization factor.
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
        self.f1_score_metric = F1Score(
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
        log_likelihood, _ = crf_log_likelihood(
            logits, tag_indices, sequence_lengths, self.transition_params
        )
        loss = -log_likelihood
        if self.scale_loss:
            loss *= _scale_loss(log_likelihood)

        return tf.reduce_mean(loss)

    def f1_score(
        self, tag_ids: tf.Tensor, pred_ids: tf.Tensor, mask: tf.Tensor
    ) -> tf.Tensor:
        """Calculates f1 score for train predictions."""
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
    """Abstract dot-product loss layer class.

    Idea based on StarSpace paper: http://arxiv.org/abs/1709.03856

    Implements similarity methods
    * `sim` (computes a similarity between vectors)
    * `get_similarities_and_confidences_from_embeddings` (calls `sim` and also computes
        confidence values)

    Specific loss functions (single- or multi-label) must be implemented in child
    classes.
    """

    def __init__(
        self,
        num_candidates: int,
        scale_loss: bool = False,
        constrain_similarities: bool = True,
        model_confidence: Text = SOFTMAX,
        similarity_type: Text = INNER,
        name: Optional[Text] = None,
        **kwargs: Any,
    ):
        """Declares instance variables with default values.

        Args:
            num_candidates: Number of labels besides the positive one. Depending on
                whether single- or multi-label loss is implemented (done in
                sub-classes), these can be all negative example labels, or a mixture of
                negative and further positive labels, respectively.
            scale_loss: Boolean, if `True` scale loss inverse proportionally to
                the confidence of the correct prediction.
            constrain_similarities: Boolean, if `True` applies sigmoid on all
                similarity terms and adds to the loss function to
                ensure that similarity values are approximately bounded.
                Used inside _loss_cross_entropy() only.
            model_confidence: Normalization of confidence values during inference.
                Currently, the only possible value is `SOFTMAX`.
            similarity_type: Similarity measure to use, either `cosine` or `inner`.
            name: Optional name of the layer.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            TFLayerConfigException: When `similarity_type` is not one of `COSINE` or
                `INNER`.
        """
        super().__init__(name=name)
        self.num_neg = num_candidates
        self.scale_loss = scale_loss
        self.constrain_similarities = constrain_similarities
        self.model_confidence = model_confidence
        self.similarity_type = similarity_type
        if self.similarity_type not in {COSINE, INNER}:
            raise TFLayerConfigException(
                f"Unsupported similarity type '{self.similarity_type}', "
                f"should be '{COSINE}' or '{INNER}'."
            )

    def sim(
        self, a: tf.Tensor, b: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Calculates similarity between `a` and `b`.

        Operates on the last dimension. When `a` and `b` are vectors, then `sim`
        computes either the dot-product, or the cosine of the angle between `a` and `b`,
        depending on `self.similarity_type`.
        Specifically, when the similarity type is `INNER`, then we compute the scalar
        product `a . b`. When the similarity type is `COSINE`, we compute
        `a . b / (|a| |b|)`, i.e. the cosine of the angle between `a` and `b`.

        Args:
            a: Any float tensor
            b: Any tensor of the same shape and type as `a`
            mask: Mask (should contain 1s for inputs and 0s for padding). Note, that
                `len(mask.shape) == len(a.shape) - 1` should hold.

        Returns:
            Similarities between vectors in `a` and `b`.
        """
        if self.similarity_type == COSINE:
            a = tf.nn.l2_normalize(a, axis=-1)
            b = tf.nn.l2_normalize(b, axis=-1)
        sim = tf.reduce_sum(a * b, axis=-1)
        if mask is not None:
            sim *= tf.expand_dims(mask, 2)

        return sim

    def get_similarities_and_confidences_from_embeddings(
        self,
        input_embeddings: tf.Tensor,
        label_embeddings: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes similary between input and label embeddings and model's confidence.

        First compute the similarity from embeddings and then apply an activation
        function if needed to get the confidence.

        Args:
            input_embeddings: Embeddings of input.
            label_embeddings: Embeddings of labels.
            mask: Mask (should contain 1s for inputs and 0s for padding). Note, that
                `len(mask.shape) == len(a.shape) - 1` should hold.

        Returns:
            similarity between input and label embeddings and model's prediction
            confidence for each label.
        """
        similarities = self.sim(input_embeddings, label_embeddings, mask)
        confidences = similarities
        if self.model_confidence == SOFTMAX:
            confidences = tf.nn.softmax(similarities)
        return similarities, confidences

    def call(self, *args: Any, **kwargs: Any) -> Tuple[tf.Tensor, tf.Tensor]:
        """Layer's logic - to be implemented in child class."""
        raise NotImplementedError

    def apply_mask_and_scaling(
        self, loss: tf.Tensor, mask: Optional[tf.Tensor]
    ) -> tf.Tensor:
        """Scales the loss and applies the mask if necessary.

        Args:
            loss: The loss tensor
            mask: (Optional) A mask to multiply with the loss

        Returns:
            The scaled loss, potentially averaged over the sequence
            dimension.
        """
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

        return loss


class SingleLabelDotProductLoss(DotProductLoss):
    """Single-label dot-product loss layer.

    This loss layer assumes that only one output (label) is correct for any given input.
    """

    def __init__(
        self,
        num_candidates: int,
        scale_loss: bool = False,
        constrain_similarities: bool = True,
        model_confidence: Text = SOFTMAX,
        similarity_type: Text = INNER,
        name: Optional[Text] = None,
        loss_type: Text = CROSS_ENTROPY,
        mu_pos: float = 0.8,
        mu_neg: float = -0.2,
        use_max_sim_neg: bool = True,
        neg_lambda: float = 0.5,
        same_sampling: bool = False,
        **kwargs: Any,
    ) -> None:
        """Declares instance variables with default values.

        Args:
            num_candidates: Positive integer, the number of incorrect labels;
                the algorithm will minimize their similarity to the input.
            loss_type: The type of the loss function, either `cross_entropy` or
                `margin`.
            mu_pos: Indicates how similar the algorithm should
                try to make embedding vectors for correct labels;
                should be 0.0 < ... < 1.0 for `cosine` similarity type.
            mu_neg: Maximum negative similarity for incorrect labels,
                should be -1.0 < ... < 1.0 for `cosine` similarity type.
            use_max_sim_neg: If `True` the algorithm only minimizes
                maximum similarity over incorrect intent labels,
                used only if `loss_type` is set to `margin`.
            neg_lambda: The scale of how important it is to minimize
                the maximum similarity between embeddings of different labels,
                used only if `loss_type` is set to `margin`.
            scale_loss: If `True` scale loss inverse proportionally to
                the confidence of the correct prediction.
            similarity_type: Similarity measure to use, either `cosine` or `inner`.
            name: Optional name of the layer.
            same_sampling: If `True` sample same negative labels
                for the whole batch.
            constrain_similarities: If `True` and loss_type is `cross_entropy`, a
                sigmoid loss term is added to the total loss to ensure that similarity
                values are approximately bounded.
            model_confidence: Normalization of confidence values during inference.
                Currently, the only possible value is `SOFTMAX`.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            num_candidates,
            scale_loss=scale_loss,
            constrain_similarities=constrain_similarities,
            model_confidence=model_confidence,
            similarity_type=similarity_type,
            name=name,
        )
        self.loss_type = loss_type
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        self.use_max_sim_neg = use_max_sim_neg
        self.neg_lambda = neg_lambda
        self.same_sampling = same_sampling

    def _get_bad_mask(
        self, labels: tf.Tensor, target_labels: tf.Tensor, idxs: tf.Tensor
    ) -> tf.Tensor:
        """Calculate bad mask for given indices.

        Checks that input features are different for positive negative samples.
        """
        pos_labels = tf.expand_dims(target_labels, axis=-2)
        neg_labels = layers_utils.get_candidate_values(labels, idxs)

        return tf.cast(
            tf.reduce_all(tf.equal(neg_labels, pos_labels), axis=-1), pos_labels.dtype
        )

    def _get_negs(
        self, embeds: tf.Tensor, labels: tf.Tensor, target_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Gets negative examples from given tensor."""
        embeds_flat = layers_utils.batch_flatten(embeds)
        labels_flat = layers_utils.batch_flatten(labels)
        target_labels_flat = layers_utils.batch_flatten(target_labels)

        total_candidates = tf.shape(embeds_flat)[0]
        target_size = tf.shape(target_labels_flat)[0]

        neg_ids = layers_utils.random_indices(
            target_size, self.num_neg, total_candidates
        )

        neg_embeds = layers_utils.get_candidate_values(embeds_flat, neg_ids)
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
        sim_pos = tf.squeeze(sim_pos, axis=-1)
        return layers_utils.reduce_mean_equal(max_all_sim, sim_pos)

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

    def _loss_cross_entropy(
        self,
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Defines cross entropy loss."""
        loss = self._compute_softmax_loss(
            sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li
        )

        if self.constrain_similarities:
            loss += self._compute_sigmoid_loss(
                sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li
            )

        loss = self.apply_mask_and_scaling(loss, mask)

        # average the loss over the batch
        return tf.reduce_mean(loss)

    @staticmethod
    def _compute_sigmoid_loss(
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
    ) -> tf.Tensor:
        # Constrain similarity values in a range by applying sigmoid
        # on them individually so that they saturate at extreme values.
        sigmoid_logits = tf.concat(
            [sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li], axis=-1
        )
        sigmoid_labels = tf.concat(
            [
                tf.ones_like(sigmoid_logits[..., :1]),
                tf.zeros_like(sigmoid_logits[..., 1:]),
            ],
            axis=-1,
        )
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=sigmoid_labels, logits=sigmoid_logits
        )
        # average over logits axis
        return tf.reduce_mean(sigmoid_loss, axis=-1)

    def _compute_softmax_loss(
        self,
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        sim_neg_ll: tf.Tensor,
        sim_neg_ii: tf.Tensor,
        sim_neg_li: tf.Tensor,
    ) -> tf.Tensor:
        # Similarity terms between input and label should be optimized relative
        # to each other and hence use them as logits for softmax term
        softmax_logits = tf.concat([sim_pos, sim_neg_il, sim_neg_li], axis=-1)
        if not self.constrain_similarities:
            # Concatenate other similarity terms as well. Due to this,
            # similarity values between input and label may not be
            # approximately bounded in a defined range.
            softmax_logits = tf.concat(
                [softmax_logits, sim_neg_ii, sim_neg_ll], axis=-1
            )
        # create label_ids for softmax
        softmax_label_ids = tf.zeros_like(softmax_logits[..., 0], tf.int32)
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=softmax_label_ids, logits=softmax_logits
        )
        return softmax_loss

    @property
    def _chosen_loss(self) -> Callable:
        """Use loss depending on given option."""
        if self.loss_type == MARGIN:
            return self._loss_margin
        elif self.loss_type == CROSS_ENTROPY:
            return self._loss_cross_entropy
        else:
            raise TFLayerConfigException(
                f"Wrong loss type '{self.loss_type}', "
                f"should be '{MARGIN}' or '{CROSS_ENTROPY}'"
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

        Args:
            inputs_embed: Embedding tensor for the batch inputs;
                shape `(batch_size, ..., num_features)`
            labels_embed: Embedding tensor for the batch labels;
                shape `(batch_size, ..., num_features)`
            labels: Tensor representing batch labels; shape `(batch_size, ..., 1)`
            all_labels_embed: Embedding tensor for the all labels;
                shape `(num_labels, num_features)`
            all_labels: Tensor representing all labels; shape `(num_labels, 1)`
            mask: Optional mask, contains `1` for inputs and `0` for padding;
                shape `(batch_size, 1)`

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


class MultiLabelDotProductLoss(DotProductLoss):
    """Multi-label dot-product loss layer.

    This loss layer assumes that multiple outputs (labels) can be correct for any given
    input. To accomodate for this, we use a sigmoid cross-entropy loss here.
    """

    def __init__(
        self,
        num_candidates: int,
        scale_loss: bool = False,
        constrain_similarities: bool = True,
        model_confidence: Text = SOFTMAX,
        similarity_type: Text = INNER,
        name: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        """Declares instance variables with default values.

        Args:
            num_candidates: Positive integer, the number of candidate labels.
            scale_loss: If `True` scale loss inverse proportionally to
                the confidence of the correct prediction.
            similarity_type: Similarity measure to use, either `cosine` or `inner`.
            name: Optional name of the layer.
            constrain_similarities: Boolean, if `True` applies sigmoid on all
                similarity terms and adds to the loss function to
                ensure that similarity values are approximately bounded.
                Used inside _loss_cross_entropy() only.
            model_confidence: Normalization of confidence values during inference.
                Currently, the only possible value is `SOFTMAX`.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            num_candidates,
            scale_loss=scale_loss,
            similarity_type=similarity_type,
            name=name,
            constrain_similarities=constrain_similarities,
            model_confidence=model_confidence,
        )

    def call(
        self,
        batch_inputs_embed: tf.Tensor,
        batch_labels_embed: tf.Tensor,
        batch_labels_ids: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_labels_ids: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculates loss and accuracy.

        Args:
            batch_inputs_embed: Embeddings of the batch inputs (e.g. featurized
                trackers); shape `(batch_size, 1, num_features)`
            batch_labels_embed: Embeddings of the batch labels (e.g. featurized intents
                for IntentTED);
                shape `(batch_size, max_num_labels_per_input, num_features)`
            batch_labels_ids: Batch label indices (e.g. indices of the intents). We
                assume that indices are integers that run from `0` to
                `(number of labels) - 1`.
                shape `(batch_size, max_num_labels_per_input, 1)`
            all_labels_embed: Embeddings for all labels in the domain;
                shape `(batch_size, num_features)`
            all_labels_ids: Indices for all labels in the domain;
                shape `(num_labels, 1)`
            mask: Optional sequence mask, which contains `1` for inputs and `0` for
                padding.

        Returns:
            loss: Total loss (based on StarSpace http://arxiv.org/abs/1709.03856);
                scalar
            accuracy: Training accuracy; scalar
        """
        (
            pos_inputs_embed,  # (batch_size, 1, 1, num_features)
            pos_labels_embed,  # (batch_size, 1, max_num_labels_per_input, num_features)
            candidate_labels_embed,  # (batch_size, 1, num_candidates, num_features)
            pos_neg_labels,  # (batch_size, num_candidates)
        ) = self._sample_candidates(
            batch_inputs_embed,
            batch_labels_embed,
            batch_labels_ids,
            all_labels_embed,
            all_labels_ids,
        )

        # Calculate similarities
        sim_pos, sim_candidate_il = self._train_sim(
            pos_inputs_embed, pos_labels_embed, candidate_labels_embed, mask
        )

        label_padding_mask = self._construct_mask_for_label_padding(
            batch_labels_ids, tf.shape(pos_neg_labels)[-1]
        )

        # Repurpose the `mask` argument of `_accuracy` and `_loss_sigmoid`
        # to pass the `label_padding_mask`. We can do this right now because
        # we don't use `MultiLabelDotProductLoss` for sequence tagging tasks
        # yet. Hence, the `mask` argument passed to this function will always
        # be empty. Whenever, we come across a use case where `mask` is
        # non-empty we'll have to refactor the `_accuracy` and `_loss_sigmoid`
        # functions to take into consideration both, sequence level masks as
        # well as label padding masks.

        accuracy = self._accuracy(
            sim_pos, sim_candidate_il, pos_neg_labels, label_padding_mask
        )
        loss = self._loss_sigmoid(
            sim_pos, sim_candidate_il, pos_neg_labels, mask=label_padding_mask
        )

        return loss, accuracy

    @staticmethod
    def _construct_mask_for_label_padding(
        batch_labels_ids: tf.Tensor, num_candidates: tf.Tensor
    ) -> tf.Tensor:
        """Constructs a mask which indicates indices for valid label ids.

        Indices corresponding to valid label ids have a
        `1` and indices corresponding to `LABEL_PAD_ID`
        have a `0`.

        Args:
            batch_labels_ids: Batch label indices (e.g. indices of the intents). We
                assume that indices are integers that run from `0` to
                `(number of labels) - 1` with a special
                value for padding which is set to `LABEL_PAD_ID`.
                shape `(batch_size, max_num_labels_per_input, 1)`
            num_candidates: Number of candidates sampled.

        Returns:
            Constructed mask.
        """
        pos_label_pad_indices = tf.cast(
            tf.equal(tf.squeeze(batch_labels_ids, -1), LABEL_PAD_ID), dtype=tf.float32
        )

        # Flip 1 and 0 to 0 and 1 respectively
        pos_label_pad_mask = 1 - pos_label_pad_indices

        # `pos_label_pad_mask` only contains the mask for label ids
        # seen in the batch. For sampled candidate label ids, the mask
        # should be a tensor of `1`s since all candidate label ids
        # are valid. From this, we construct the padding mask for
        # all label ids: label ids seen in the batch + label ids sampled.
        all_label_pad_mask = tf.concat(
            [
                pos_label_pad_mask,
                tf.ones(
                    (tf.shape(batch_labels_ids)[0], num_candidates), dtype=tf.float32
                ),
            ],
            axis=-1,
        )

        return all_label_pad_mask

    def _train_sim(
        self,
        pos_inputs_embed: tf.Tensor,
        pos_labels_embed: tf.Tensor,
        candidate_labels_embed: tf.Tensor,
        mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        sim_pos = self.sim(
            pos_inputs_embed, pos_labels_embed, mask
        )  # (batch_size, 1, max_labels_per_input)
        sim_candidate_il = self.sim(
            pos_inputs_embed, candidate_labels_embed, mask
        )  # (batch_size, 1, num_candidates)

        return sim_pos, sim_candidate_il

    def _sample_candidates(
        self,
        batch_inputs_embed: tf.Tensor,
        batch_labels_embed: tf.Tensor,
        batch_labels_ids: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_labels_ids: tf.Tensor,
    ) -> Tuple[
        tf.Tensor,  # (batch_size, 1, 1, num_features)
        tf.Tensor,  # (batch_size, 1, num_features)
        tf.Tensor,  # (batch_size, 1, num_candidates, num_features)
        tf.Tensor,  # (batch_size, num_candidates)
    ]:
        """Samples candidate examples.

        Args:
            batch_inputs_embed: Embeddings of the batch inputs (e.g. featurized
                trackers) # (batch_size, 1, num_features)
            batch_labels_embed: Embeddings of the batch labels (e.g. featurized intents
                for IntentTED) # (batch_size, max_num_labels_per_input, num_features)
            batch_labels_ids: Batch label indices (e.g. indices of the
                intents) # (batch_size, max_num_labels_per_input, 1)
            all_labels_embed: Embeddings for all labels in
                the domain # (num_labels, num_features)
            all_labels_ids: Indices for all labels in the
                domain # (num_labels, 1)

        Returns:
            pos_inputs_embed: Embeddings of the batch inputs
            pos_labels_embed: Embeddings of the batch labels with an extra
                dimension inserted.
            candidate_labels_embed: More examples of embeddings of labels, some positive
                some negative
            pos_neg_indicators: Indicator for which candidates are positives and which
                are negatives
        """
        pos_inputs_embed = tf.expand_dims(
            batch_inputs_embed, axis=-2, name="expand_pos_input"
        )

        pos_labels_embed = tf.expand_dims(
            batch_labels_embed, axis=1, name="expand_pos_labels"
        )

        # Pick random examples from the batch
        candidate_ids = layers_utils.random_indices(
            batch_size=tf.shape(batch_inputs_embed)[0],
            n=self.num_neg,
            n_max=tf.shape(all_labels_embed)[0],
        )

        # Get the label embeddings corresponding to candidate indices
        candidate_labels_embed = layers_utils.get_candidate_values(
            all_labels_embed, candidate_ids
        )
        candidate_labels_embed = tf.expand_dims(candidate_labels_embed, axis=1)

        # Get binary indicators of whether a candidate is positive or not
        pos_neg_indicators = self._get_pos_neg_indicators(
            all_labels_ids, batch_labels_ids, candidate_ids
        )

        return (
            pos_inputs_embed,
            pos_labels_embed,
            candidate_labels_embed,
            pos_neg_indicators,
        )

    def _get_pos_neg_indicators(
        self,
        all_labels_ids: tf.Tensor,
        batch_labels_ids: tf.Tensor,
        candidate_ids: tf.Tensor,
    ) -> tf.Tensor:
        """Computes indicators for which candidates are positive labels.

        Args:
            all_labels_ids: Indices of all the labels
            batch_labels_ids: Indices of the labels in the examples
            candidate_ids: Indices of labels that may or may not appear in the examples

        Returns:
            Binary indicators of whether or not a label is positive
        """
        candidate_labels_ids = layers_utils.get_candidate_values(
            all_labels_ids, candidate_ids
        )
        candidate_labels_ids = tf.expand_dims(candidate_labels_ids, axis=1)

        # Determine how many distinct labels exist (highest label index)
        max_label_id = tf.cast(tf.math.reduce_max(all_labels_ids), dtype=tf.int32)

        # Convert the positive label ids to their one_hot representation.
        # Note: -1 indices yield a zeros-only vector. We use -1 as a padding token,
        # as the number of positive labels in each example can differ. The padding is
        # added in the TrackerFeaturizer.
        batch_labels_one_hot = tf.one_hot(
            tf.cast(tf.squeeze(batch_labels_ids, axis=-1), tf.int32),
            max_label_id + 1,
            axis=-1,
        )  # (batch_size, max_num_labels_per_input, max_label_id)

        # Collapse the extra dimension and convert to a multi-hot representation
        # by aggregating all ones in the one-hot representation.
        # We use tf.reduce_any instead of tf.reduce_sum because several examples can
        # have the same postivie label.
        batch_labels_multi_hot = tf.cast(
            tf.math.reduce_any(tf.cast(batch_labels_one_hot, dtype=tf.bool), axis=-2),
            tf.float32,
        )  # (batch_size, max_label_id)

        # Remove extra dimensions for gather
        candidate_labels_ids = tf.squeeze(tf.squeeze(candidate_labels_ids, 1), -1)

        # Collect binary indicators of whether or not a label is positive
        return tf.gather(
            batch_labels_multi_hot,
            tf.cast(candidate_labels_ids, tf.int32),
            batch_dims=1,
            name="gather_labels",
        )

    def _loss_sigmoid(
        self,
        sim_pos: tf.Tensor,  # (batch_size, 1, max_num_labels_per_input)
        sim_candidates_il: tf.Tensor,  # (batch_size, 1, num_candidates)
        pos_neg_labels: tf.Tensor,  # (batch_size, num_candidates)
        mask: Optional[
            tf.Tensor
        ] = None,  # (batch_size, max_num_labels_per_input + num_candidates)
    ) -> tf.Tensor:  # ()
        """Computes the sigmoid loss."""
        # Concatenate the guaranteed positive examples with the candidate examples,
        # some of which are positives and others are negatives. Which are which
        # is stored in `pos_neg_labels`.
        logits = tf.concat([sim_pos, sim_candidates_il], axis=-1, name="logit_concat")
        logits = tf.squeeze(logits, 1)

        # Create label_ids for sigmoid. `mask` will take care of the
        # extra 1s we create as label ids for indices corresponding
        # to padding ids.
        pos_label_ids = tf.squeeze(tf.ones_like(sim_pos, tf.float32), 1)
        label_ids = tf.concat(
            [pos_label_ids, pos_neg_labels], axis=-1, name="gt_concat"
        )

        # Compute the sigmoid cross-entropy loss. When minimized, the embeddings
        # for the two classes (positive and negative) are pushed away from each
        # other in the embedding space, while it is allowed that any input embedding
        # corresponds to more than one label.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_ids, logits=logits)

        loss = self.apply_mask_and_scaling(loss, mask)

        # Average the loss over the batch
        return tf.reduce_mean(loss)

    @staticmethod
    def _accuracy(
        sim_pos: tf.Tensor,  # (batch_size, 1, max_num_labels_per_input)
        sim_candidates: tf.Tensor,  # (batch_size, 1, num_candidates)
        pos_neg_indicators: tf.Tensor,  # (batch_size, num_candidates)
        mask: tf.Tensor,  # (batch_size, max_num_labels_per_input + num_candidates)
    ) -> tf.Tensor:  # ()
        """Calculates the accuracy."""
        all_preds = tf.concat(
            [sim_pos, sim_candidates], axis=-1, name="acc_concat_preds"
        )
        all_preds_sigmoid = tf.nn.sigmoid(all_preds)
        all_pred_labels = tf.squeeze(tf.math.round(all_preds_sigmoid), 1)

        # Create an indicator for the positive labels by concatenating the 1 for all
        # guaranteed positive labels and the `pos_neg_indicators`
        all_positives = tf.concat(
            [tf.squeeze(tf.ones_like(sim_pos), axis=1), pos_neg_indicators],
            axis=-1,
            name="acc_concat_gt",
        )

        return layers_utils.reduce_mean_equal(all_pred_labels, all_positives, mask=mask)
