import logging
from typing import List, Optional, Text, Tuple, Callable, Union, Any
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
                outputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], -1)
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

        return tf.while_loop(
            cond,
            body,
            loop_vars=[idx1, out1],
            shape_invariants=[idx1.shape, tf.TensorShape([None, self.num_neg])],
            parallel_iterations=self.parallel_iterations,
            back_prop=False,
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


class MultiLabelDotProductLoss(DotProductLoss):
    def _get_neg_values(self, tensor, neg_ids):

        tensor_expanded = tf.tile(
            tf.expand_dims(self._make_flat(tensor), 0), (tf.shape(tensor)[0], 1, 1)
        )
        neg_tensor = tf.gather(tensor_expanded, neg_ids, batch_dims=1)
        neg_tensor = tf.expand_dims(neg_tensor, axis=1)

        return neg_tensor

    def _sample_negatives(
        self,
        inputs_embed: tf.Tensor,
        labels_embed: tf.Tensor,
        label_ids: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_label_ids: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Sample negative examples."""

        # print("Inputs embed", inputs_embed.shape, "labels embed", labels_embed.shape)

        pos_inputs_embed = tf.expand_dims(
            inputs_embed, axis=-2, name="expand_pos_input"
        )
        # Take only the first label as positive, others may or may not be picked during random sampling
        pos_labels_embed = tf.expand_dims(
            tf.expand_dims(labels_embed[:, 0, ...], axis=-2),
            axis=1,
            name="expand_pos_labels",
        )

        # tf.print("tensorsssss")
        # tf.print(label_ids)
        # tf.print("------------")

        # print("Pos input embed shape", pos_inputs_embed.shape)
        # print("Pos labels embed shape", pos_labels_embed.shape)

        # get neg ids from batch itself
        neg_ids = self._get_neg_indices(
            tf.shape(inputs_embed)[0], tf.shape(inputs_embed)[0]
        )

        # tf.print(neg_ids)
        # tf.print("------------")

        # print("Neg ids shape", neg_ids.shape)
        #
        # print("labels_embed shape", labels_embed.shape)

        # sample negative labels
        # print("Neg ids", neg_ids.shape)
        # neg_labels_embed = self._get_negs(labels_embed, label_ids, label_ids, neg_ids)
        neg_labels_embed = self._get_neg_values(labels_embed[:, 0, ...], neg_ids)
        # neg_labels_embed = tf.squeeze(neg_labels_emb)
        # neg_labels_embed = labels_embed
        # neg_label_ids = self._get_negs(label_ids, label_ids, label_ids, neg_ids)
        neg_labels_ids = self._get_neg_values(label_ids[:, 0, ...], neg_ids)

        # tf.print(neg_labels_ids)
        # tf.print("------------")

        # print("Neg labels embed shape", neg_labels_embed.shape)
        # print("Neg label ids shape", neg_labels_ids.shape)

        # print('Neg labels', neg_labels.shape, 'all_gt_labels', all_gt_labels.shape)
        max_label_id = tf.cast(tf.math.reduce_max(all_label_ids), dtype=tf.int32)

        # tf.print(max_label_id)
        # tf.print("------------")

        # +1 for padding and dimension size is 1 indexed and hence 1 more than maximum label id
        depth_needed = max_label_id + 1

        batch_labels_one_hot = tf.one_hot(
            tf.cast(tf.squeeze(label_ids, axis=-1), tf.int32), depth_needed, axis=-1
        )  # bs x num_pos_labels(varied) x num label ids

        # tf.print(batch_labels_one_hot)
        # tf.print("------------")

        # Collapse the extra dimension aggregating all ones
        # Here tf.reduce_any is important and tf.reduce_sum
        # cannot be used. Reason being that due to padding in label_ids,
        # there can be more than one 1's at the 0'th dimension.
        batch_labels_multi_hot = tf.cast(
            tf.math.reduce_any(tf.cast(batch_labels_one_hot, dtype=tf.bool), axis=-2),
            tf.float32,
        )  # bs x num label ids

        # tf.print(batch_labels_multi_hot)
        # tf.print("------------")

        # print("batch mh labels shape", batch_labels_multi_hot.shape)

        # Remove extra dimensions for gather
        neg_labels_ids = tf.squeeze(tf.squeeze(neg_labels_ids, 1), -1)

        # tf.print(neg_labels_ids)
        # tf.print("------------")

        # sample pos neg labels
        pos_neg_labels = tf.gather(
            batch_labels_multi_hot,
            tf.cast(neg_labels_ids, tf.int32),
            batch_dims=1,
            name="gather_labels",
        )

        pos_neg_labels = tf.cast(pos_neg_labels, tf.float32)
        # tf.print(pos_neg_labels)
        # tf.print("------------")
        # print("Pos neg labels shape", pos_neg_labels.shape)

        # tf.print("Label ids")
        # tf.print(label_ids)
        # tf.print("---------------")
        # tf.print(neg_ids)
        # tf.print("----------------")
        # tf.print(batch_labels_multi_hot)
        # tf.print("----------------")
        # tf.print(pos_neg_labels)
        # tf.print("=================================")
        return pos_inputs_embed, pos_labels_embed, neg_labels_embed, pos_neg_labels

    def _get_neg_indices(self, target_size, total_candidates):

        # print('Target, cands = ', target_size, total_candidates)

        neg_ids = self._random_indices(target_size, total_candidates)

        return neg_ids

    def _get_negs(
        self,
        embeds: tf.Tensor,
        labels: tf.Tensor,
        target_labels: tf.Tensor,
        neg_ids: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get negative examples from given tensor."""

        embeds_flat = self._make_flat(embeds)
        target_size = tf.shape(embeds_flat)[0]
        neg_embeds = self._sample_idxs(target_size, embeds_flat, neg_ids)

        # print("neg embeds inside _get_negs shape", neg_embeds.shape)

        # check if inputs have sequence dimension
        if len(embeds.shape) == 3:
            # tensors were flattened for sampling, reshape back
            # add sequence dimension if it was present in the inputs
            target_shape = tf.shape(target_labels)
            neg_embeds = tf.reshape(
                neg_embeds, (target_shape[0], target_shape[1], -1, embeds.shape[-1])
            )

        return neg_embeds

    def _get_bad_mask(self, pos_d: tf.Tensor, neg_d: tf.Tensor):

        bad_mask = tf.cast(tf.reduce_all(tf.equal(pos_d, neg_d), axis=-1), pos_d.dtype)

        return bad_mask

    def _loss_sigmoid(
        self,
        sim_pos: tf.Tensor,
        sim_neg_il: tf.Tensor,
        pos_neg_labels: tf.Tensor,
        # sim_neg_ll: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Define sigmoid loss."""

        # print("Sim pos", sim_pos.shape)
        # print("Sim il", sim_neg_il.shape)
        # print("Pos neg labels", pos_neg_labels.shape)

        logits = tf.concat([sim_pos, sim_neg_il], axis=-1, name="logit_concat")
        logits = tf.squeeze(logits, 1)

        # print("logits", logits.shape)

        # create label_ids for sigmoid
        pos_label_ids = tf.ones_like(sim_pos, tf.float32)
        pos_label_ids = tf.squeeze(pos_label_ids, -1)

        # print("pos label ids", pos_label_ids.shape)

        label_ids = tf.concat(
            [pos_label_ids, pos_neg_labels], axis=-1, name="gt_concat"
        )

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_ids, logits=logits)

        # print("Loss shape", loss.shape)

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

    def _train_sim(
        self,
        pos_inputs_embed: tf.Tensor,
        pos_labels_embed: tf.Tensor,
        neg_labels_embed: tf.Tensor,
        # bad_neg_labels: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Define similarity."""

        # calculate similarity with several
        # embedded actions for the loss
        # neg_inf = tf.constant(-1e9)

        sim_pos = self.sim(pos_inputs_embed, pos_labels_embed, mask)
        sim_neg_il = self.sim(pos_inputs_embed, neg_labels_embed, mask)
        # sem_neg_ll = self.sim(pos_labels_embed, neg_labels_embed, mask)

        # print("Sim pos shape", sim_pos.shape)
        # print("Sim neg_il shape", sim_neg_il.shape)

        # sim_neg_ll = (
        #     self.sim(pos_labels_embed, neg_labels_embed, mask) + neg_inf * bad_neg_labels
        # )

        return sim_pos, sim_neg_il  # , sim_neg_ll

    def call(
        self,
        batch_inputs_embed: tf.Tensor,
        batch_labels_embed: tf.Tensor,
        batch_label_ids: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_label_ids: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate loss and accuracy.
        Arguments:
            batch_inputs_embed: Embedding tensor for the batch inputs.
            batch_labels_embed: Embedding tensor for the batch labels.
            batch_label_ids: Tensor representing batch labels.
            all_labels_embed: Embedding tensor for the all labels.
            all_label_ids: Tensor representing all labels.
            mask: Optional tensor representing sequence mask,
                contains `1` for inputs and `0` for padding.
        Returns:
            loss: Total loss.
            accuracy: Training accuracy.
        """
        (
            pos_inputs_embed,
            pos_labels_embed,
            neg_labels_embed,
            pos_neg_labels,
        ) = self._sample_negatives(
            batch_inputs_embed,
            batch_labels_embed,
            batch_label_ids,
            all_labels_embed,
            all_label_ids,
        )

        # print("Sampled ------------")
        # print(
        #     pos_inputs_embed.shape,
        #     pos_labels_embed.shape,
        #     neg_labels_embed.shape,
        #     pos_neg_labels.shape,
        # )

        # print("mask shape", mask.shape)

        # calculate similarities
        sim_pos, sim_neg_il = self._train_sim(
            pos_inputs_embed, pos_labels_embed, neg_labels_embed, mask
        )

        accuracy = self._calc_accuracy(sim_pos, sim_neg_il, pos_neg_labels)

        loss = self._loss_sigmoid(
            sim_pos,
            sim_neg_il,
            pos_neg_labels
            # , sim_neg_ll
            ,
            mask,
        )

        return loss, accuracy

    @staticmethod
    def _calc_accuracy(
        sim_pos: tf.Tensor, sim_neg: tf.Tensor, pos_neg_labels: tf.Tensor
    ) -> tf.Tensor:
        """Calculate accuracy."""

        all_preds = tf.concat([sim_pos, sim_neg], axis=-1, name="acc_concat_preds")
        all_preds_sigmoid = tf.nn.sigmoid(all_preds)
        all_pred_labels = tf.squeeze(tf.math.round(all_preds_sigmoid), 1)

        complete_gt = tf.concat(
            [tf.squeeze(tf.ones_like(sim_pos), axis=-1), pos_neg_labels],
            axis=-1,
            name="acc_concat_gt",
        )

        # tf.print(all_pred_labels)
        # tf.print(complete_gt)

        acc = tf.reduce_mean(
            tf.cast(tf.math.equal(all_pred_labels, complete_gt), tf.float32)
        )

        return acc

    @staticmethod
    def confidence_from_sim(sim: tf.Tensor, similarity_type: Text) -> tf.Tensor:
        if similarity_type == COSINE:
            # clip negative values to zero
            return tf.nn.relu(sim)
        else:
            # normalize result to [0, 1] with sigmoid
            return tf.nn.sigmoid(sim)
