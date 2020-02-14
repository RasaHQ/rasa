import logging
from typing import List, Optional, Text, Tuple, Callable, Union
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers

logger = logging.getLogger(__name__)


class SparseDropout(tf.keras.layers.Dropout):
    def call(
        self, inputs: tf.Tensor, training: Optional[Union[tf.Tensor, bool]] = None
    ) -> tf.Tensor:
        if training is None:
            training = K.learning_phase()

        def dropped_inputs() -> tf.Tensor:
            to_retain_prob = tf.random.uniform(
                tf.shape(inputs.values), 0, 1, inputs.values.dtype
            )
            to_retain = tf.greater_equal(to_retain_prob, self.rate)
            return tf.sparse.retain(inputs, to_retain)

        outputs = tf_utils.smart_cond(
            training, dropped_inputs, lambda: tf.identity(inputs)
        )
        # noinspection PyProtectedMember
        outputs._dense_shape = inputs._dense_shape

        return outputs


class DenseForSparse(tf.keras.layers.Dense):
    """Dense layer for sparse input tensor."""

    def __init__(self, reg_lambda: float = 0, **kwargs) -> None:
        if reg_lambda > 0:
            regularizer = tf.keras.regularizers.l1(reg_lambda)
        else:
            regularizer = None

        super().__init__(kernel_regularizer=regularizer, **kwargs)

    def call(self, inputs: tf.SparseTensor) -> tf.Tensor:
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
    def __init__(self, sparsity: int = 0.8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sparsity = sparsity

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        # create random mask to set some weights to 0
        kernel_mask = tf.random.uniform(tf.shape(self.kernel), 0, 1)
        kernel_mask = tf.cast(
            tf.greater_equal(kernel_mask, self.sparsity), self.kernel.dtype
        )
        self.kernel_mask = tf.Variable(initial_value=kernel_mask, trainable=False)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # set some weights to 0 according to precomputed mask
        self.kernel.assign(self.kernel * self.kernel_mask)
        return super().call(inputs)


class Ffnn(tf.keras.layers.Layer):
    """Create feed-forward network with hidden layers and name suffix."""

    def __init__(
        self,
        layer_sizes: List[int],
        dropout_rate: float,
        reg_lambda: float,
        layer_name_suffix: Text,
    ) -> None:
        super().__init__(name=f"ffnn_{layer_name_suffix}")

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._ffn_layers = []
        for i, layer_size in enumerate(layer_sizes):
            self._ffn_layers.append(
                DenseWithSparseWeights(
                    units=layer_size,
                    activation=tfa.activations.gelu,
                    kernel_regularizer=l2_regularizer,
                    name=f"hidden_layer_{layer_name_suffix}_{i}",
                )
            )
            self._ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))

    def call(
        self, x: tf.Tensor, training: Optional[Union[tf.Tensor, bool]] = None
    ) -> tf.Tensor:
        if training is None:
            training = K.learning_phase()

        for layer in self._ffn_layers:
            x = layer(x, training=training)

        return x


class Embed(tf.keras.layers.Layer):
    """Create dense embedding layer with a name."""

    def __init__(
        self,
        embed_dim: int,
        reg_lambda: float,
        layer_name_suffix: Text,
        similarity_type: Optional[Text] = None,
    ) -> None:
        super().__init__(name=f"embed_{layer_name_suffix}")

        self.similarity_type = similarity_type
        if self.similarity_type and self.similarity_type not in {"cosine", "inner"}:
            raise ValueError(
                f"Wrong similarity type '{self.similarity_type}', "
                f"should be 'cosine' or 'inner'"
            )

        regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._dense = tf.keras.layers.Dense(
            units=embed_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name=f"embed_layer_{layer_name_suffix}",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self._dense(x)
        if self.similarity_type == "cosine":
            x = tf.nn.l2_normalize(x, -1)

        return x


class InputMask(tf.keras.layers.Layer):
    def build(self, input_shape: tf.TensorShape) -> None:
        self.mask_vector = self.add_weight(
            shape=(1, 1, input_shape[-1]), name="mask_vector"
        )
        self.built = True

    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor,
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly mask input sequences."""

        if training is None:
            training = K.learning_phase()

        lm_mask_prob = tf.random.uniform(tf.shape(mask), 0, 1, mask.dtype) * mask
        lm_mask_bool = tf.greater_equal(lm_mask_prob, 0.85)

        def x_masked():
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
                other_prob < 0.70,
                mask_vector,
                tf.where(other_prob < 0.80, x_shuffle, x),
            )

            return tf.where(tf.tile(lm_mask_bool, (1, 1, x.shape[-1])), x_other, x)

        return (
            tf_utils.smart_cond(training, x_masked, lambda: tf.identity(x)),
            lm_mask_bool,
        )


class CRF(tf.keras.layers.Layer):
    def __init__(self, num_tags: int, reg_lambda: float, name: Text = None) -> None:
        super().__init__(name=name)
        self.num_tags = num_tags
        self.regularizer = tf.keras.regularizers.l1(reg_lambda)

    def build(self, input_shape: tf.TensorShape) -> None:
        # should be created in `build` to apply random_seed
        self.transition_params = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            regularizer=self.regularizer,
            name="transitions",
        )
        self.built = True

    def call(self, logits: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        pred_ids, _ = tfa.text.crf.crf_decode(
            logits, self.transition_params, sequence_lengths
        )
        # set prediction index for padding to `0`
        mask = tf.sequence_mask(
            sequence_lengths, maxlen=tf.shape(pred_ids)[1], dtype=pred_ids.dtype
        )

        return pred_ids * mask

    def loss(
        self, logits: tf.Tensor, tag_indices: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> tf.Tensor:
        log_likelihood, _ = tfa.text.crf.crf_log_likelihood(
            logits, tag_indices, sequence_lengths, self.transition_params
        )
        return tf.reduce_mean(-log_likelihood)


class DotProductLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        num_neg: int,
        loss_type: Text,
        mu_pos: float,
        mu_neg: float,
        use_max_sim_neg: bool,
        neg_lambda: float,
        scale_loss: bool,
        name: Text = None,
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

    def _random_indices(self, batch_size: tf.Tensor, total_candidates: tf.Tensor):
        def rand_idxs():
            """Create random tensor of indices"""

            # (1, num_neg)
            return tf.expand_dims(
                tf.random.shuffle(tf.range(total_candidates))[: self.num_neg], 0
            )

        if self.same_sampling:
            return tf.tile(rand_idxs(), (batch_size, 1))

        def cond(i, out):
            """Condition for while loop"""
            return i < batch_size

        def body(i, out):
            """Body of the while loop"""
            return [
                # increment counter
                i + 1,
                # add random indices
                tf.concat([out, rand_idxs()], 0),
            ]

        # first tensor already created
        i1 = tf.constant(1)
        # create first random array of indices
        out1 = rand_idxs()  # (1, num_neg)

        return tf.while_loop(
            cond,
            body,
            loop_vars=[i1, out1],
            shape_invariants=[i1.shape, tf.TensorShape([None, self.num_neg])],
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

        pos_labels = tf.expand_dims(target_labels, -2)
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

        if len(target_labels.shape) == 3:
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

        pos_inputs_embed = tf.expand_dims(inputs_embed, -2)
        pos_labels_embed = tf.expand_dims(labels_embed, -2)

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

        sim = tf.reduce_sum(a * b, -1)
        if mask is not None:
            sim *= tf.expand_dims(mask, 2)

        return sim

    @staticmethod
    def confidence_from_sim(sim: tf.Tensor, similarity_type: Text) -> tf.Tensor:
        if similarity_type == "cosine":
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

        max_all_sim = tf.reduce_max(tf.concat([sim_pos, sim_neg], -1), -1)
        return tf.reduce_mean(
            tf.cast(tf.math.equal(max_all_sim, tf.squeeze(sim_pos, -1)), tf.float32)
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
        loss = tf.maximum(0.0, self.mu_pos - tf.squeeze(sim_pos, -1))

        # loss for minimizing similarity with `num_neg` incorrect actions
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg_il = tf.reduce_max(sim_neg_il, -1)
            loss += tf.maximum(0.0, self.mu_neg + max_sim_neg_il)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0.0, self.mu_neg + sim_neg_il)
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between pos bot and neg bot embeddings
        max_sim_neg_ll = tf.maximum(0.0, self.mu_neg + tf.reduce_max(sim_neg_ll, -1))
        loss += max_sim_neg_ll * self.neg_lambda

        # penalize max similarity between pos dial and neg dial embeddings
        max_sim_neg_ii = tf.maximum(0.0, self.mu_neg + tf.reduce_max(sim_neg_ii, -1))
        loss += max_sim_neg_ii * self.neg_lambda

        # penalize max similarity between pos bot and neg dial embeddings
        max_sim_neg_li = tf.maximum(0.0, self.mu_neg + tf.reduce_max(sim_neg_li, -1))
        loss += max_sim_neg_li * self.neg_lambda

        if mask is not None:
            # mask loss for different length sequences
            loss *= mask
            # average the loss over sequence length
            loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)

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
            [sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li], -1
        )

        # pos_labels = tf.ones_like(logits[..., :1])
        # neg_labels = tf.zeros_like(logits[..., 1:])
        # labels = tf.concat([pos_labels, neg_labels], -1)

        # create label_ids for softmax
        label_ids = tf.zeros_like(logits[..., 0], tf.int32)

        # loss = entmax15_loss_with_logits(labels, logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_ids, logits=logits
        )

        if mask is None:
            mask = 1.0

        if self.scale_loss:
            # mask loss by prediction confidence
            pos_pred = tf.stop_gradient(tf.nn.softmax(logits)[..., 0])
            scale_mask = mask * tf.pow(tf.minimum(0.5, 1 - pos_pred) / 0.5, 4)
            # scale loss
            loss *= scale_mask

        if len(loss.shape) == 2:
            # average over the sequence
            loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, -1)

        # average the loss over all examples
        loss = tf.reduce_mean(loss)

        return loss

    @property
    def _chosen_loss(self) -> Callable:
        """Use loss depending on given option."""

        if self.loss_type == "margin":
            return self._loss_margin
        elif self.loss_type == "softmax":
            return self._loss_softmax
        else:
            raise ValueError(
                f"Wrong loss type '{self.loss_type}', "
                f"should be 'margin' or 'softmax'"
            )

    def call(
        self,
        inputs_embed: tf.Tensor,
        labels_embed: tf.Tensor,
        labels: tf.Tensor,
        all_labels_embed: tf.Tensor,
        all_labels: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate loss and accuracy."""

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

        acc = self._calc_accuracy(sim_pos, sim_neg_il)

        loss = self._chosen_loss(
            sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li, mask
        )

        return loss, acc


# https://gist.github.com/justheuristic/60167e77a95221586be315ae527c3cbd
def entmax15(inputs, axis=-1):
    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """

    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope('entmax'):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= tf.reduce_max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope('entmax_grad'):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keepdims=True)
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs

        return outputs, grad_fn

    return _entmax_inner(inputs)


@tf.custom_gradient
def sparse_entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param labels: reference answers vector int64[batch_size] \in [0, num_classes)
    :param logits: output matrix float32[batch_size, num_classes] (not actually logits :)
    :returns: elementwise loss, float32[batch_size]
    """
    assert labels.shape.ndims == logits.shape.ndims - 1
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - tf.one_hot(labels, depth=tf.shape(logits)[-1], axis=-1)
        # loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)
        loss = omega_entmax15 + tf.reduce_sum(p_star * logits, axis=-1)

    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


@tf.custom_gradient
def entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param logits: "logits" matrix float32[batch_size, num_classes]
    :param labels: reference answers indicators, float32[batch_size, num_classes]
    :returns: elementwise loss, float32[batch_size]
    WARNING: this function does not propagate gradients through :labels:
    This behavior is the same as like softmax_crossentropy_with_logits v1
    It may become an issue if you do something like co-distillation
    """
    assert labels.shape.ndims == logits.shape.ndims
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - labels
        # loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)
        loss = omega_entmax15 + tf.reduce_sum(p_star * logits, axis=-1)

    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


def top_k_over_axis(inputs, k, axis=-1, **kwargs):
    """ performs tf.nn.top_k over any chosen axis """
    with tf.name_scope('top_k_along_axis'):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = tf.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = tf.nn.top_k(
            input_perm, k=k, **kwargs)

        input_sorted = tf.transpose(input_perm_sorted, inv_order)
        sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axis(values, indices, gather_axis):
    """
    replicates the behavior of torch.gather for tf<=1.8;
    for newer versions use tf.gather with batch_dims
    :param values: tensor [d0, ..., dn]
    :param indices: int64 tensor of same shape as values except for gather_axis
    :param gather_axis: performs gather along this axis
    :returns: gathered values, same shape as values except for gather_axis
        If gather_axis == 2
        gathered_values[i, j, k, ...] = values[i, j, indices[i, j, k, ...], ...]
        see torch.gather for more detils
    """
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype)
            index_i = tf.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = tf.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))


def entmax_threshold_and_support(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with tf.name_scope('entmax_threshold_and_support'):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axis(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, inputs_sorted), dtype=tf.int64), axis=axis, keepdims=True)

        tau_star = gather_over_axis(tau, support_size - 1, axis)
    return tau_star, support_size
