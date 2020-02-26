import logging
from typing import List, Optional, Text, Tuple, Callable, Union, Any
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K
from rasa.utils.tensorflow.constants import SOFTMAX, MARGIN, COSINE, INNER

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
        # need to explicitly set shape, because it becomes dynamic after `retain`
        # noinspection PyProtectedMember
        outputs._dense_shape = inputs._dense_shape

        return outputs


class DenseForSparse(tf.keras.layers.Dense):
    """Dense layer for sparse input tensor."""

    def __init__(self, reg_lambda: float = 0, **kwargs: Any) -> None:
        if reg_lambda > 0:
            regularizer = tf.keras.regularizers.l2(reg_lambda)
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
    def __init__(self, sparsity: float = 0.8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sparsity = sparsity

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)
        # create random mask to set some weights to 0
        kernel_mask = tf.random.uniform(tf.shape(self.kernel), 0, 1)
        kernel_mask = tf.cast(
            tf.greater_equal(kernel_mask, self.sparsity), self.kernel.dtype
        )
        self.kernel_mask = tf.Variable(
            initial_value=kernel_mask, trainable=False, name="kernel_mask"
        )

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

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self._dense(x)
        if self.similarity_type == COSINE:
            x = tf.nn.l2_normalize(x, axis=-1)

        return x


class InputMask(tf.keras.layers.Layer):
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


class CRF(tf.keras.layers.Layer):
    def __init__(
        self, num_tags: int, reg_lambda: float, name: Optional[Text] = None
    ) -> None:
        super().__init__(name=name)
        self.num_tags = num_tags
        self.transition_regularizer = tf.keras.regularizers.l2(reg_lambda)

    def build(self, input_shape: tf.TensorShape) -> None:
        # the weights should be created in `build` to apply random_seed
        self.transition_params = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            regularizer=self.transition_regularizer,
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

        if mask is None:
            mask = 1.0

        if self.scale_loss:
            # mask loss by prediction confidence
            pos_pred = tf.stop_gradient(tf.nn.softmax(logits)[..., 0])
            # the scaling parameters are found empirically
            scale_mask = mask * tf.pow(tf.minimum(0.5, 1 - pos_pred) / 0.5, 4)
            # scale loss
            loss *= scale_mask

        if len(loss.shape) == 2:
            # average over the sequence
            loss = tf.reduce_sum(loss, axis=-1) / tf.reduce_sum(mask, axis=-1)

        # average the loss over all examples
        loss = tf.reduce_mean(loss)

        return loss

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
