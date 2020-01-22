import logging
from typing import List, Optional, Text, Tuple, Callable
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

logger = logging.getLogger(__name__)


class SparseDropout(tf.keras.layers.Dropout):
    def call(self, inputs: tf.Tensor, training: tf.Tensor) -> tf.Tensor:

        to_retain_prob = tf.random.uniform(
            tf.shape(inputs.values), 0, 1, inputs.values.dtype
        )
        to_retain = tf.greater_equal(to_retain_prob, self.rate)
        dropped_inputs = tf.sparse.retain(inputs, to_retain)
        outputs = tf.cond(training, lambda: dropped_inputs, lambda: inputs)
        # noinspection PyProtectedMember
        outputs._dense_shape = inputs._dense_shape

        return outputs


class DenseForSparse(tf.keras.layers.Dense):
    """Dense layer for sparse input tensor"""

    # noinspection PyPep8Naming
    def __init__(self, reg_lambda: float, **kwargs) -> None:
        l1_regularizer = tf.keras.regularizers.l1(reg_lambda)

        super(DenseForSparse, self).__init__(
            kernel_regularizer=l1_regularizer, **kwargs
        )

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


class ReluFfn(tf.keras.layers.Layer):
    """Create feed-forward network with hidden layers and name suffix."""

    def __init__(
        self,
        layer_sizes: List[int],
        droprate: float,
        reg_lambda: float,
        layer_name_suffix: Text,
    ) -> None:
        super(ReluFfn, self).__init__(name=f"ffnn_{layer_name_suffix}")

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._ffn_layers = []
        for i, layer_size in enumerate(layer_sizes):
            self._ffn_layers.append(
                tf.keras.layers.Dense(
                    units=layer_size,
                    activation="relu",
                    kernel_regularizer=l2_regularizer,
                    name=f"hidden_layer_{layer_name_suffix}_{i}",
                )
            )
            self._ffn_layers.append(tf.keras.layers.Dropout(rate=droprate))

    def call(self, x: tf.Tensor, training: tf.Tensor) -> tf.Tensor:
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
        super(Embed, self).__init__(name=f"embed_{layer_name_suffix}")

        self.similarity_type = similarity_type
        if self.similarity_type and self.similarity_type not in {"cosine", "inner"}:
            raise ValueError(
                f"Wrong similarity type '{self.similarity_type}', "
                f"should be 'cosine' or 'inner'"
            )

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._dense = tf.keras.layers.Dense(
            units=embed_dim,
            activation=None,
            kernel_regularizer=l2_regularizer,
            name=f"embed_layer_{layer_name_suffix}",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self._dense(x)
        if self.similarity_type == "cosine":
            x = tf.nn.l2_normalize(x, -1)

        return x


# from https://www.tensorflow.org/tutorials/text/transformer
# and https://github.com/tensorflow/tensor2tensor
# TODO implement relative attention
# TODO save attention weights
class MultiHeadAttention(tf.keras.layers.Layer):
    @staticmethod
    def _scaled_dot_product_attention(q, k, v, pad_mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          pad_mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if pad_mask is not None:
            logits += pad_mask * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(
            logits, axis=-1
        )  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def __init__(self, d_model: int, num_heads: int, reg_lambda: float) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self._depth = d_model // self.num_heads

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._wq = tf.keras.layers.Dense(
            d_model, use_bias=False, kernel_regularizer=l2_regularizer
        )
        self._wk = tf.keras.layers.Dense(
            d_model, use_bias=False, kernel_regularizer=l2_regularizer
        )
        self._wv = tf.keras.layers.Dense(
            d_model, use_bias=False, kernel_regularizer=l2_regularizer
        )
        self._dense = tf.keras.layers.Dense(d_model, kernel_regularizer=l2_regularizer)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)
        """

        x = tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads, self._depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Inverse of split_heads.

        Args:
          x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

        Returns:
          a Tensor with shape [batch, length, channels]
        """

        x = tf.transpose(
            x, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)
        return tf.reshape(
            x, (tf.shape(x)[0], -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

    def call(
        self,
        v: tf.Tensor,
        k: tf.Tensor,
        q: tf.Tensor,
        pad_mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        q = self._wq(q)  # (batch_size, seq_len_q, d_model)
        k = self._wk(k)  # (batch_size, seq_len_k, d_model)
        v = self._wv(v)  # (batch_size, seq_len_v, d_model)

        q = self._split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self._split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        attention, attention_weights = self._scaled_dot_product_attention(
            q, k, v, pad_mask
        )
        # attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        attention = self._combine_heads(attention)  # (batch_size, seq_len_q, d_model)

        output = self._dense(attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        reg_lambda: float,
        rate: float = 0.1,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._mha = MultiHeadAttention(d_model, num_heads, reg_lambda)
        self._dropout = tf.keras.layers.Dropout(rate)

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._ffn_layers = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(
                dff, activation="relu", kernel_regularizer=l2_regularizer
            ),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(
                d_model, kernel_regularizer=l2_regularizer
            ),  # (batch_size, seq_len, d_model)
            tf.keras.layers.Dropout(rate),
        ]

    def call(self, x: tf.Tensor, pad_mask: tf.Tensor, training: tf.Tensor) -> tf.Tensor:

        x_norm = self._layernorm(x)  # (batch_size, seq_len, d_model)
        attn_out, _ = self._mha(x_norm, x_norm, x_norm, pad_mask)
        attn_out = self._dropout(attn_out, training=training)
        x += attn_out

        ffn_out = x  # (batch_size, seq_len, d_model)
        for layer in self._ffn_layers:
            ffn_out = layer(ffn_out, training=training)
        x += ffn_out

        return x  # (batch_size, seq_len, d_model)


class TransformerEncoder(tf.keras.layers.Layer):
    @staticmethod
    def _look_ahead_pad_mask(seq_len: int) -> tf.Tensor:
        pad_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return pad_mask[tf.newaxis, tf.newaxis, :, :]  # (1, 1, seq_len, seq_len)

    @staticmethod
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @classmethod
    def _positional_encoding(cls, position, d_model) -> tf.Tensor:
        angle_rads = cls._get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        max_seq_length: int,
        reg_lambda: float,
        rate: float = 0.1,
        unidirectional: bool = False,
        name: Optional[Text] = None,
    ) -> None:
        super(TransformerEncoder, self).__init__(name=name)

        self.d_model = d_model
        self.unidirectional = unidirectional

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._embedding = tf.keras.layers.Dense(
            units=d_model, kernel_regularizer=l2_regularizer
        )

        self._pos_encoding = self._positional_encoding(max_seq_length, self.d_model)

        self._dropout = tf.keras.layers.Dropout(rate)

        self._enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, reg_lambda, rate)
            for _ in range(num_layers)
        ]
        self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(
        self, x: tf.Tensor, pad_mask: tf.Tensor, training: tf.Tensor
    ) -> tf.Tensor:

        # adding embedding and position encoding.
        x = self._embedding(x)  # (batch_size, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self._pos_encoding[:, : tf.shape(x)[1], :] * (1 - pad_mask)
        x = self._dropout(x, training=training)

        pad_mask = tf.squeeze(pad_mask, -1)  # (batch_size, seq_len)
        pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
        if self.unidirectional:
            # add look ahead pad mask to emulate unidirectional behavior
            pad_mask = tf.minimum(
                1.0, pad_mask + self._look_ahead_pad_mask(tf.shape(pad_mask)[-1])
            )  # (batch_size, 1, seq_len, seq_len)

        for layer in self._enc_layers:
            x = layer(x, pad_mask, training)  # (batch_size, seq_len, d_model)

        # if normalization is done in encoding layers, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        return self._layernorm(x)  # (batch_size, seq_len, d_model)


class InputMask(tf.keras.layers.Layer):
    def build(self, input_shape: "tf.TensorShape") -> None:
        initializer = tf.keras.initializers.GlorotUniform()
        self.mask_vector = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer=initializer,
            trainable=True,
            name="mask_vector",
        )
        self.built = True

    def call(
        self, x: tf.Tensor, mask: tf.Tensor, training: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly mask input sequences."""

        # do not substitute with cls token
        pad_mask_up_to_last = tf.math.cumprod(
            1 - mask, axis=1, exclusive=True, reverse=True
        )
        mask_up_to_last = 1 - pad_mask_up_to_last

        x_random_pad = (
            tf.random.uniform(tf.shape(x), tf.reduce_min(x), tf.reduce_max(x), x.dtype)
            * pad_mask_up_to_last
        )
        # shuffle over batch dim
        x_shuffle = tf.random.shuffle(x * mask_up_to_last + x_random_pad)

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
            other_prob < 0.70, mask_vector, tf.where(other_prob < 0.80, x_shuffle, x)
        )

        lm_mask_prob = tf.random.uniform(tf.shape(mask), 0, 1, mask.dtype) * mask
        lm_mask_bool = tf.greater_equal(lm_mask_prob, 0.85)
        x_masked = tf.where(tf.tile(lm_mask_bool, (1, 1, x.shape[-1])), x_other, x)

        x_masked = tf.cond(training, lambda: x_masked, lambda: x)

        return x_masked, lm_mask_bool


class CRF(tf.keras.layers.Layer):
    def __init__(self, num_tags: int, reg_lambda: float, name: Text = None) -> None:
        super().__init__(name=name)

        initializer = tf.keras.initializers.GlorotUniform()
        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self.transition_params = self.add_weight(
            shape=(num_tags, num_tags),
            initializer=initializer,
            regularizer=l2_regularizer,
            trainable=True,
            name="transitions",
        )

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
        self,
        logits: tf.Tensor,
        tag_indices: tf.Tensor,
        sequence_lengths: tf.Tensor,
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
    ) -> None:
        super().__init__(name=name)
        self.num_neg = num_neg
        self.loss_type = loss_type
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        self.use_max_sim_neg = use_max_sim_neg
        self.neg_lambda = neg_lambda
        self.scale_loss = scale_loss

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

        # return tf.tile(rand_idxs(), (batch_size, 1))

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
            parallel_iterations=1000,
            back_prop=False,
        )[1]

    @staticmethod
    def _sample_idxs(
        batch_size: tf.Tensor, x: tf.Tensor, idxs: tf.Tensor
    ) -> tf.Tensor:
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
    ) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
    ]:
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
    def sim(
        a: tf.Tensor, b: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Calculate similarity between given tensors."""

        sim = tf.reduce_sum(a * b, -1)
        if mask is not None:
            sim *= tf.expand_dims(mask, 2)

        return sim

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
        """Calculate accuracy"""

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
