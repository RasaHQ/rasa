from typing import List, Optional, Text, Tuple, Callable
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from rasa.utils.tensorflow.layers import DenseWithSparseWeights


# from https://www.tensorflow.org/tutorials/text/transformer
# and https://github.com/tensorflow/tensor2tensor
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_dropout_rate: float = 0.0,
        unidirectional: bool = False,
        use_relative_position: bool = False,
        max_relative_position: Optional[int] = None,
        heads_share_relative_embedding: bool = False,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.unidirectional = unidirectional
        self.use_relative_position = use_relative_position
        self.max_relative_position = max_relative_position

        assert d_model % self.num_heads == 0

        self._depth = d_model // self.num_heads

        self._wq = DenseWithSparseWeights(units=d_model, use_bias=False)
        self._wk = DenseWithSparseWeights(units=d_model, use_bias=False)
        self._wv = DenseWithSparseWeights(units=d_model, use_bias=False)

        if use_relative_position:
            if not max_relative_position:
                raise ValueError(
                    f"Max relative position {max_relative_position} "
                    f"should be > 0 when using relative attention."
                )

            if unidirectional:
                max_relative_position_unmasked = max_relative_position
            else:
                max_relative_position_unmasked = 2 * max_relative_position - 1

            if heads_share_relative_embedding:
                relative_embedding_shape = (max_relative_position_unmasked, self._depth)
            else:
                relative_embedding_shape = (
                    num_heads,
                    max_relative_position_unmasked,
                    self._depth,
                )

            initializer = tf.keras.initializers.TruncatedNormal(
                stddev=self._depth ** -0.5
            )
            self.key_relative_embeddings = self.add_weight(
                shape=relative_embedding_shape,
                initializer=initializer,
                trainable=True,
                name="key_relative_embeddings",
            )
            self.value_relative_embeddings = self.add_weight(
                shape=relative_embedding_shape,
                initializer=initializer,
                trainable=True,
                name="value_relative_embeddings",
            )
        else:
            self.key_relative_embeddings = None
            self.value_relative_embeddings = None

        self._attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate)

        self._dense = DenseWithSparseWeights(units=d_model)

    def _scaled_dot_product_attention(self, q, k, v, pad_mask, training):
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

        # TODO add key relative embeddings

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

        attention_weights = self._attention_dropout(
            attention_weights, training=training
        )

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        # TODO add value relative embedding to values

        return output, attention_weights

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
        pad_mask: Optional[tf.Tensor],
        training: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        q = self._wq(q)  # (batch_size, seq_len_q, d_model)
        k = self._wk(k)  # (batch_size, seq_len_k, d_model)
        v = self._wv(v)  # (batch_size, seq_len_v, d_model)

        q = self._split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self._split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        attention, attention_weights = self._scaled_dot_product_attention(
            q, k, v, pad_mask, training
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
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        unidirectional: bool = False,
    ) -> None:
        super().__init__()

        self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._mha = MultiHeadAttention(
            d_model, num_heads, attention_dropout_rate, unidirectional
        )
        self._dropout = tf.keras.layers.Dropout(dropout_rate)

        self._ffn_layers = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            DenseWithSparseWeights(
                units=dff, activation=tfa.activations.gelu
            ),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dropout(dropout_rate),
            DenseWithSparseWeights(units=d_model),  # (batch_size, seq_len, d_model)
            tf.keras.layers.Dropout(dropout_rate),
        ]

    def call(self, x: tf.Tensor, pad_mask: tf.Tensor, training: tf.Tensor) -> tf.Tensor:
        x_norm = self._layernorm(x)  # (batch_size, seq_len, d_model)
        attn_out, _ = self._mha(x_norm, x_norm, x_norm, pad_mask, training=training)
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
    def _get_angles(pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        angle_dropout_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_dropout_rates

    @classmethod
    def _positional_encoding(cls, max_position: int, d_model: int) -> tf.Tensor:
        angle_rads = cls._get_angles(
            np.arange(max_position)[:, np.newaxis],
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
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        unidirectional: bool = False,
        name: Optional[Text] = None,
    ) -> None:
        super().__init__(name=name)

        self.d_model = d_model
        self.unidirectional = unidirectional

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._embedding = DenseWithSparseWeights(
            units=d_model, kernel_regularizer=l2_regularizer
        )

        self._pos_encoding = self._positional_encoding(max_seq_length, self.d_model)

        self._dropout = tf.keras.layers.Dropout(dropout_rate)

        self._enc_layers = [
            TransformerEncoderLayer(
                d_model,
                num_heads,
                dff,
                dropout_rate,
                attention_dropout_rate,
                unidirectional,
            )
            for _ in range(num_layers)
        ]
        self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, pad_mask: tf.Tensor, training: tf.Tensor) -> tf.Tensor:

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
