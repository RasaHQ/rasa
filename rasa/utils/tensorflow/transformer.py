from typing import List, Optional, Text, Tuple, Callable
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.utils import tf_utils
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
        use_key_relative_position: bool = False,
        use_value_relative_position: bool = False,
        max_relative_position: Optional[int] = None,
        heads_share_relative_embedding: bool = False,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.attention_dropout_rate = attention_dropout_rate
        self.unidirectional = unidirectional
        self.use_key_relative_position = use_key_relative_position
        self.use_value_relative_position = use_value_relative_position
        self.max_relative_position = max_relative_position
        self.heads_share_relative_embedding = heads_share_relative_embedding

        assert d_model % self.num_heads == 0

        self._depth = d_model // self.num_heads

        self._wq = DenseWithSparseWeights(units=d_model, use_bias=False)
        self._wk = DenseWithSparseWeights(units=d_model, use_bias=False)
        self._wv = DenseWithSparseWeights(units=d_model, use_bias=False)

        self._dense = DenseWithSparseWeights(units=d_model)

        self._add_relative_embeddings()

    def _add_relative_embeddings(self) -> None:
        """Instantiate relative embeddings."""

        if self.use_key_relative_position or self.use_value_relative_position:
            if not self.max_relative_position:
                raise ValueError(
                    f"Max relative position {self.max_relative_position} "
                    f"should be > 0 when using relative attention."
                )

            if self.unidirectional:
                max_relative_position_unmasked = self.max_relative_position
            else:
                max_relative_position_unmasked = 2 * self.max_relative_position - 1

            if self.heads_share_relative_embedding:
                relative_embedding_shape = (max_relative_position_unmasked, self._depth)
            else:
                relative_embedding_shape = (
                    self.num_heads,
                    max_relative_position_unmasked,
                    self._depth,
                )
        else:
            relative_embedding_shape = None

        if self.use_key_relative_position:
            self.key_relative_embeddings = self.add_weight(
                shape=relative_embedding_shape, name="key_relative_embeddings",
            )
        else:
            self.key_relative_embeddings = None

        if self.use_value_relative_position:
            self.value_relative_embeddings = self.add_weight(
                shape=relative_embedding_shape, name="value_relative_embeddings",
            )
        else:
            self.value_relative_embeddings = None

    def _pad_relative_embeddings(self, x: tf.Tensor, length: tf.Tensor) -> tf.Tensor:
        # pad the left side to length
        pad_left = x[:, :, :, :1, :]
        pad_left = tf.tile(pad_left, (1, 1, 1, length - self.max_relative_position, 1))

        # pad the right side to length
        if self.unidirectional:
            m_right = 1  # current time
            pad_right = tf.zeros_like(x[:, :, :, -1:, :])
        else:
            m_right = self.max_relative_position
            pad_right = x[:, :, :, -1:, :]
        pad_right = tf.tile(pad_right, (1, 1, 1, length - m_right, 1))

        return tf.concat([pad_left, x, pad_right], axis=-2)

    def _slice_relative_embeddings(self, x: tf.Tensor, length: tf.Tensor) -> tf.Tensor:
        if self.unidirectional:
            # pad the right side to length
            pad_right = tf.zeros_like(x[:, :, :, -1:, :])
            pad_right = tf.tile(pad_right, (1, 1, 1, length - 1, 1))
            x = tf.concat([x, pad_right], axis=-2)

        dl = self.max_relative_position - length
        m = tf.shape(x)[-2]
        return x[:, :, :, dl : m - dl, :]

    def _relative_to_absolute_position(self, x: tf.Tensor) -> tf.Tensor:
        """Universal method to convert tensor from relative to absolute indexing.

        x.shape =
        (batch, num_heads, length, relative_length, depth)
        or (batch, num_heads, length, relative_length)
        "Slides" relative embeddings by 45 degree """

        x_dim = len(x.shape)

        if x_dim < 4 or x_dim > 5:
            raise ValueError("Relative tensor has a wrong shape.")
        if x_dim == 4:
            # add fake depth dimension
            x = tf.expand_dims(x, axis=-1)

        batch = tf.shape(x)[0]
        num_heads = tf.shape(x)[1]
        length = tf.shape(x)[2]
        depth = tf.shape(x)[-1]

        x = tf.cond(
            length > self.max_relative_position,
            lambda: self._pad_relative_embeddings(x, length),
            lambda: self._slice_relative_embeddings(x, length),
        )

        # add a column of zeros to "slide" columns to diagonals through reshape
        pad_shift = tf.zeros_like(x[:, :, :, -1:, :])
        x = tf.concat([x, pad_shift], axis=-2)

        # flatten length dimensions
        x = tf.reshape(x, (batch, num_heads, -1, depth))
        width = 2 * length

        # add zeros so that the result of back reshape is still a matrix
        pad_flat = tf.zeros_like(
            x[:, :, : (width - 1) - width * length % (width - 1), :]
        )
        x = tf.concat([x, pad_flat], axis=-2)

        # "slide" columns to diagonals through reshape
        x = tf.reshape(x, (batch, num_heads, -1, width - 1, depth))

        # slice needed "diagonal" matrix
        x = x[:, :, :-1, -length:, :]

        if x_dim == 4:
            # remove fake depth dimension
            x = tf.squeeze(x, axis=-1)

        return x

    def _matmul_with_relative_keys(self, x: tf.Tensor) -> tf.Tensor:
        y = self.key_relative_embeddings

        if self.heads_share_relative_embedding:
            matmul = tf.einsum("bhld,md->bhlm", x, y)
        else:
            matmul = tf.einsum("bhld,hmd->bhlm", x, y)

        return self._relative_to_absolute_position(matmul)

    def _tile_relative_embeddings(self, x: tf.Tensor, length: tf.Tensor) -> tf.Tensor:
        if self.heads_share_relative_embedding:
            x = tf.expand_dims(x, axis=0)  # add head dimension

        x = tf.expand_dims(x, axis=1)  # add length dimension
        x = tf.tile(x, (1, length, 1, 1))
        return tf.expand_dims(x, axis=0)  # add batch dimension

    def _squeeze_relative_embeddings(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.squeeze(x, axis=0)  # squeeze batch dimension
        if self.heads_share_relative_embedding:
            x = tf.squeeze(x, axis=1)  # squeeze head dimension
        return x

    def _matmul_with_relative_values(self, x: tf.Tensor) -> tf.Tensor:
        y = self._tile_relative_embeddings(
            self.value_relative_embeddings, tf.shape(x)[-2]
        )
        y = self._relative_to_absolute_position(y)
        y = self._squeeze_relative_embeddings(y)

        if self.heads_share_relative_embedding:
            return tf.einsum("bhlm,lmd->bhld", x, y)
        else:
            return tf.einsum("bhlm,hlmd->bhld", x, y)

    def _drop_attention_logits(
        self, logits: tf.Tensor, pad_mask: tf.Tensor, training: tf.Tensor
    ) -> tf.Tensor:
        def droped_logits() -> tf.Tensor:
            keep_prob = tf.random.uniform(tf.shape(logits), 0, 1) + pad_mask
            drop_mask = tf.cast(
                tf.less(keep_prob, self.attention_dropout_rate), logits.dtype
            )

            return logits + drop_mask * -1e9

        return tf_utils.smart_cond(training, droped_logits, lambda: tf.identity(logits))

    def _scaled_dot_product_attention(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        pad_mask: tf.Tensor,
        training: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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

        if self.use_key_relative_position:
            matmul_qk += self._matmul_with_relative_keys(q)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if pad_mask is not None:
            logits += pad_mask * -1e9

        # apply attention dropout before softmax to maintain attention_weights norm as 1
        if self.attention_dropout_rate > 0:
            logits = self._drop_attention_logits(logits, pad_mask, training)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(
            logits, axis=-1
        )  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        if self.use_value_relative_position:
            output += self._matmul_with_relative_values(attention_weights)

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
        use_key_relative_position: bool = False,
        use_value_relative_position: bool = False,
        max_relative_position: Optional[int] = None,
        heads_share_relative_embedding: bool = False,
    ) -> None:
        super().__init__()

        self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._mha = MultiHeadAttention(
            d_model,
            num_heads,
            attention_dropout_rate,
            unidirectional,
            use_key_relative_position,
            use_value_relative_position,
            max_relative_position,
            heads_share_relative_embedding,
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
        use_key_relative_position: bool = False,
        use_value_relative_position: bool = False,
        max_relative_position: Optional[int] = None,
        heads_share_relative_embedding: bool = False,
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
                use_key_relative_position,
                use_value_relative_position,
                max_relative_position,
                heads_share_relative_embedding,
            )
            for _ in range(num_layers)
        ]
        self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

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
