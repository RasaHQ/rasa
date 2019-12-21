import logging
import typing
from typing import (
    List,
    Optional,
    Text,
    Dict,
    Tuple,
    Union,
    Generator,
    Callable,
    Any,
    NamedTuple,
)
import tensorflow as tf
import numpy as np

if typing.TYPE_CHECKING:
    from tensor2tensor.utils.hparam import HParams

logger = logging.getLogger(__name__)


class SparseDropout(tf.keras.layers.Dropout):

    def call(self, inputs, training):
        if training is None:
            training = tf.keras.backend.learning_phase()

        to_retain_prob = tf.random.uniform(
            tf.shape(inputs.values), 0, 1, inputs.values.dtype
        )
        to_retain = tf.greater_equal(to_retain_prob, self.rate)
        dropped_inputs = tf.sparse.retain(inputs, to_retain)
        outputs = tf.cond(training, lambda: dropped_inputs, lambda: inputs)
        outputs._dense_shape = inputs._dense_shape

        return outputs


class DenseForSparse(tf.keras.layers.Dense):
    """Dense layer for sparse input tensor"""

    # noinspection PyPep8Naming
    def __init__(self,
                 C2: float,
                 activation: Optional[Callable] = tf.nn.relu,
                 **kwargs):
        kernel_regularizer = tf.keras.regularizers.l1(C2)

        super(DenseForSparse, self).__init__(kernel_regularizer=kernel_regularizer,
                                             activation=activation,
                                             **kwargs)

    def call(self, inputs):
        if not isinstance(inputs, tf.SparseTensor):
            raise ValueError("Input tensor should be sparse.")

        # outputs will be 2D
        outputs = tf.sparse.sparse_dense_matmul(tf.sparse.reshape(inputs, [-1, tf.shape(inputs)[-1]]), self.kernel)

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


class Ffnn(tf.keras.layers.Layer):
    """Create feed-forward nn with hidden layers and name suffix."""

    # noinspection PyPep8Naming
    def __init__(
        self,
        layer_sizes: List[int],
        droprate: float,
        C2: float,
        layer_name_suffix: Text,
        activation: Optional[Callable] = tf.nn.relu,
        use_bias: bool = True,
        kernel_initializer: Optional["tf.keras.initializers.Initializer"] = None,
    ):
        super(Ffnn, self).__init__(name=f"ffnn_{layer_name_suffix}")

        self._layers = []
        for i, layer_size in enumerate(layer_sizes):
            self._layers.append(tf.keras.layers.Dense(
                units=layer_size,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(C2),
                name=f"hidden_layer_{layer_name_suffix}_{i}",
            ))
            self._layers.append(tf.keras.layers.Dropout(rate=droprate))

    def call(self, inputs, training):
        x = inputs
        for layer in self._layers:
            x = layer(x, training=training)

        return x


class Embed(tf.keras.layers.Layer):
    """Create dense embedding layer with a name."""

    # noinspection PyPep8Naming
    def __init__(
            self,
            embed_dim: int,
            C2: float,
            layer_name_suffix: Text,
            similarity_type: Optional[Text] = None,
    ):
        super(Embed, self).__init__(name=f"embed_{layer_name_suffix}")

        self.similarity_type = similarity_type
        if self.similarity_type and self.similarity_type not in {"cosine", "inner"}:
            raise ValueError(
                f"Wrong similarity type '{self.similarity_type}', "
                f"should be 'cosine' or 'inner'"
            )

        self._layers = [tf.keras.layers.Dense(
            units=embed_dim,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(C2),
            name=f"embed_layer_{layer_name_suffix}",
        )]

    def call(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        if self.similarity_type == "cosine":
            x = tf.nn.l2_normalize(x, -1)

        return x


# from https://www.tensorflow.org/tutorials/text/transformer
# and https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/transformer_layers.py#L137
# TODO add weight regularization (L1)
# TODO collect losses
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
            logits += (pad_mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False)

        self.dense = tf.keras.layers.Dense(d_model, use_bias=False)

    def _split_heads(self, x):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """

        x = tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        """Inverse of split_heads.

        Args:
          x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

        Returns:
          a Tensor with shape [batch, length, channels]
        """

        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        return tf.reshape(x, (tf.shape(x)[0], -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    def call(self, v, k, q, pad_mask=None):
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self._split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self._split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        attention, attention_weights = self._scaled_dot_product_attention(q, k, v, pad_mask)
        # attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        attention = self._combine_heads(attention)  # (batch_size, seq_len_q, d_model)

        output = self.dense(attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# TODO add weight regularization (L2)
# TODO collect losses
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout = tf.keras.layers.Dropout(rate)

        self.ffn_layers = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
            tf.keras.layers.Dropout(rate),
        ]

    def call(self, x, pad_mask, training):

        x_norm = self.layernorm(x)  # (batch_size, input_seq_len, d_model)
        attn, _ = self.mha(x_norm, x_norm, x_norm, pad_mask)  # (batch_size, input_seq_len, d_model)
        attn = self.dropout(attn, training=training)
        x += attn

        ffn = x
        for layer in self.ffn_layers:
            ffn = layer(ffn, training=training)  # (batch_size, input_seq_len, d_model)
        x += ffn

        return x


# TODO collect losses
class TransformerEncoder(tf.keras.layers.Layer):

    @staticmethod
    def _look_ahead_pad_mask(size):
        pad_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return pad_mask[tf.newaxis, tf.newaxis, :, :]   # (1, 1, seq_len, seq_len)

    @staticmethod
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @classmethod
    def _positional_encoding(cls, position, d_model):
        angle_rads = cls._get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def __init__(self, num_layers, d_model, num_heads, dff,
                 max_seq_length, rate=0.1, unidirectional=False):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.unidirectional = unidirectional

        # TODO use Embed
        self.embedding = tf.keras.layers.Dense(units=d_model, use_bias=False)
        self.pos_encoding = self._positional_encoding(max_seq_length,
                                                      self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, pad_mask, training):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :] * (1 - pad_mask)
        x = self.dropout(x, training=training)

        pad_mask = tf.squeeze(pad_mask, -1)
        pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
        if self.unidirectional:
            pad_mask = tf.minimum(
                1.0, pad_mask + self._look_ahead_pad_mask(tf.shape(pad_mask)[-1])
            )  # (batch_size, 1, seq_len, seq_len)

        for layer in self.enc_layers:
            x = layer(x, pad_mask, training)

        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        return self.layernorm(x)  # (batch_size, input_seq_len, d_model)
