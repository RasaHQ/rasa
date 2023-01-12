from typing import Optional, Text, Tuple, Union

import numpy as np
import tensorflow as tf

# TODO: The following is not (yet) available via tf.keras
from keras.utils.control_flow_util import smart_cond
from tensorflow.keras import backend as K

import rasa.shared.utils.cli
from rasa.utils.tensorflow.layers import RandomlyConnectedDense


# from https://www.tensorflow.org/tutorials/text/transformer
# and https://github.com/tensorflow/tensor2tensor
class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-headed attention layer.

    Arguments:
        units: Positive integer, output dim of hidden layer.
        num_heads: Positive integer, number of heads
            to repeat the same attention structure.
        attention_dropout_rate: Float, dropout rate inside attention for training.
        density: Approximate fraction of trainable weights (in
            `RandomlyConnectedDense` layers).
        unidirectional: Boolean, use a unidirectional or bidirectional encoder.
        use_key_relative_position: Boolean, if 'True' use key
            relative embeddings in attention.
        use_value_relative_position: Boolean, if 'True' use value
            relative embeddings in attention.
        max_relative_position: Positive integer, max position for relative embeddings.
        heads_share_relative_embedding: Boolean, if 'True'
            heads will share relative embeddings.
    """

    def __init__(
        self,
        units: int,
        num_heads: int,
        attention_dropout_rate: float = 0.0,
        density: float = 0.2,
        unidirectional: bool = False,
        use_key_relative_position: bool = False,
        use_value_relative_position: bool = False,
        max_relative_position: int = 5,
        heads_share_relative_embedding: bool = False,
    ) -> None:
        super().__init__()

        if units % num_heads != 0:
            rasa.shared.utils.cli.print_error_and_exit(
                f"Value Error: The given transformer size {units} should be a "
                f"multiple of the number of attention heads {num_heads}."
            )

        self.num_heads = num_heads
        self.units = units
        self.attention_dropout_rate = attention_dropout_rate
        self.unidirectional = unidirectional
        self.use_key_relative_position = use_key_relative_position
        self.use_value_relative_position = use_value_relative_position
        self.relative_length = max_relative_position
        self.relative_length += 1  # include current time
        self.heads_share_relative_embedding = heads_share_relative_embedding

        self._depth = units // self.num_heads

        # process queries
        self._query_dense_layer = RandomlyConnectedDense(
            units=units, use_bias=False, density=density
        )
        # process keys
        self._key_dense_layer = RandomlyConnectedDense(
            units=units, use_bias=False, density=density
        )
        # process values
        self._value_dense_layer = RandomlyConnectedDense(
            units=units, use_bias=False, density=density
        )
        # process attention output
        self._output_dense_layer = RandomlyConnectedDense(units=units, density=density)

        self._create_relative_embeddings()

    def _create_relative_embeddings(self) -> None:
        """Create relative embeddings."""
        relative_embedding_shape: Optional[
            Union[Tuple[int, int], Tuple[int, int, int]]
        ] = None
        self.key_relative_embeddings = None
        self.value_relative_embeddings = None

        if self.use_key_relative_position or self.use_value_relative_position:
            if not self.relative_length:
                raise ValueError(
                    f"Max relative position {self.relative_length} "
                    f"should be > 0 when using relative attention."
                )

            if self.unidirectional:
                relative_length = self.relative_length
            else:
                relative_length = 2 * self.relative_length - 1

            if self.heads_share_relative_embedding:
                relative_embedding_shape = (relative_length, self._depth)
            else:
                relative_embedding_shape = (
                    self.num_heads,
                    relative_length,
                    self._depth,
                )

        if self.use_key_relative_position:
            self.key_relative_embeddings = self.add_weight(
                shape=relative_embedding_shape, name="key_relative_embeddings"
            )

        if self.use_value_relative_position:
            self.value_relative_embeddings = self.add_weight(
                shape=relative_embedding_shape, name="value_relative_embeddings"
            )

    def _pad_relative_embeddings(self, x: tf.Tensor, length: tf.Tensor) -> tf.Tensor:
        # pad the left side to length
        pad_left = x[:, :, :, :1, :]
        pad_left = tf.tile(pad_left, (1, 1, 1, length - self.relative_length, 1))

        # pad the right side to length
        if self.unidirectional:
            right_relative_length = 1  # current time
            pad_right = tf.zeros_like(x[:, :, :, -1:, :])
        else:
            right_relative_length = self.relative_length
            pad_right = x[:, :, :, -1:, :]
        pad_right = tf.tile(pad_right, (1, 1, 1, length - right_relative_length, 1))

        return tf.concat([pad_left, x, pad_right], axis=-2)

    def _slice_relative_embeddings(self, x: tf.Tensor, length: tf.Tensor) -> tf.Tensor:
        if self.unidirectional:
            # pad the right side to relative_length
            pad_right = tf.zeros_like(x[:, :, :, -1:, :])
            pad_right = tf.tile(pad_right, (1, 1, 1, self.relative_length - 1, 1))
            x = tf.concat([x, pad_right], axis=-2)

        extra_length = self.relative_length - length
        full_length = tf.shape(x)[-2]
        return x[:, :, :, extra_length : full_length - extra_length, :]

    def _relative_to_absolute_position(self, x: tf.Tensor) -> tf.Tensor:
        """Universal method to convert tensor from relative to absolute indexing.

        "Slides" relative embeddings by 45 degree.

        Arguments:
        x: A tensor of shape (batch, num_heads, length, relative_length, depth)
            or (batch, num_heads, length, relative_length)

        Returns:
            A tensor of shape (batch, num_heads, length, length, depth)
            or (batch, num_heads, length, length)
        """

        x_dim = len(x.shape)

        if x_dim < 4 or x_dim > 5:
            raise ValueError(
                f"Relative tensor has a wrong shape {x.shape}, "
                f"it should have 4 or 5 dimensions."
            )
        if x_dim == 4:
            # add fake depth dimension
            x = tf.expand_dims(x, axis=-1)

        batch = tf.shape(x)[0]
        num_heads = tf.shape(x)[1]
        length = tf.shape(x)[2]
        depth = tf.shape(x)[-1]

        x = tf.cond(
            length > self.relative_length,
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
            x[:, :, : ((width - 1) - width * length % (width - 1)) % (width - 1), :]
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

        return smart_cond(training, droped_logits, lambda: tf.identity(logits))

    def _scaled_dot_product_attention(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        pad_mask: tf.Tensor,
        training: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate the attention weights.

        query, key, value must have matching leading dimensions.
        key, value must have matching penultimate dimension,
        i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type (padding or look ahead)
        but it must be broadcastable for addition.

        Arguments:
            query: A tensor with shape (..., length, depth).
            key: A tensor with shape (..., length, depth).
            value: A tensor with shape (..., length, depth).
            pad_mask: Float tensor with shape broadcastable
                to (..., length, length). Defaults to None.

        Returns:
            output: A tensor with shape (..., length, depth).
            attention_weights: A tensor with shape (..., length, length).
        """

        matmul_qk = tf.matmul(query, key, transpose_b=True)  # (..., length, length)

        if self.use_key_relative_position:
            matmul_qk += self._matmul_with_relative_keys(query)

        # scale matmul_qk
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if pad_mask is not None:
            logits += pad_mask * -1e9

        # apply attention dropout before softmax to maintain attention_weights norm as 1
        if self.attention_dropout_rate > 0:
            logits = self._drop_attention_logits(logits, pad_mask, training)

        # softmax is normalized on the last axis (length) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(logits, axis=-1)  # (..., length, length)

        output = tf.matmul(attention_weights, value)  # (..., length, depth)
        if self.use_value_relative_position:
            output += self._matmul_with_relative_values(attention_weights)

        return output, attention_weights

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is
        (batch_size, num_heads, length, depth)
        """

        x = tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads, self._depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Inverse of split_heads.

        Args:
            x: A Tensor with shape [batch, num_heads, length, units / num_heads]

        Returns:
            A Tensor with shape [batch, length, units]
        """

        # (batch_size, length, num_heads, depth)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # (batch_size, length, units)
        return tf.reshape(x, (tf.shape(x)[0], -1, self.units))

    # noinspection PyMethodOverriding
    def call(
        self,
        query_input: tf.Tensor,
        source_input: tf.Tensor,
        pad_mask: Optional[tf.Tensor] = None,
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply attention mechanism to query_input and source_input.

        Arguments:
            query_input: A tensor with shape [batch_size, length, input_size].
            source_input: A tensor with shape [batch_size, length, input_size].
            pad_mask: Float tensor with shape broadcastable
                to (..., length, length). Defaults to None.
            training: A bool, whether in training mode or not.

        Returns:
            Attention layer output with shape [batch_size, length, units]
        """
        if training is None:
            training = K.learning_phase()

        query = self._query_dense_layer(query_input)  # (batch_size, length, units)
        key = self._key_dense_layer(source_input)  # (batch_size, length, units)
        value = self._value_dense_layer(source_input)  # (batch_size, length, units)

        query = self._split_heads(query)  # (batch_size, num_heads, length, depth)
        key = self._split_heads(key)  # (batch_size, num_heads, length, depth)
        value = self._split_heads(value)  # (batch_size, num_heads, length, depth)

        attention, attention_weights = self._scaled_dot_product_attention(
            query, key, value, pad_mask, training
        )
        # attention.shape == (batch_size, num_heads, length, depth)
        # attention_weights.shape == (batch_size, num_heads, length, length)
        attention = self._combine_heads(attention)  # (batch_size, length, units)

        output = self._output_dense_layer(attention)  # (batch_size, length, units)

        return output, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Transformer encoder layer.

    The layer is composed of the sublayers:
        1. Self-attention layer
        2. Feed-forward network (which is 2 fully-connected layers)

    Arguments:
        units: Positive integer, output dim of hidden layer.
        num_heads: Positive integer, number of heads
            to repeat the same attention structure.
        filter_units: Positive integer, output dim of the first ffn hidden layer.
        dropout_rate: Float between 0 and 1; fraction of the input units to drop.
        attention_dropout_rate: Float, dropout rate inside attention for training.
        density: Fraction of trainable weights in `RandomlyConnectedDense` layers.
        unidirectional: Boolean, use a unidirectional or bidirectional encoder.
        use_key_relative_position: Boolean, if 'True' use key
            relative embeddings in attention.
        use_value_relative_position: Boolean, if 'True' use value
            relative embeddings in attention.
        max_relative_position: Positive integer, max position for relative embeddings.
        heads_share_relative_embedding: Boolean, if 'True'
            heads will share relative embeddings.
    """

    def __init__(
        self,
        units: int,
        num_heads: int,
        filter_units: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        density: float = 0.2,
        unidirectional: bool = False,
        use_key_relative_position: bool = False,
        use_value_relative_position: bool = False,
        max_relative_position: int = 5,
        heads_share_relative_embedding: bool = False,
    ) -> None:
        super().__init__()

        self._layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._mha = MultiHeadAttention(
            units,
            num_heads,
            attention_dropout_rate,
            density,
            unidirectional,
            use_key_relative_position,
            use_value_relative_position,
            max_relative_position,
            heads_share_relative_embedding,
        )
        self._dropout = tf.keras.layers.Dropout(dropout_rate)

        self._ffn_layers = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            RandomlyConnectedDense(
                units=filter_units, activation=tf.nn.gelu, density=density
            ),  # (batch_size, length, filter_units)
            tf.keras.layers.Dropout(dropout_rate),
            RandomlyConnectedDense(
                units=units, density=density
            ),  # (batch_size, length, units)
            tf.keras.layers.Dropout(dropout_rate),
        ]

    def call(
        self,
        x: tf.Tensor,
        pad_mask: Optional[tf.Tensor] = None,
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply transformer encoder layer.

        Arguments:
            x: A tensor with shape [batch_size, length, units].
            pad_mask: Float tensor with shape broadcastable
                to (..., length, length). Defaults to None.
            training: A bool, whether in training mode or not.

        Returns:
            Transformer encoder layer output with shape [batch_size, length, units]
        """
        if training is None:
            training = K.learning_phase()

        x_norm = self._layer_norm(x)  # (batch_size, length, units)
        attn_out, attn_weights = self._mha(
            x_norm, x_norm, pad_mask=pad_mask, training=training
        )
        attn_out = self._dropout(attn_out, training=training)
        x += attn_out

        ffn_out = x  # (batch_size, length, units)
        for layer in self._ffn_layers:
            ffn_out = layer(ffn_out, training=training)
        x += ffn_out

        # (batch_size, length, units), (batch_size, num_heads, length, length)
        return x, attn_weights


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder.

    Encoder stack is made up of `num_layers` identical encoder layers.

    Arguments:
        num_layers: Positive integer, number of encoder layers.
        units: Positive integer, output dim of hidden layer.
        num_heads: Positive integer, number of heads
            to repeat the same attention structure.
        filter_units: Positive integer, output dim of the first ffn hidden layer.
        reg_lambda: Float, regularization factor.
        dropout_rate: Float between 0 and 1; fraction of the input units to drop.
        attention_dropout_rate: Float, dropout rate inside attention for training.
        density: Approximate fraction of trainable weights (in
            `RandomlyConnectedDense` layers).
        unidirectional: Boolean, use a unidirectional or bidirectional encoder.
        use_key_relative_position: Boolean, if 'True' use key
            relative embeddings in attention.
        use_value_relative_position: Boolean, if 'True' use value
            relative embeddings in attention.
        max_relative_position: Positive integer, max position for relative embeddings.
        heads_share_relative_embedding: Boolean, if 'True'
            heads will share relative embeddings.
        name: Optional name of the layer.
    """

    def __init__(
        self,
        num_layers: int,
        units: int,
        num_heads: int,
        filter_units: int,
        reg_lambda: float,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        density: float = 0.2,
        unidirectional: bool = False,
        use_key_relative_position: bool = False,
        use_value_relative_position: bool = False,
        max_relative_position: int = 5,
        heads_share_relative_embedding: bool = False,
        name: Optional[Text] = None,
    ) -> None:
        super().__init__(name=name)

        self.units = units
        self.unidirectional = unidirectional

        l2_regularizer = tf.keras.regularizers.l2(reg_lambda)
        self._embedding = RandomlyConnectedDense(
            units=units, kernel_regularizer=l2_regularizer, density=density
        )
        # positional encoding helpers
        self._angles = self._get_angles()
        self._even_indices = np.arange(0, self.units, 2, dtype=np.int32)[:, np.newaxis]
        self._odd_indices = np.arange(1, self.units, 2, dtype=np.int32)[:, np.newaxis]

        self._dropout = tf.keras.layers.Dropout(dropout_rate)

        self._enc_layers = [
            TransformerEncoderLayer(
                units,
                num_heads,
                filter_units,
                dropout_rate,
                attention_dropout_rate,
                density,
                unidirectional,
                use_key_relative_position,
                use_value_relative_position,
                max_relative_position,
                heads_share_relative_embedding,
            )
            for _ in range(num_layers)
        ]
        self._layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def _get_angles(self) -> np.ndarray:
        array_2d = np.arange(self.units)[np.newaxis, :]
        return 1 / np.power(10000, (2 * (array_2d // 2)) / np.float32(self.units))

    def _positional_encoding(self, max_position: tf.Tensor) -> tf.Tensor:
        max_position = tf.cast(max_position, dtype=tf.float32)
        angle_rads = tf.range(max_position)[:, tf.newaxis] * self._angles

        # transpose for easy slicing
        angle_rads = tf.transpose(angle_rads, perm=[1, 0])
        shape = tf.shape(angle_rads)
        # apply sin to even indices in the array; 2i
        sin_even = tf.sin(tf.gather_nd(angle_rads, self._even_indices))
        pos_encoding_even = tf.scatter_nd(self._even_indices, sin_even, shape)
        # apply cos to odd indices in the array; 2i+1
        cos_odd = tf.cos(tf.gather_nd(angle_rads, self._odd_indices))
        pos_encoding_odd = tf.scatter_nd(self._odd_indices, cos_odd, shape)
        # combine even and odd positions and transpose back
        pos_encoding = tf.transpose(pos_encoding_even + pos_encoding_odd, perm=[1, 0])
        # add batch dimension
        return tf.stop_gradient(pos_encoding[tf.newaxis, ...])

    @staticmethod
    def _look_ahead_pad_mask(max_position: tf.Tensor) -> tf.Tensor:
        pad_mask = 1 - tf.linalg.band_part(tf.ones((max_position, max_position)), -1, 0)
        return pad_mask[tf.newaxis, tf.newaxis, :, :]  # (1, 1, seq_len, seq_len)

    def call(
        self,
        x: tf.Tensor,
        pad_mask: Optional[tf.Tensor] = None,
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply transformer encoder.

        Arguments:
            x: A tensor with shape [batch_size, length, input_size].
            pad_mask: Float tensor with shape broadcastable
                to (..., length, length). Defaults to None.
            training: A bool, whether in training mode or not.

        Returns:
            Transformer encoder output with shape [batch_size, length, units]
        """
        # adding embedding and position encoding.
        x = self._embedding(x)  # (batch_size, length, units)
        x *= tf.math.sqrt(tf.cast(self.units, tf.float32))
        x += self._positional_encoding(tf.shape(x)[1])
        x = self._dropout(x, training=training)

        if pad_mask is not None:
            pad_mask = tf.squeeze(pad_mask, -1)  # (batch_size, length)
            pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
            # pad_mask.shape = (batch_size, 1, 1, length)
            if self.unidirectional:
                # add look ahead pad mask to emulate unidirectional behavior
                pad_mask = tf.minimum(
                    1.0, pad_mask + self._look_ahead_pad_mask(tf.shape(pad_mask)[-1])
                )  # (batch_size, 1, length, length)

        layer_attention_weights = []

        for layer in self._enc_layers:
            x, attn_weights = layer(x, pad_mask=pad_mask, training=training)
            layer_attention_weights.append(attn_weights)

        # if normalization is done in encoding layers, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        x = self._layer_norm(x)  # (batch_size, length, units)

        # Keep the batch dimension on the first axis
        attention_weights_as_output = tf.transpose(
            tf.stack(layer_attention_weights), (1, 0, 2, 3, 4)
        )

        # (batch_size, length, units),
        # (batch_size, num_layers, num_heads, length, length)
        return x, attention_weights_as_output
