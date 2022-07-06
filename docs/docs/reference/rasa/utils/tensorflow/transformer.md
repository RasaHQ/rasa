---
sidebar_label: rasa.utils.tensorflow.transformer
title: rasa.utils.tensorflow.transformer
---
## MultiHeadAttention Objects

```python
class MultiHeadAttention(tf.keras.layers.Layer)
```

Multi-headed attention layer.

**Arguments**:

- `units` - Positive integer, output dim of hidden layer.
- `num_heads` - Positive integer, number of heads
  to repeat the same attention structure.
- `attention_dropout_rate` - Float, dropout rate inside attention for training.
- `density` - Approximate fraction of trainable weights (in
  `RandomlyConnectedDense` layers).
- `unidirectional` - Boolean, use a unidirectional or bidirectional encoder.
- `use_key_relative_position` - Boolean, if &#x27;True&#x27; use key
  relative embeddings in attention.
- `use_value_relative_position` - Boolean, if &#x27;True&#x27; use value
  relative embeddings in attention.
- `max_relative_position` - Positive integer, max position for relative embeddings.
- `heads_share_relative_embedding` - Boolean, if &#x27;True&#x27;
  heads will share relative embeddings.

#### call

```python
def call(query_input: tf.Tensor, source_input: tf.Tensor, pad_mask: Optional[tf.Tensor] = None, training: Optional[Union[tf.Tensor, bool]] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Apply attention mechanism to query_input and source_input.

**Arguments**:

- `query_input` - A tensor with shape [batch_size, length, input_size].
- `source_input` - A tensor with shape [batch_size, length, input_size].
- `pad_mask` - Float tensor with shape broadcastable
  to (..., length, length). Defaults to None.
- `training` - A bool, whether in training mode or not.
  

**Returns**:

  Attention layer output with shape [batch_size, length, units]

## TransformerEncoderLayer Objects

```python
class TransformerEncoderLayer(tf.keras.layers.Layer)
```

Transformer encoder layer.

The layer is composed of the sublayers:
1. Self-attention layer
2. Feed-forward network (which is 2 fully-connected layers)

**Arguments**:

- `units` - Positive integer, output dim of hidden layer.
- `num_heads` - Positive integer, number of heads
  to repeat the same attention structure.
- `filter_units` - Positive integer, output dim of the first ffn hidden layer.
- `dropout_rate` - Float between 0 and 1; fraction of the input units to drop.
- `attention_dropout_rate` - Float, dropout rate inside attention for training.
- `density` - Fraction of trainable weights in `RandomlyConnectedDense` layers.
- `unidirectional` - Boolean, use a unidirectional or bidirectional encoder.
- `use_key_relative_position` - Boolean, if &#x27;True&#x27; use key
  relative embeddings in attention.
- `use_value_relative_position` - Boolean, if &#x27;True&#x27; use value
  relative embeddings in attention.
- `max_relative_position` - Positive integer, max position for relative embeddings.
- `heads_share_relative_embedding` - Boolean, if &#x27;True&#x27;
  heads will share relative embeddings.

#### call

```python
def call(x: tf.Tensor, pad_mask: Optional[tf.Tensor] = None, training: Optional[Union[tf.Tensor, bool]] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Apply transformer encoder layer.

**Arguments**:

- `x` - A tensor with shape [batch_size, length, units].
- `pad_mask` - Float tensor with shape broadcastable
  to (..., length, length). Defaults to None.
- `training` - A bool, whether in training mode or not.
  

**Returns**:

  Transformer encoder layer output with shape [batch_size, length, units]

## TransformerEncoder Objects

```python
class TransformerEncoder(tf.keras.layers.Layer)
```

Transformer encoder.

Encoder stack is made up of `num_layers` identical encoder layers.

**Arguments**:

- `num_layers` - Positive integer, number of encoder layers.
- `units` - Positive integer, output dim of hidden layer.
- `num_heads` - Positive integer, number of heads
  to repeat the same attention structure.
- `filter_units` - Positive integer, output dim of the first ffn hidden layer.
- `reg_lambda` - Float, regularization factor.
- `dropout_rate` - Float between 0 and 1; fraction of the input units to drop.
- `attention_dropout_rate` - Float, dropout rate inside attention for training.
- `density` - Approximate fraction of trainable weights (in
  `RandomlyConnectedDense` layers).
- `unidirectional` - Boolean, use a unidirectional or bidirectional encoder.
- `use_key_relative_position` - Boolean, if &#x27;True&#x27; use key
  relative embeddings in attention.
- `use_value_relative_position` - Boolean, if &#x27;True&#x27; use value
  relative embeddings in attention.
- `max_relative_position` - Positive integer, max position for relative embeddings.
- `heads_share_relative_embedding` - Boolean, if &#x27;True&#x27;
  heads will share relative embeddings.
- `name` - Optional name of the layer.

#### call

```python
def call(x: tf.Tensor, pad_mask: Optional[tf.Tensor] = None, training: Optional[Union[tf.Tensor, bool]] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Apply transformer encoder.

**Arguments**:

- `x` - A tensor with shape [batch_size, length, input_size].
- `pad_mask` - Float tensor with shape broadcastable
  to (..., length, length). Defaults to None.
- `training` - A bool, whether in training mode or not.
  

**Returns**:

  Transformer encoder output with shape [batch_size, length, units]

