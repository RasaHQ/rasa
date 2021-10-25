---
sidebar_label: rasa.utils.tensorflow.crf
title: rasa.utils.tensorflow.crf
---
## CrfDecodeForwardRnnCell Objects

```python
class CrfDecodeForwardRnnCell(tf.keras.layers.AbstractRNNCell)
```

Computes the forward decoding in a linear-chain CRF.

#### \_\_init\_\_

```python
 | @typechecked
 | __init__(transition_params: TensorLike, **kwargs: Any) -> None
```

Initialize the CrfDecodeForwardRnnCell.

**Arguments**:

- `transition_params` - A [num_tags, num_tags] matrix of binary
  potentials. This matrix is expanded into a
  [1, num_tags, num_tags] in preparation for the broadcast
  summation occurring within the cell.

#### output\_size

```python
 | @property
 | output_size() -> int
```

Returns count of tags.

#### build

```python
 | build(input_shape: Union[TensorShape, List[TensorShape]]) -> None
```

Creates the variables of the layer.

#### call

```python
 | call(inputs: TensorLike, state: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]
```

Build the CrfDecodeForwardRnnCell.

**Arguments**:

- `inputs` - A [batch_size, num_tags] matrix of unary potentials.
- `state` - A [batch_size, num_tags] matrix containing the previous step&#x27;s
  score values.
  

**Returns**:

- `output` - A [batch_size, num_tags * 2] matrix of backpointers and scores.
- `new_state` - A [batch_size, num_tags] matrix of new score values.

#### crf\_decode\_forward

```python
crf_decode_forward(inputs: TensorLike, state: TensorLike, transition_params: TensorLike, sequence_lengths: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]
```

Computes forward decoding in a linear-chain CRF.

**Arguments**:

- `inputs` - A [batch_size, num_tags] matrix of unary potentials.
- `state` - A [batch_size, num_tags] matrix containing the previous step&#x27;s
  score values.
- `transition_params` - A [num_tags, num_tags] matrix of binary potentials.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
  

**Returns**:

- `output` - A [batch_size, num_tags * 2] matrix of backpointers and scores.
- `new_state` - A [batch_size, num_tags] matrix of new score values.

#### crf\_decode\_backward

```python
crf_decode_backward(backpointers: TensorLike, scores: TensorLike, state: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]
```

Computes backward decoding in a linear-chain CRF.

**Arguments**:

- `backpointers` - A [batch_size, num_tags] matrix of backpointer of next step
  (in time order).
- `scores` - A [batch_size, num_tags] matrix of scores of next step (in time order).
- `state` - A [batch_size, 1] matrix of tag index of next step.
  

**Returns**:

- `new_tags` - A [batch_size, num_tags] tensor containing the new tag indices.
- `new_scores` - A [batch_size, num_tags] tensor containing the new score values.

#### crf\_decode

```python
crf_decode(potentials: TensorLike, transition_params: TensorLike, sequence_length: TensorLike) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
```

Decode the highest scoring sequence of tags.

**Arguments**:

- `potentials` - A [batch_size, max_seq_len, num_tags] tensor of
  unary potentials.
- `transition_params` - A [num_tags, num_tags] matrix of
  binary potentials.
- `sequence_length` - A [batch_size] vector of true sequence lengths.
  

**Returns**:

- `decode_tags` - A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
  Contains the highest scoring tag indices.
- `decode_scores` - A [batch_size, max_seq_len] matrix, containing the score of
  `decode_tags`.
- `best_score` - A [batch_size] vector, containing the best score of `decode_tags`.

