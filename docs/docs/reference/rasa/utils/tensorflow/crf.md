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
def __init__(transition_params: TensorLike, **kwargs: Any) -> None
```

Initialize the CrfDecodeForwardRnnCell.

**Arguments**:

- `transition_params` - A [num_tags, num_tags] matrix of binary
  potentials. This matrix is expanded into a
  [1, num_tags, num_tags] in preparation for the broadcast
  summation occurring within the cell.

#### output\_size

```python
@property
def output_size() -> int
```

Returns count of tags.

#### build

```python
def build(input_shape: Union[TensorShape, List[TensorShape]]) -> None
```

Creates the variables of the layer.

#### call

```python
def call(inputs: TensorLike, state: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]
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
def crf_decode_forward(
        inputs: TensorLike, state: TensorLike, transition_params: TensorLike,
        sequence_lengths: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]
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
def crf_decode_backward(backpointers: TensorLike, scores: TensorLike,
                        state: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]
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
def crf_decode(
        potentials: TensorLike, transition_params: TensorLike,
        sequence_length: TensorLike) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
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

#### crf\_unary\_score

```python
def crf_unary_score(tag_indices: TensorLike, sequence_lengths: TensorLike,
                    inputs: TensorLike) -> tf.Tensor
```

Computes the unary scores of tag sequences.

**Arguments**:

- `tag_indices` - A [batch_size, max_seq_len] matrix of tag indices.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
- `inputs` - A [batch_size, max_seq_len, num_tags] tensor of unary potentials.

**Returns**:

- `unary_scores` - A [batch_size] vector of unary scores.

#### crf\_binary\_score

```python
def crf_binary_score(tag_indices: TensorLike, sequence_lengths: TensorLike,
                     transition_params: TensorLike) -> tf.Tensor
```

Computes the binary scores of tag sequences.

**Arguments**:

- `tag_indices` - A [batch_size, max_seq_len] matrix of tag indices.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
- `transition_params` - A [num_tags, num_tags] matrix of binary potentials.

**Returns**:

- `binary_scores` - A [batch_size] vector of binary scores.

#### crf\_sequence\_score

```python
def crf_sequence_score(inputs: TensorLike, tag_indices: TensorLike,
                       sequence_lengths: TensorLike,
                       transition_params: TensorLike) -> tf.Tensor
```

Computes the unnormalized score for a tag sequence.

**Arguments**:

- `inputs` - A [batch_size, max_seq_len, num_tags] tensor of unary potentials
  to use as input to the CRF layer.
- `tag_indices` - A [batch_size, max_seq_len] matrix of tag indices for which
  we compute the unnormalized score.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
- `transition_params` - A [num_tags, num_tags] transition matrix.

**Returns**:

- `sequence_scores` - A [batch_size] vector of unnormalized sequence scores.

#### crf\_forward

```python
def crf_forward(inputs: TensorLike, state: TensorLike,
                transition_params: TensorLike,
                sequence_lengths: TensorLike) -> tf.Tensor
```

Computes the alpha values in a linear-chain CRF.

See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.

**Arguments**:

- `inputs` - A [batch_size, num_tags] matrix of unary potentials.
- `state` - A [batch_size, num_tags] matrix containing the previous alpha
  values.
- `transition_params` - A [num_tags, num_tags] matrix of binary potentials.
  This matrix is expanded into a [1, num_tags, num_tags] in preparation
  for the broadcast summation occurring within the cell.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
  

**Returns**:

- `new_alphas` - A [batch_size, num_tags] matrix containing the
  new alpha values.

#### crf\_log\_norm

```python
def crf_log_norm(inputs: TensorLike, sequence_lengths: TensorLike,
                 transition_params: TensorLike) -> tf.Tensor
```

Computes the normalization for a CRF.

**Arguments**:

- `inputs` - A [batch_size, max_seq_len, num_tags] tensor of unary potentials
  to use as input to the CRF layer.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
- `transition_params` - A [num_tags, num_tags] transition matrix.

**Returns**:

- `log_norm` - A [batch_size] vector of normalizers for a CRF.

#### crf\_log\_likelihood

```python
def crf_log_likelihood(
    inputs: TensorLike,
    tag_indices: TensorLike,
    sequence_lengths: TensorLike,
    transition_params: Optional[TensorLike] = None
) -> Tuple[tf.Tensor, tf.Tensor]
```

Computes the log-likelihood of tag sequences in a CRF.

**Arguments**:

- `inputs` - A [batch_size, max_seq_len, num_tags] tensor of unary potentials
  to use as input to the CRF layer.
- `tag_indices` - A [batch_size, max_seq_len] matrix of tag indices for which
  we compute the log-likelihood.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
- `transition_params` - A [num_tags, num_tags] transition matrix,
  if available.

**Returns**:

- `log_likelihood` - A [batch_size] `Tensor` containing the log-likelihood of
  each example, given the sequence of tag indices.
- `transition_params` - A [num_tags, num_tags] transition matrix. This is
  either provided by the caller or created in this function.

